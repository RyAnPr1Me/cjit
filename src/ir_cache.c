/**
 * ir_cache.c – Multi-generation LRU IR cache implementation.
 *
 * See ir_cache.h for the full design description.
 *
 * Internal data-structure notes
 * ─────────────────────────────
 * Each generation is a doubly-linked intrusive list stored inside the
 * ir_node_t array.  The head pointer is the MRU (most-recently-used) end;
 * the tail pointer is the LRU (eviction) end.
 *
 *   head ←→ [newer] ←→ … ←→ [older] ←→ tail
 *   MRU                                 LRU
 *
 * The three list-manipulation helpers are:
 *   list_remove()     – unlink a node from whatever list it is currently in.
 *   list_push_front() – insert at MRU end (access / promotion).
 *   list_pop_back()   – remove from LRU end (eviction).
 *
 * All list operations and generation transitions are performed under
 * `cache->lock`.  The lock is lightweight (never held during file I/O or
 * malloc, those happen outside the critical section).
 *
 * Disk I/O is done while the lock is NOT held, using a local copy of the
 * disk path.  This prevents the cache mutex from becoming a bottleneck when
 * a cold function must be loaded from disk.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ir_cache.h"

#include <stdlib.h>    /* malloc / free / calloc  */
#include <string.h>    /* memset / strncpy / snprintf */
#include <stdio.h>     /* FILE / fopen / fclose / fprintf */
#include <time.h>      /* clock_gettime */
#include <sys/stat.h>  /* mkdir */
#include <unistd.h>    /* getpid */
#include <errno.h>

/* ═══════════════════════════ internal helpers ═════════════════════════════ */

static uint64_t irc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/* ── doubly-linked list helpers ───────────────────────────────────────────
 * All three helpers assume the caller holds cache->lock.                  */

static void list_remove(ir_node_t **head, ir_node_t **tail, ir_node_t *node)
{
    if (node->prev) node->prev->next = node->next;
    else            *head            = node->next;
    if (node->next) node->next->prev = node->prev;
    else            *tail            = node->prev;
    node->prev = node->next = NULL;
}

static void list_push_front(ir_node_t **head, ir_node_t **tail, ir_node_t *node)
{
    node->next = *head;
    node->prev = NULL;
    if (*head) (*head)->prev = node;
    else       *tail         = node;
    *head = node;
}

/** Remove and return the tail (LRU) node, or NULL if the list is empty. */
static ir_node_t *list_pop_back(ir_node_t **head, ir_node_t **tail)
{
    ir_node_t *node = *tail;
    if (!node) return NULL;
    list_remove(head, tail, node);
    return node;
}

/* ── disk helpers ─────────────────────────────────────────────────────────
 * These are called WITHOUT holding the cache lock.                        */

static bool irc_write_to_disk(const char *path, const char *ir)
{
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fputs(ir, f);
    fclose(f);
    return true;
}

/** Read IR from disk.  Returns malloc-allocated string; caller must free(). */
static char *irc_read_from_disk(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t nr = fread(buf, 1, (size_t)sz, f);
    buf[nr] = '\0';
    fclose(f);
    return buf;
}

/** Sanitise a function name so it is safe to use as a filename component. */
static void sanitise_name(char *dst, const char *src, size_t dstsz)
{
    size_t i;
    for (i = 0; i + 1 < dstsz && src[i]; ++i) {
        char c = src[i];
        dst[i] = ((c >= 'a' && c <= 'z') ||
                  (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') ||
                   c == '_') ? c : '_';
    }
    dst[i] = '\0';
}

/* ── capacity-enforcement helpers (called under lock) ─────────────────────
 *
 * make_room_in_hot:
 *   If hot_count == hot_capacity, demote the LRU-HOT node to WARM.
 *   If that causes warm overflow, cascade by demoting LRU-WARM to COLD.
 *
 * make_room_in_warm:
 *   If warm_count == warm_capacity, demote the LRU-WARM node to COLD.   */

static void demote_warm_to_cold(ir_lru_cache_t *cache, ir_node_t *node)
{
    /* Remove from warm list. */
    list_remove(&cache->warm_head, &cache->warm_tail, node);
    --cache->warm_count;

    /* Free the in-memory IR string; the disk copy remains. */
    free(node->ir_source);
    node->ir_source = NULL;
    node->gen = IR_GEN_COLD;
    ++cache->cold_count;

    atomic_fetch_add_explicit(&cache->stat_evictions, 1, memory_order_relaxed);
}

static void make_room_in_warm(ir_lru_cache_t *cache)
{
    if (cache->warm_count < cache->warm_capacity) return;
    ir_node_t *lru = list_pop_back(&cache->warm_head, &cache->warm_tail);
    if (!lru) return;
    --cache->warm_count;
    free(lru->ir_source);
    lru->ir_source = NULL;
    lru->gen = IR_GEN_COLD;
    ++cache->cold_count;
    atomic_fetch_add_explicit(&cache->stat_evictions, 1, memory_order_relaxed);
}

static void make_room_in_hot(ir_lru_cache_t *cache)
{
    if (cache->hot_count < cache->hot_capacity) return;
    /* Demote LRU-HOT to WARM (IR string stays alive). */
    ir_node_t *lru = list_pop_back(&cache->hot_head, &cache->hot_tail);
    if (!lru) return;
    --cache->hot_count;
    lru->gen = IR_GEN_WARM;

    /* Make room in warm if necessary before inserting. */
    make_room_in_warm(cache);

    list_push_front(&cache->warm_head, &cache->warm_tail, lru);
    ++cache->warm_count;
    atomic_fetch_add_explicit(&cache->stat_evictions, 1, memory_order_relaxed);
}

/* ═══════════════════════════ public API ═══════════════════════════════════ */

ir_lru_cache_t *ir_cache_create(uint32_t    max_funcs,
                                 uint32_t    hot_cap,
                                 uint32_t    warm_cap,
                                 const char *ir_dir)
{
    if (!max_funcs) return NULL;

    ir_lru_cache_t *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;

    cache->nodes = calloc(max_funcs, sizeof(ir_node_t));
    if (!cache->nodes) { free(cache); return NULL; }

    cache->max_funcs      = max_funcs;
    cache->hot_capacity   = hot_cap  ? hot_cap  : 64;
    cache->warm_capacity  = warm_cap ? warm_cap : 128;

    pthread_mutex_init(&cache->lock, NULL);

    atomic_init(&cache->stat_disk_writes,  0);
    atomic_init(&cache->stat_disk_reads,   0);
    atomic_init(&cache->stat_evictions,    0);
    atomic_init(&cache->stat_promotions,   0);
    atomic_init(&cache->stat_cache_hits,   0);
    atomic_init(&cache->stat_cache_misses, 0);

    /* Build IR directory path. */
    if (ir_dir && ir_dir[0]) {
        snprintf(cache->ir_dir, sizeof(cache->ir_dir), "%s", ir_dir);
    } else {
        snprintf(cache->ir_dir, sizeof(cache->ir_dir),
                 "/tmp/cjit_ir_%d", (int)getpid());
    }

    /* Create the directory (ignore EEXIST). */
    if (mkdir(cache->ir_dir, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr,
                "[ir_cache] WARNING: cannot create IR dir '%s': %s\n",
                cache->ir_dir, strerror(errno));
        /* Non-fatal: compilation will still work if disk writes fail. */
    }

    return cache;
}

void ir_cache_destroy(ir_lru_cache_t *cache)
{
    if (!cache) return;

    /* Free all heap-allocated IR strings. */
    for (uint32_t i = 0; i < cache->total_registered; ++i) {
        free(cache->nodes[i].ir_source);
    }
    free(cache->nodes);
    pthread_mutex_destroy(&cache->lock);
    free(cache);
}

bool ir_cache_register(ir_lru_cache_t *cache,
                        func_id_t       func_id,
                        const char     *func_name,
                        const char     *ir_source)
{
    if (!cache || !func_name || !ir_source) return false;
    if (func_id >= cache->max_funcs) return false;

    ir_node_t *node = &cache->nodes[func_id];
    if (node->registered) return false; /* already registered */

    /* ── Initialise node fields (outside lock; no list links yet) ─────── */
    node->func_id = func_id;
    sanitise_name(node->name, func_name,  sizeof(node->name));

    /* Build the disk path. */
    snprintf(node->disk_path, sizeof(node->disk_path),
             "%s/%u_%s.ir", cache->ir_dir, func_id, node->name);

    /* Make a heap copy of the IR for the in-memory side. */
    node->ir_source = strdup(ir_source);
    if (!node->ir_source) return false;

    node->last_access_ms = irc_now_ms();
    node->access_cnt     = 0;
    node->registered     = true;

    /* ── Write IR to disk unconditionally (permanent backup) ──────────── */
    /* Done outside the lock; no other thread touches this node yet. */
    bool wrote = irc_write_to_disk(node->disk_path, ir_source);
    if (wrote) {
        atomic_fetch_add_explicit(&cache->stat_disk_writes, 1,
                                   memory_order_relaxed);
    } else {
        fprintf(stderr,
                "[ir_cache] WARNING: could not write IR for '%s' to '%s'\n",
                func_name, node->disk_path);
        /* Non-fatal: in-memory copy is still valid. */
    }

    /* ── Insert into HOT generation (under lock) ──────────────────────── */
    pthread_mutex_lock(&cache->lock);

    /* Enforce capacity: may cascade HOT→WARM→COLD evictions. */
    make_room_in_hot(cache);

    node->gen = IR_GEN_HOT;
    list_push_front(&cache->hot_head, &cache->hot_tail, node);
    ++cache->hot_count;
    ++cache->total_registered;

    pthread_mutex_unlock(&cache->lock);
    return true;
}

char *ir_cache_get_ir(ir_lru_cache_t *cache, func_id_t func_id)
{
    if (!cache || func_id >= cache->max_funcs) return NULL;

    ir_node_t *node = &cache->nodes[func_id];
    if (!node->registered) return NULL;

    /* ── Fast path: node is in memory ─────────────────────────────────── */
    pthread_mutex_lock(&cache->lock);

    if (node->gen == IR_GEN_HOT) {
        /* Move to MRU position in HOT (no gen change, just refresh). */
        list_remove(&cache->hot_head, &cache->hot_tail, node);
        list_push_front(&cache->hot_head, &cache->hot_tail, node);
        node->last_access_ms = irc_now_ms();
        ++node->access_cnt;
        char *copy = node->ir_source ? strdup(node->ir_source) : NULL;
        pthread_mutex_unlock(&cache->lock);
        atomic_fetch_add_explicit(&cache->stat_cache_hits, 1, memory_order_relaxed);
        return copy;
    }

    if (node->gen == IR_GEN_WARM) {
        /*
         * Promote WARM → HOT.
         * 1. Remove from warm list.
         * 2. Make room in hot (may demote LRU-hot to warm).
         * 3. Push to hot MRU.
         */
        list_remove(&cache->warm_head, &cache->warm_tail, node);
        --cache->warm_count;

        make_room_in_hot(cache);

        node->gen = IR_GEN_HOT;
        list_push_front(&cache->hot_head, &cache->hot_tail, node);
        ++cache->hot_count;
        node->last_access_ms = irc_now_ms();
        ++node->access_cnt;
        char *copy = node->ir_source ? strdup(node->ir_source) : NULL;
        pthread_mutex_unlock(&cache->lock);
        atomic_fetch_add_explicit(&cache->stat_cache_hits,  1, memory_order_relaxed);
        atomic_fetch_add_explicit(&cache->stat_promotions,  1, memory_order_relaxed);
        return copy;
    }

    /* ── Slow path: COLD – must load from disk ─────────────────────────── */
    /*
     * Copy the disk path under lock, then release the lock while doing I/O
     * so other threads are not blocked during the file read.
     */
    char disk_path_copy[sizeof(node->disk_path)];
    memcpy(disk_path_copy, node->disk_path, sizeof(disk_path_copy));
    pthread_mutex_unlock(&cache->lock);

    atomic_fetch_add_explicit(&cache->stat_cache_misses, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->stat_disk_reads,   1, memory_order_relaxed);

    /* Load from disk (no lock held during I/O). */
    char *loaded = irc_read_from_disk(disk_path_copy);
    if (!loaded) {
        fprintf(stderr,
                "[ir_cache] ERROR: failed to read IR for func %u from '%s'\n",
                func_id, disk_path_copy);
        return NULL;
    }

    /* ── Re-acquire lock and promote COLD → WARM ──────────────────────── */
    pthread_mutex_lock(&cache->lock);

    /*
     * Re-check: another thread may have already loaded and promoted this node
     * while we were doing file I/O.  If so, discard our freshly loaded copy
     * and return a duplicate of the in-memory one.
     */
    if (node->gen != IR_GEN_COLD) {
        char *copy = node->ir_source ? strdup(node->ir_source) : strdup(loaded);
        pthread_mutex_unlock(&cache->lock);
        free(loaded);
        atomic_fetch_add_explicit(&cache->stat_cache_hits, 1, memory_order_relaxed);
        return copy;
    }

    /* Install the loaded IR into the node. */
    node->ir_source = loaded;

    /* Make room in WARM (may cascade LRU-WARM→COLD). */
    make_room_in_warm(cache);

    node->gen = IR_GEN_WARM;
    list_push_front(&cache->warm_head, &cache->warm_tail, node);
    ++cache->warm_count;
    --cache->cold_count;
    node->last_access_ms = irc_now_ms();
    ++node->access_cnt;

    char *copy = strdup(loaded); /* return a copy; node keeps its own */
    pthread_mutex_unlock(&cache->lock);

    atomic_fetch_add_explicit(&cache->stat_promotions, 1, memory_order_relaxed);
    return copy;
}

ir_gen_t ir_cache_get_generation(const ir_lru_cache_t *cache, func_id_t func_id)
{
    if (!cache || func_id >= cache->max_funcs) return IR_GEN_COLD;
    if (!cache->nodes[func_id].registered)     return IR_GEN_COLD;
    return cache->nodes[func_id].gen;
}

ir_cache_stats_t ir_cache_get_stats(const ir_lru_cache_t *cache)
{
    ir_cache_stats_t s;
    memset(&s, 0, sizeof(s));
    if (!cache) return s;

    /* hot/warm/cold counts require the lock for a consistent snapshot;
     * for a diagnostic stats call we accept a slightly racy read.        */
    s.hot_count        = cache->hot_count;
    s.warm_count       = cache->warm_count;
    s.cold_count       = cache->cold_count;
    s.total_registered = cache->total_registered;

    s.disk_writes  = atomic_load_explicit(&cache->stat_disk_writes,  memory_order_relaxed);
    s.disk_reads   = atomic_load_explicit(&cache->stat_disk_reads,   memory_order_relaxed);
    s.evictions    = atomic_load_explicit(&cache->stat_evictions,    memory_order_relaxed);
    s.promotions   = atomic_load_explicit(&cache->stat_promotions,   memory_order_relaxed);
    s.cache_hits   = atomic_load_explicit(&cache->stat_cache_hits,   memory_order_relaxed);
    s.cache_misses = atomic_load_explicit(&cache->stat_cache_misses, memory_order_relaxed);
    return s;
}

void ir_cache_print_stats(const ir_lru_cache_t *cache)
{
    ir_cache_stats_t s = ir_cache_get_stats(cache);
    fprintf(stderr,
            "╔══════════════════════════════════════╗\n"
            "║     IR LRU Cache Statistics           ║\n"
            "╠══════════════════════════════════════╣\n"
            "║  Registered functions  : %6u        ║\n"
            "║  HOT  (in memory)      : %6u        ║\n"
            "║  WARM (in memory)      : %6u        ║\n"
            "║  COLD (on disk)        : %6u        ║\n"
            "╠══════════════════════════════════════╣\n"
            "║  Cache hits            : %6llu        ║\n"
            "║  Cache misses (disk)   : %6llu        ║\n"
            "║  Disk writes           : %6llu        ║\n"
            "║  Disk reads            : %6llu        ║\n"
            "║  Evictions (→disk)     : %6llu        ║\n"
            "║  Promotions (←disk)    : %6llu        ║\n"
            "╚══════════════════════════════════════╝\n",
            s.total_registered,
            s.hot_count, s.warm_count, s.cold_count,
            (unsigned long long)s.cache_hits,
            (unsigned long long)s.cache_misses,
            (unsigned long long)s.disk_writes,
            (unsigned long long)s.disk_reads,
            (unsigned long long)s.evictions,
            (unsigned long long)s.promotions);
}
