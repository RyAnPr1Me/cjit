/**
 * ir_cache.c – Multi-generation LRU IR cache with memory-pressure awareness.
 *
 * See ir_cache.h for the full design description.
 *
 * Key implementation notes
 * ────────────────────────
 * • All LRU list mutations (list_remove, list_push_front, list_pop_back) are
 *   performed under cache->lock.  Disk I/O and malloc/free happen outside
 *   the lock.
 *
 * • The pressure-monitor thread atomically updates cache->pressure and the
 *   mem_available_kb / mem_total_kb snapshots.  It calls
 *   irc_trim_to_pressure() when pressure rises, which re-acquires the lock
 *   to evict entries.
 *
 * • make_room_in_hot() and make_room_in_warm() consult the current atomic
 *   pressure level to compute the effective capacity, so new insertions
 *   automatically respect tightened limits even between pressure-thread wakes.
 *
 * • The COLD→WARM promotion in ir_cache_get_ir() drops the lock during disk
 *   I/O to avoid blocking other threads.  It re-checks the generation after
 *   re-acquiring the lock to handle the (rare) race where two threads load
 *   the same COLD entry simultaneously.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ir_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

/* IR generation tags (mirrored as macros for readability) */
#define IR_GEN_HOT  ((uint8_t)0)
#define IR_GEN_WARM ((uint8_t)1)
#define IR_GEN_COLD ((uint8_t)2)

/* ═══════════════════════════ small helpers ════════════════════════════════ */

static uint64_t irc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/** Replace any character that is not alphanumeric / '_' with '_'. */
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

/**
 * Reconstruct the on-disk IR path for a node into out[sz].
 *
 * The path is not stored in ir_node_t (saving 512 bytes per node).
 * Instead it is computed on demand from the two stable, write-once fields:
 * node->func_id and node->name.  This function is only called during disk
 * I/O (registration write, COLD promotion read), so the snprintf overhead
 * is negligible.
 *
 * Format: "<cache->ir_dir>/<func_id>_<name>.ir"
 */
static void node_disk_path(const ir_lru_cache_t *cache, const ir_node_t *node,
                            char *out, size_t sz)
{
    snprintf(out, sz, "%s/%u_%s.ir", cache->ir_dir, node->func_id, node->name);
}

/*
 * Maximum length of a reconstructed on-disk IR path.
 *
 * Components:
 *   sizeof(ir_dir)  — base directory (up to 256 chars including NUL)
 *   + 1             — path separator '/'
 *   + 10            — decimal representation of func_id (uint32_t max = 10 digits)
 *   + 1             — '_' separator between func_id and name
 *   + CJIT_NAME_MAX — sanitised function name (NUL included)
 *   + 3             — ".ir" extension (no NUL; already covered by CJIT_NAME_MAX)
 *
 * Total = 256 + 1 + 10 + 1 + CJIT_NAME_MAX + 3 = 271 + CJIT_NAME_MAX.
 * A value of 400 is used to provide a comfortable margin and round up to a
 * convenient size.
 */
#define IRC_PATH_MAX 400u

/* ═══════════════════════════ /proc/meminfo reader ═════════════════════════ */

/**
 * Read MemTotal and MemAvailable from /proc/meminfo.
 *
 * Returns true if both fields were parsed successfully.
 * Falls back gracefully when /proc/meminfo is absent (non-Linux platforms).
 */
static bool irc_read_meminfo(uint64_t *total_kb, uint64_t *avail_kb)
{
    *total_kb = *avail_kb = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return false;

    char line[128];
    int  found = 0;
    while (found < 2 && fgets(line, sizeof(line), f)) {
        unsigned long long v;
        if (sscanf(line, "MemTotal: %llu kB",     &v) == 1) { *total_kb = v; ++found; }
        else if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) { *avail_kb = v; ++found; }
    }
    fclose(f);
    return *total_kb > 0 && *avail_kb > 0;
}

/* ═══════════════════════════ pressure helpers ═════════════════════════════ */

/**
 * Compute pressure level from a memory snapshot.
 * Returns MEM_PRESSURE_NORMAL if total_kb is 0 (i.e. /proc/meminfo unavailable).
 */
static mem_pressure_t irc_calc_pressure(uint64_t avail_kb, uint64_t total_kb,
                                         uint32_t low_pct,
                                         uint32_t high_pct,
                                         uint32_t critical_pct)
{
    if (total_kb == 0) return MEM_PRESSURE_NORMAL;
    uint32_t avail_pct = (uint32_t)((avail_kb * 100ULL) / total_kb);
    if (avail_pct < critical_pct) return MEM_PRESSURE_CRITICAL;
    if (avail_pct < high_pct)     return MEM_PRESSURE_HIGH;
    if (avail_pct < low_pct)      return MEM_PRESSURE_MEDIUM;
    return MEM_PRESSURE_NORMAL;
}

/**
 * Compute the effective (pressure-adjusted) capacity for a given nominal cap.
 *
 * Scaling factors:
 *   NORMAL   → 100%   (no reduction)
 *   MEDIUM   →  75%
 *   HIGH     →  50%
 *   CRITICAL →  25%   (but never below 1)
 */
static uint32_t irc_effective_cap(uint32_t base, mem_pressure_t p)
{
    switch (p) {
    case MEM_PRESSURE_NORMAL:   return base;
    case MEM_PRESSURE_MEDIUM:   return base * 3 / 4;
    case MEM_PRESSURE_HIGH:     return base / 2;
    case MEM_PRESSURE_CRITICAL: return base / 4 + 1;
    }
    return base;
}

/* Convenience: read current pressure from the atomic field. */
static inline mem_pressure_t irc_current_pressure(const ir_lru_cache_t *cache)
{
    return (mem_pressure_t)atomic_load_explicit(&cache->pressure,
                                                 memory_order_relaxed);
}

/* ═══════════════════════════ LRU list helpers (lock held) ═════════════════ */

static void list_remove(ir_node_t **head, ir_node_t **tail, ir_node_t *node)
{
    if (node->prev) node->prev->next = node->next; else *head = node->next;
    if (node->next) node->next->prev = node->prev; else *tail = node->prev;
    node->prev = node->next = NULL;
}

static void list_push_front(ir_node_t **head, ir_node_t **tail, ir_node_t *node)
{
    node->next = *head;
    node->prev = NULL;
    if (*head) (*head)->prev = node; else *tail = node;
    *head = node;
}

static ir_node_t *list_pop_back(ir_node_t **head, ir_node_t **tail)
{
    ir_node_t *node = *tail;
    if (!node) return NULL;
    list_remove(head, tail, node);
    return node;
}

/* ═══════════════════════════ eviction (lock held) ═════════════════════════ */

/**
 * Demote the LRU-WARM entry to COLD: free its IR string, move it off the
 * WARM list.  Increments stat_evictions.  Called under cache->lock.
 */
static void irc_evict_lru_warm(ir_lru_cache_t *cache)
{
    ir_node_t *lru = list_pop_back(&cache->warm_head, &cache->warm_tail);
    if (!lru) return;
    --cache->warm_count;
    free(lru->ir_source);
    lru->ir_source = NULL;
    lru->gen = IR_GEN_COLD;
    ++cache->cold_count;
    atomic_fetch_add_explicit(&cache->stat_evictions, 1, memory_order_relaxed);
}

/**
 * Ensure warm_count < effective warm capacity.
 * Called under cache->lock.
 */
static void make_room_in_warm(ir_lru_cache_t *cache)
{
    uint32_t eff = irc_effective_cap(cache->warm_capacity,
                                     irc_current_pressure(cache));
    while (cache->warm_count >= eff)
        irc_evict_lru_warm(cache);
}

/**
 * Ensure hot_count < effective hot capacity.
 * Demotes LRU-HOT to WARM (cascading into COLD if warm is also full).
 * Called under cache->lock.
 */
static void make_room_in_hot(ir_lru_cache_t *cache)
{
    uint32_t eff_hot = irc_effective_cap(cache->hot_capacity,
                                          irc_current_pressure(cache));
    while (cache->hot_count >= eff_hot) {
        ir_node_t *lru = list_pop_back(&cache->hot_head, &cache->hot_tail);
        if (!lru) break;
        --cache->hot_count;
        lru->gen = IR_GEN_WARM;

        /* Make room in warm before inserting. */
        make_room_in_warm(cache);

        list_push_front(&cache->warm_head, &cache->warm_tail, lru);
        ++cache->warm_count;
        atomic_fetch_add_explicit(&cache->stat_evictions, 1, memory_order_relaxed);
    }
}

/* ═══════════════════════════ pressure trim ════════════════════════════════ */

/**
 * Proactively trim HOT and WARM lists to the effective capacities for the
 * given pressure level.  Called by the pressure thread when pressure rises.
 *
 * Strategy:
 *   1. Trim WARM→COLD first to make space for displaced HOT entries.
 *   2. Trim HOT→WARM (or directly→COLD if warm is still full).
 *
 * Each eviction increments both stat_evictions and stat_pressure_evictions.
 */
static void irc_trim_to_pressure(ir_lru_cache_t *cache, mem_pressure_t p)
{
    uint32_t eff_hot  = irc_effective_cap(cache->hot_capacity,  p);
    uint32_t eff_warm = irc_effective_cap(cache->warm_capacity, p);

    pthread_mutex_lock(&cache->lock);

    /* Step 1: shrink WARM → COLD */
    while (cache->warm_count > eff_warm) {
        ir_node_t *lru = list_pop_back(&cache->warm_head, &cache->warm_tail);
        if (!lru) break;
        --cache->warm_count;
        free(lru->ir_source);
        lru->ir_source = NULL;
        lru->gen = IR_GEN_COLD;
        ++cache->cold_count;
        atomic_fetch_add_explicit(&cache->stat_evictions,          1, memory_order_relaxed);
        atomic_fetch_add_explicit(&cache->stat_pressure_evictions, 1, memory_order_relaxed);
    }

    /* Step 2: shrink HOT → WARM (or directly → COLD when warm also full) */
    while (cache->hot_count > eff_hot) {
        ir_node_t *lru = list_pop_back(&cache->hot_head, &cache->hot_tail);
        if (!lru) break;
        --cache->hot_count;

        if (cache->warm_count < eff_warm) {
            lru->gen = IR_GEN_WARM;
            list_push_front(&cache->warm_head, &cache->warm_tail, lru);
            ++cache->warm_count;
        } else {
            free(lru->ir_source);
            lru->ir_source = NULL;
            lru->gen = IR_GEN_COLD;
            ++cache->cold_count;
        }
        atomic_fetch_add_explicit(&cache->stat_evictions,          1, memory_order_relaxed);
        atomic_fetch_add_explicit(&cache->stat_pressure_evictions, 1, memory_order_relaxed);
    }

    pthread_mutex_unlock(&cache->lock);
}

/* ═══════════════════════════ pressure-monitor thread ══════════════════════ */

static void *pressure_monitor_fn(void *arg)
{
    ir_lru_cache_t *cache = (ir_lru_cache_t *)arg;

    struct timespec sleep_ts;
    sleep_ts.tv_sec  = cache->mem_check_interval_ms / 1000;
    sleep_ts.tv_nsec = (long)(cache->mem_check_interval_ms % 1000) * 1000000L;

    mem_pressure_t prev_pressure = MEM_PRESSURE_NORMAL;

    while (!atomic_load_explicit(&cache->stop_pressure_flag, memory_order_acquire)) {
        nanosleep(&sleep_ts, NULL);

        uint64_t total_kb = 0, avail_kb = 0;
        if (!irc_read_meminfo(&total_kb, &avail_kb)) {
            /* /proc/meminfo unreadable (non-Linux); nothing to do. */
            continue;
        }

        /* Update observable snapshots atomically. */
        atomic_store_explicit(&cache->mem_total_kb,     total_kb, memory_order_relaxed);
        atomic_store_explicit(&cache->mem_available_kb, avail_kb, memory_order_relaxed);

        mem_pressure_t new_p = irc_calc_pressure(avail_kb, total_kb,
                                                   cache->mem_low_pct,
                                                   cache->mem_high_pct,
                                                   cache->mem_critical_pct);

        mem_pressure_t old_p = (mem_pressure_t)
            atomic_exchange_explicit(&cache->pressure, (int)new_p,
                                     memory_order_acq_rel);

        if (new_p > old_p) {
            /*
             * Pressure increased: proactively evict entries to stay within
             * the tighter effective capacities.  This happens in the
             * background and does not block runtime threads.
             */
            irc_trim_to_pressure(cache, new_p);
        }

        if (new_p != prev_pressure) {
            static const char *const pnames[] =
                { "NORMAL", "MEDIUM", "HIGH", "CRITICAL" };
            fprintf(stderr,
                    "[ir_cache/pressure] %s → %s  "
                    "(avail=%llu MB / total=%llu MB)\n",
                    pnames[prev_pressure], pnames[new_p],
                    (unsigned long long)(avail_kb / 1024),
                    (unsigned long long)(total_kb / 1024));
            prev_pressure = new_p;
        }
    }

    return NULL;
}

/* ═══════════════════════════ disk helpers ═════════════════════════════════ */

static bool irc_write_to_disk(const char *path, const char *ir)
{
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fputs(ir, f);
    fclose(f);
    return true;
}

/** Read entire file into a malloc'd NUL-terminated buffer. Caller frees. */
static char *irc_read_from_disk(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0)                      { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf)                         { fclose(f); return NULL; }
    size_t nr = fread(buf, 1, (size_t)sz, f);
    buf[nr] = '\0';
    fclose(f);
    return buf;
}

/* ═══════════════════════════ async I/O thread pool ════════════════════════ */

/**
 * Main loop of an async I/O prefetch thread.
 *
 * Blocks on the prefetch condvar when the queue is empty.  On each wake,
 * dequeues one func_id and calls ir_cache_get_ir() which handles the
 * COLD→WARM promotion (disk read + LRU insertion) with all proper locking.
 * The returned IR copy is freed immediately — the goal is only to warm the
 * cache, not to use the IR ourselves.
 */
static void *io_thread_fn(void *arg)
{
    ir_lru_cache_t *cache = (ir_lru_cache_t *)arg;

    /*
     * Keep the mutex held across the stop check and the wait so there is no
     * window between "queue empty" and "pthread_cond_wait" where a broadcast
     * from ir_cache_destroy() could be missed.
     *
     * Protocol:
     *   1. Lock mutex.
     *   2. While queue empty AND not stopping: wait on condvar (atomically
     *      releases mutex and sleeps; re-acquires on wake).
     *   3. If stopping: unlock and exit.
     *   4. Dequeue one item, unlock, do I/O, lock again.
     */
    pthread_mutex_lock(&cache->pf_mutex);
    for (;;) {
        /* Wait until there is work or we are asked to stop. */
        while (cache->pf_count == 0 &&
               !atomic_load_explicit(&cache->stop_io_flag, memory_order_relaxed))
            pthread_cond_wait(&cache->pf_cond, &cache->pf_mutex);

        if (atomic_load_explicit(&cache->stop_io_flag, memory_order_relaxed))
            break;

        /* Dequeue one request while still holding the mutex. */
        func_id_t id    = cache->pf_buf[cache->pf_head];
        cache->pf_head  = (cache->pf_head + 1) % IRC_PREFETCH_CAP;
        --cache->pf_count;
        pthread_mutex_unlock(&cache->pf_mutex);

        /*
         * ir_cache_get_ir handles COLD→WARM: drops cache->lock during disk I/O
         * so other threads are not blocked.  For HOT/WARM entries the call is
         * a cheap in-memory hit.  The returned copy is freed immediately; we
         * only care about the side-effect of warming the cache entry.
         */
        char *ir = ir_cache_get_ir(cache, id);
        free(ir);

        pthread_mutex_lock(&cache->pf_mutex);
    }
    pthread_mutex_unlock(&cache->pf_mutex);
    return NULL;
}

bool ir_cache_prefetch(ir_lru_cache_t *cache, func_id_t func_id)
{
    if (!cache || !cache->num_io_threads) return false;
    if (func_id >= cache->max_funcs)       return false;

    pthread_mutex_lock(&cache->pf_mutex);

    if (cache->pf_count >= IRC_PREFETCH_CAP) {
        pthread_mutex_unlock(&cache->pf_mutex);
        return false;  /* queue full; caller falls back to synchronous load */
    }

    uint32_t tail       = (cache->pf_head + cache->pf_count) % IRC_PREFETCH_CAP;
    cache->pf_buf[tail] = func_id;
    ++cache->pf_count;

    pthread_cond_signal(&cache->pf_cond);  /* wake one I/O thread */
    pthread_mutex_unlock(&cache->pf_mutex);
    return true;
}



ir_lru_cache_t *ir_cache_create(const ir_cache_config_t *cfg)
{
    if (!cfg || !cfg->max_funcs) return NULL;

    ir_lru_cache_t *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;

    cache->nodes = calloc(cfg->max_funcs, sizeof(ir_node_t));
    if (!cache->nodes) { free(cache); return NULL; }

    cache->max_funcs     = cfg->max_funcs;
    cache->hot_capacity  = cfg->hot_cap  ? cfg->hot_cap  : 64;
    cache->warm_capacity = cfg->warm_cap ? cfg->warm_cap : 128;

    pthread_mutex_init(&cache->lock, NULL);

    /* Memory-pressure config (apply defaults for zero values). */
    cache->mem_check_interval_ms = cfg->mem_check_interval_ms
                                   ? cfg->mem_check_interval_ms : 500;
    cache->mem_low_pct      = cfg->mem_low_pct      ? cfg->mem_low_pct      : 20;
    cache->mem_high_pct     = cfg->mem_high_pct     ? cfg->mem_high_pct     : 10;
    cache->mem_critical_pct = cfg->mem_critical_pct ? cfg->mem_critical_pct :  5;

    atomic_init(&cache->pressure,             (int)MEM_PRESSURE_NORMAL);
    atomic_init(&cache->mem_available_kb,     0);
    atomic_init(&cache->mem_total_kb,         0);
    atomic_init(&cache->stop_pressure_flag,   false);
    atomic_init(&cache->stat_disk_writes,     0);
    atomic_init(&cache->stat_disk_reads,      0);
    atomic_init(&cache->stat_evictions,       0);
    atomic_init(&cache->stat_promotions,      0);
    atomic_init(&cache->stat_cache_hits,      0);
    atomic_init(&cache->stat_cache_misses,    0);
    atomic_init(&cache->stat_pressure_evictions, 0);

    atomic_init(&cache->stop_io_flag,    false);

    /* Build IR directory path. */
    if (cfg->ir_dir && cfg->ir_dir[0]) {
        snprintf(cache->ir_dir, sizeof(cache->ir_dir), "%s", cfg->ir_dir);
    } else {
        snprintf(cache->ir_dir, sizeof(cache->ir_dir),
                 "/tmp/cjit_ir_%d", (int)getpid());
    }

    if (mkdir(cache->ir_dir, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "[ir_cache] WARNING: cannot create IR dir '%s': %s\n",
                cache->ir_dir, strerror(errno));
    }

    /* I/O prefetch pool – initialise FIFO and start threads. */
    pthread_mutex_init(&cache->pf_mutex, NULL);
    pthread_cond_init(&cache->pf_cond, NULL);
    cache->pf_head = cache->pf_count = 0;

    cache->num_io_threads = cfg->num_io_threads ? cfg->num_io_threads : 2;
    cache->io_threads = calloc(cache->num_io_threads, sizeof(pthread_t));
    if (!cache->io_threads) {
        /* Non-fatal: degrade gracefully to no async prefetch. */
        cache->num_io_threads = 0;
        fprintf(stderr, "[ir_cache] WARNING: cannot allocate I/O thread array\n");
    } else {
        for (uint32_t i = 0; i < cache->num_io_threads; ++i)
            pthread_create(&cache->io_threads[i], NULL, io_thread_fn, cache);
    }

    /* Start the pressure-monitor thread immediately so the first
     * compilation already benefits from an accurate pressure reading.   */
    pthread_create(&cache->pressure_thread, NULL, pressure_monitor_fn, cache);

    return cache;
}

void ir_cache_destroy(ir_lru_cache_t *cache)
{
    if (!cache) return;

    /* Stop I/O threads.
     *
     * Setting stop_io_flag must happen INSIDE the mutex so there is no window
     * between an I/O thread's stop_io_flag check and its pthread_cond_wait
     * call where the flag could be set and the broadcast missed.  If the flag
     * were set outside the mutex, an I/O thread that evaluated "!stop_io_flag"
     * as false, then yielded before calling pthread_cond_wait, would sleep
     * indefinitely after missing the broadcast.
     */
    if (cache->num_io_threads > 0) {
        pthread_mutex_lock(&cache->pf_mutex);
        atomic_store_explicit(&cache->stop_io_flag, true, memory_order_relaxed);
        pthread_cond_broadcast(&cache->pf_cond);
        pthread_mutex_unlock(&cache->pf_mutex);
        for (uint32_t i = 0; i < cache->num_io_threads; ++i)
            pthread_join(cache->io_threads[i], NULL);
        free(cache->io_threads);
        cache->io_threads = NULL;
    }
    pthread_cond_destroy(&cache->pf_cond);
    pthread_mutex_destroy(&cache->pf_mutex);

    /* Stop pressure thread. */
    atomic_store_explicit(&cache->stop_pressure_flag, true, memory_order_release);
    pthread_join(cache->pressure_thread, NULL);

    /* Free all heap IR strings. */
    for (uint32_t i = 0; i < cache->total_registered; ++i)
        free(cache->nodes[i].ir_source);

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
    if (func_id >= cache->max_funcs)        return false;

    ir_node_t *node = &cache->nodes[func_id];
    if (node->registered) return false;

    /* Initialise fields outside the lock (node not yet visible to others). */
    node->func_id = func_id;
    sanitise_name(node->name, func_name, sizeof(node->name));

    node->ir_source      = strdup(ir_source);
    if (!node->ir_source) return false;
    node->last_access_ms = irc_now_ms();
    node->access_cnt     = 0;
    node->registered     = true;

    /* Write permanent backup to disk (outside lock; I/O can be slow). */
    char disk_path[IRC_PATH_MAX];
    node_disk_path(cache, node, disk_path, sizeof(disk_path));
    bool wrote = irc_write_to_disk(disk_path, ir_source);
    if (wrote)
        atomic_fetch_add_explicit(&cache->stat_disk_writes, 1, memory_order_relaxed);
    else
        fprintf(stderr, "[ir_cache] WARNING: cannot write IR for '%s' to '%s'\n",
                func_name, disk_path);

    /* Insert into HOT generation under lock. */
    pthread_mutex_lock(&cache->lock);
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

    pthread_mutex_lock(&cache->lock);

    /* ── HOT: just refresh MRU position ─────────────────────────────── */
    if (node->gen == IR_GEN_HOT) {
        list_remove(&cache->hot_head, &cache->hot_tail, node);
        list_push_front(&cache->hot_head, &cache->hot_tail, node);
        node->last_access_ms = irc_now_ms();
        ++node->access_cnt;
        char *copy = node->ir_source ? strdup(node->ir_source) : NULL;
        pthread_mutex_unlock(&cache->lock);
        atomic_fetch_add_explicit(&cache->stat_cache_hits, 1, memory_order_relaxed);
        return copy;
    }

    /* ── WARM: promote to HOT ────────────────────────────────────────── */
    if (node->gen == IR_GEN_WARM) {
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

    /* ── COLD: load from disk (drop lock during I/O) ─────────────────── */
    /*
     * func_id and name are write-once at registration; they are stable and
     * safe to read without the lock.  Reconstruct the path here so we avoid
     * storing 512 bytes of disk_path per node.
     */
    char disk_path[IRC_PATH_MAX];
    node_disk_path(cache, node, disk_path, sizeof(disk_path));
    pthread_mutex_unlock(&cache->lock);

    atomic_fetch_add_explicit(&cache->stat_cache_misses, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&cache->stat_disk_reads,   1, memory_order_relaxed);

    char *loaded = irc_read_from_disk(disk_path);
    if (!loaded) {
        fprintf(stderr, "[ir_cache] ERROR: failed to read IR for func %u from '%s'\n",
                func_id, disk_path);
        return NULL;
    }

    /* Re-acquire lock; re-check generation (another thread may have loaded). */
    pthread_mutex_lock(&cache->lock);

    if (node->gen != IR_GEN_COLD) {
        /* Already promoted by a concurrent thread; return their in-memory copy. */
        char *copy = node->ir_source ? strdup(node->ir_source) : strdup(loaded);
        pthread_mutex_unlock(&cache->lock);
        free(loaded);
        atomic_fetch_add_explicit(&cache->stat_cache_hits, 1, memory_order_relaxed);
        return copy;
    }

    node->ir_source = loaded;   /* install loaded string */
    make_room_in_warm(cache);
    node->gen = IR_GEN_WARM;
    list_push_front(&cache->warm_head, &cache->warm_tail, node);
    ++cache->warm_count;
    --cache->cold_count;
    node->last_access_ms = irc_now_ms();
    ++node->access_cnt;

    char *copy = strdup(loaded);
    pthread_mutex_unlock(&cache->lock);

    atomic_fetch_add_explicit(&cache->stat_promotions, 1, memory_order_relaxed);
    return copy;
}

uint8_t ir_cache_get_generation(const ir_lru_cache_t *cache, func_id_t func_id)
{
    if (!cache || func_id >= cache->max_funcs)  return IR_GEN_COLD;
    if (!cache->nodes[func_id].registered)       return IR_GEN_COLD;
    return cache->nodes[func_id].gen;
}

bool ir_cache_update_ir(ir_lru_cache_t *cache,
                         func_id_t       func_id,
                         const char     *func_name,
                         const char     *new_ir)
{
    (void)func_name;   /* Reserved for future logging; name is in node->name */
    if (!cache || !new_ir || func_id >= cache->max_funcs) return false;

    ir_node_t *node = &cache->nodes[func_id];
    if (!node->registered) return false;

    /* Allocate the new heap copy outside the lock. */
    char *new_copy = strdup(new_ir);
    if (!new_copy) return false;

    /* Write the updated IR to the on-disk backup (outside lock; I/O is slow). */
    char disk_path[IRC_PATH_MAX];
    node_disk_path(cache, node, disk_path, sizeof(disk_path));
    if (irc_write_to_disk(disk_path, new_ir))
        atomic_fetch_add_explicit(&cache->stat_disk_writes, 1, memory_order_relaxed);
    else
        fprintf(stderr, "[ir_cache] WARNING: cannot update IR on disk for func %u\n",
                func_id);

    pthread_mutex_lock(&cache->lock);

    char *old_copy = node->ir_source;   /* may be NULL when COLD */

    if (node->gen == IR_GEN_COLD) {
        /*
         * Promote the entry from COLD to WARM so the next compilation
         * gets the new IR from memory rather than reading from disk.
         */
        make_room_in_warm(cache);
        node->ir_source = new_copy;
        node->gen = IR_GEN_WARM;
        list_push_front(&cache->warm_head, &cache->warm_tail, node);
        ++cache->warm_count;
        --cache->cold_count;
        atomic_fetch_add_explicit(&cache->stat_promotions, 1, memory_order_relaxed);
        old_copy = NULL;   /* was NULL for COLD entries; nothing to free */
    } else {
        /* HOT or WARM: replace the in-memory IR string in-place. */
        node->ir_source = new_copy;
    }

    node->access_cnt++;
    node->last_access_ms = irc_now_ms();

    pthread_mutex_unlock(&cache->lock);

    free(old_copy);   /* free outside the lock; free(NULL) is safe */
    return true;
}

mem_pressure_t ir_cache_get_pressure(const ir_lru_cache_t *cache)
{
    if (!cache) return MEM_PRESSURE_NORMAL;
    return (mem_pressure_t)atomic_load_explicit(&cache->pressure,
                                                 memory_order_relaxed);
}

ir_cache_stats_t ir_cache_get_stats(const ir_lru_cache_t *cache)
{
    ir_cache_stats_t s;
    memset(&s, 0, sizeof(s));
    if (!cache) return s;

    /* hot/warm/cold counts: racy but acceptable for diagnostics */
    s.hot_count        = cache->hot_count;
    s.warm_count       = cache->warm_count;
    s.cold_count       = cache->cold_count;
    s.total_registered = cache->total_registered;

    s.disk_writes          = atomic_load_explicit(&cache->stat_disk_writes,         memory_order_relaxed);
    s.disk_reads           = atomic_load_explicit(&cache->stat_disk_reads,          memory_order_relaxed);
    s.evictions            = atomic_load_explicit(&cache->stat_evictions,           memory_order_relaxed);
    s.promotions           = atomic_load_explicit(&cache->stat_promotions,          memory_order_relaxed);
    s.cache_hits           = atomic_load_explicit(&cache->stat_cache_hits,          memory_order_relaxed);
    s.cache_misses         = atomic_load_explicit(&cache->stat_cache_misses,        memory_order_relaxed);
    s.pressure_evictions   = atomic_load_explicit(&cache->stat_pressure_evictions,  memory_order_relaxed);
    s.pressure             = (mem_pressure_t)atomic_load_explicit(&cache->pressure, memory_order_relaxed);

    uint64_t avail_kb = atomic_load_explicit(&cache->mem_available_kb, memory_order_relaxed);
    uint64_t total_kb = atomic_load_explicit(&cache->mem_total_kb,     memory_order_relaxed);
    s.mem_available_mb = avail_kb / 1024;
    s.mem_total_mb     = total_kb / 1024;
    return s;
}

void ir_cache_print_stats(const ir_lru_cache_t *cache)
{
    ir_cache_stats_t s = ir_cache_get_stats(cache);
    static const char *const pnames[] = { "NORMAL", "MEDIUM", "HIGH", "CRITICAL" };
    fprintf(stderr,
            "╔══════════════════════════════════════════╗\n"
            "║      IR LRU Cache + Memory Pressure       ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Memory pressure  : %-8s              ║\n"
            "║  Mem available    : %6llu MB              ║\n"
            "║  Mem total        : %6llu MB              ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Registered       : %6u                ║\n"
            "║  HOT  (in memory) : %6u                ║\n"
            "║  WARM (in memory) : %6u                ║\n"
            "║  COLD (on disk)   : %6u                ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Cache hits       : %6llu                ║\n"
            "║  Cache misses     : %6llu                ║\n"
            "║  Disk writes      : %6llu                ║\n"
            "║  Disk reads       : %6llu                ║\n"
            "║  LRU evictions    : %6llu                ║\n"
            "║  Pressure evicts  : %6llu                ║\n"
            "║  Promotions       : %6llu                ║\n"
            "╚══════════════════════════════════════════╝\n",
            pnames[s.pressure],
            (unsigned long long)s.mem_available_mb,
            (unsigned long long)s.mem_total_mb,
            s.total_registered,
            s.hot_count, s.warm_count, s.cold_count,
            (unsigned long long)s.cache_hits,
            (unsigned long long)s.cache_misses,
            (unsigned long long)s.disk_writes,
            (unsigned long long)s.disk_reads,
            (unsigned long long)s.evictions,
            (unsigned long long)s.pressure_evictions,
            (unsigned long long)s.promotions);
}
