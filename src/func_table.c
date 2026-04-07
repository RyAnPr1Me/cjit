/**
 * func_table.c – Atomic function-pointer table implementation.
 *
 * See func_table.h for design notes and thread-safety guarantees.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "func_table.h"

#include <stdlib.h>   /* malloc / free     */
#include <string.h>   /* strncpy / memset  */
#include <stdio.h>    /* fprintf           */

/* ─────────────────────────── helpers ──────────────────────────────────────── */

static void entry_init(func_table_entry_t *e, func_id_t id)
{
    /*
     * Zero-initialise everything first so that atomic fields start in a
     * well-defined state (all bits zero is a valid value for every atomic
     * type used here on all platforms we target).
     */
    memset(e, 0, sizeof(*e));

    atomic_init(&e->func_ptr,   NULL);
    atomic_init(&e->call_cnt,   0);
    atomic_init(&e->version,    0);
    atomic_init(&e->cur_level,  (int)OPT_NONE);
    atomic_init(&e->in_queue,   false);
    atomic_init(&e->last_compile_duration_ms, 0);

    e->id        = id;
    e->dl_handle = NULL;
    e->ir_source = NULL;

    pthread_mutex_init(&e->compile_lock, NULL);
}

/* ─────────────────────────── public API ─────────────────────────────────────── */

func_table_t *func_table_create(uint32_t capacity)
{
    if (capacity == 0 || capacity > CJIT_MAX_FUNCTIONS) return NULL;

    func_table_t *ft = malloc(sizeof(*ft));
    if (!ft) return NULL;

    ft->entries = calloc(capacity, sizeof(func_table_entry_t));
    if (!ft->entries) { free(ft); return NULL; }

    ft->capacity = capacity;
    atomic_init(&ft->count, 0);

    for (uint32_t i = 0; i < capacity; ++i) {
        entry_init(&ft->entries[i], i);
    }

    return ft;
}

void func_table_destroy(func_table_t *ft)
{
    if (!ft) return;

    uint32_t n = atomic_load_explicit(&ft->count, memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        pthread_mutex_destroy(&ft->entries[i].compile_lock);
    }
    free(ft->entries);
    free(ft);
}

func_id_t func_table_register(func_table_t *ft,
                               const char   *name,
                               const char   *ir_source,
                               jit_func_t    initial_fn)
{
    uint32_t idx = atomic_load_explicit(&ft->count, memory_order_relaxed);
    if (idx >= ft->capacity) {
        fprintf(stderr, "[cjit] func_table_register: table full (%u entries)\n",
                ft->capacity);
        return CJIT_INVALID_FUNC_ID;
    }

    func_table_entry_t *e = &ft->entries[idx];
    entry_init(e, (func_id_t)idx);

    strncpy(e->name, name, CJIT_NAME_MAX - 1);
    e->name[CJIT_NAME_MAX - 1] = '\0';
    e->ir_source = ir_source;

    /*
     * Install the AOT fallback as the initial function pointer.
     * Release ordering so that a subsequent acquire-load by another thread
     * observes the fully initialised entry.
     */
    atomic_store_explicit(&e->func_ptr,  initial_fn, memory_order_release);
    atomic_store_explicit(&e->cur_level, (int)OPT_NONE, memory_order_relaxed);

    /* Commit the registration by incrementing the count. */
    atomic_fetch_add_explicit(&ft->count, 1, memory_order_release);

    return (func_id_t)idx;
}

func_table_entry_t *func_table_get(func_table_t *ft, func_id_t id)
{
    uint32_t n = atomic_load_explicit(&ft->count, memory_order_acquire);
    if (id >= n) return NULL;
    return &ft->entries[id];
}

void *func_table_swap(func_table_t *ft,
                      func_id_t     id,
                      jit_func_t    new_fn,
                      void         *new_handle,
                      opt_level_t   new_level)
{
    func_table_entry_t *e = func_table_get(ft, id);
    if (!e) return NULL;

    /*
     * Capture the old dlopen handle BEFORE the atomic store so that we can
     * retire it safely.  The compile_lock is held by the calling compiler
     * thread, preventing another compiler thread from simultaneously
     * overwriting dl_handle.
     */
    void *old_handle = e->dl_handle;
    e->dl_handle = new_handle;

    /*
     * The atomic store uses memory_order_release so that all writes
     * (including e->dl_handle = new_handle above) are visible to any thread
     * that subsequently performs an acquire-load of func_ptr.
     *
     * Runtime threads read func_ptr with memory_order_acquire, so they will
     * observe the updated dl_handle assignment as well.  That's not strictly
     * necessary for correctness (runtime threads never dereference dl_handle),
     * but it makes the happens-before relationship clear.
     */
    atomic_store_explicit(&e->func_ptr,  new_fn,             memory_order_release);
    atomic_store_explicit(&e->cur_level, (int)new_level,     memory_order_relaxed);
    atomic_fetch_add_explicit(&e->version, 1,                memory_order_relaxed);

    return old_handle;
}
