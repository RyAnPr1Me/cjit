/**
 * cjit.c – Non-blocking JIT compiler engine.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                       Architecture Overview                             │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │                                                                         │
 * │   Runtime threads (any number)                                          │
 * │   ────────────────────────────                                          │
 * │   Hot path per call:                                                    │
 * │     fn = atomic_load(&table[id].func_ptr);  ← single acquire-load      │
 * │     fn(args…);                              ← indirect call             │
 * │     atomic_fetch_add(&table[id].call_cnt);  ← relaxed increment        │
 * │                                                                         │
 * │   Background threads (started by cjit_start)                           │
 * │   ─────────────────────────────────────────                             │
 * │                                                                         │
 * │   ┌──────────────────────────────────────────────────┐                 │
 * │   │  Monitor thread (×1)                             │                 │
 * │   │   • Wakes every monitor_interval_ms              │                 │
 * │   │   • Scans call_cnt of all registered functions   │                 │
 * │   │   • Detects hot functions (cnt ≥ hot_threshold)  │                 │
 * │   │   • Enqueues compile_task onto MPMC work-queue   │                 │
 * │   └──────────────────────────┬───────────────────────┘                 │
 * │                              │  (lock-free MPMC queue)                 │
 * │   ┌──────────────────────────▼───────────────────────┐                 │
 * │   │  Compiler threads (×3, CJIT_COMPILER_THREADS)   │                 │
 * │   │   • Spin-sleep waiting for compile_task          │                 │
 * │   │   • Call codegen_compile(ir_source, level, …)    │                 │
 * │   │   • On success: func_table_swap(new_fn, handle)  │                 │
 * │   │   • Retire old handle via dgc_retire()           │                 │
 * │   └──────────────────────────┬───────────────────────┘                 │
 * │                              │  (lock-free retire stack)               │
 * │   ┌──────────────────────────▼───────────────────────┐                 │
 * │   │  GC thread (×1, inside deferred_gc)              │                 │
 * │   │   • Wakes every sweep_interval_ms                │                 │
 * │   │   • dlclose handles older than grace_period_ms   │                 │
 * │   └──────────────────────────────────────────────────┘                 │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Thread safety
 * ─────────────
 * func_ptr  : Written with memory_order_release (compiler thread only).
 *             Read  with memory_order_acquire   (runtime + monitor threads).
 *
 * call_cnt  : Incremented with memory_order_relaxed (runtime threads).
 *             Read    with memory_order_relaxed (monitor thread).
 *             Relaxed ordering is correct here because we only care about
 *             the approximate magnitude, not strict ordering with any
 *             other shared variable.
 *
 * compile_lock per entry: Held by a compiler thread during the entire
 *             compilation pass.  Prevents two compiler threads from
 *             simultaneously recompiling the same function.  This mutex
 *             is NEVER held on the runtime call path.
 *
 * in_queue   : Atomic bool set by monitor/request before enqueue, cleared
 *             by compiler thread after dequeue (but before compilation so
 *             that a new enqueue can happen if needed while this compile
 *             is in flight).
 *
 * Global stats counters: updated with relaxed atomics; no ordering required.
 *
 * RCU-style atomic swap
 * ─────────────────────
 *   1. Compiler thread compiles new code → new_fn, new_handle.
 *   2. Acquires entry->compile_lock.
 *   3. old_handle = entry->dl_handle.
 *   4. entry->dl_handle = new_handle.
 *   5. atomic_store(&entry->func_ptr, new_fn, release).
 *      ↑ This is the "pointer publish" – all subsequent acquire-loads by
 *        runtime threads will see new_fn.  Any runtime thread that already
 *        loaded old_fn before this store will finish its call using old code
 *        (which is still mapped, since dlclose is deferred).
 *   6. Releases compile_lock.
 *   7. dgc_retire(dgc, old_handle).
 *      ↑ After grace_period_ms, the GC thread calls dlclose(old_handle).
 *        By that time all calls using old code have completed.
 *
 * Incremental recompilation
 * ─────────────────────────
 * The same IR source can be recompiled multiple times at increasing
 * optimisation tiers.  The monitor thread escalates:
 *   call_cnt ≥ hot_threshold_t1  → enqueue at OPT_O2
 *   call_cnt ≥ hot_threshold_t2  → enqueue at OPT_O3
 *
 * A version counter per entry is incremented on each swap.  Stale compile
 * tasks (whose version_req < current version) are discarded so that the
 * compiler threads do not perform redundant work.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../include/cjit.h"
#include "work_queue.h"
#include "deferred_gc.h"
#include "func_table.h"
#include "codegen.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>

/* ══════════════════════════ engine structure ══════════════════════════════ */

struct cjit_engine {
    /* Configuration (immutable after cjit_start). */
    cjit_config_t   cfg;

    /* Function table: the heart of the engine. */
    func_table_t   *ftable;

    /* Single compile work-queue shared by all compiler threads. */
    mpmc_queue_t    work_queue;

    /* Deferred GC for safe dlclose of retired handles. */
    deferred_gc_t   dgc;

    /* Background thread handles. */
    pthread_t       compiler_threads[CJIT_COMPILER_THREADS];
    pthread_t       monitor_thread;

    /* Lifecycle control. */
    atomic_bool     running;
    atomic_bool     stop_requested;

    /* Global statistics (relaxed atomics; no strict ordering needed). */
    atomic_uint_fast64_t stat_compilations;
    atomic_uint_fast64_t stat_failed;
    atomic_uint_fast64_t stat_swaps;
};

/* ══════════════════════════ compiler thread ═══════════════════════════════ */

/**
 * Main loop of a background compiler thread.
 *
 * The thread blocks in a tight spin-sleep on the MPMC queue.  When a task
 * arrives it attempts to compile the function's IR at the requested
 * optimisation level.
 *
 * Stale-task detection:
 *   If entry->version has advanced past task.version_req, this task was
 *   generated before the last successful compilation.  The entry is already
 *   at least as optimised as requested; discard and continue.
 *
 * Concurrent compilation prevention:
 *   entry->compile_lock is a per-entry mutex.  trylock() is used so that if
 *   another compiler thread raced ahead on the same function, this thread
 *   simply skips it (the racing thread will perform the compilation).
 */
static void *compiler_thread_fn(void *arg)
{
    cjit_engine_t *engine = (cjit_engine_t *)arg;

    /* Build codegen options once – they don't change after cjit_start(). */
    codegen_opts_t copts = {
        .enable_inlining      = engine->cfg.enable_inlining,
        .enable_vectorization = engine->cfg.enable_vectorization,
        .enable_loop_unroll   = engine->cfg.enable_loop_unroll,
        .enable_native_arch   = engine->cfg.enable_native_arch,
        .verbose              = engine->cfg.verbose,
    };

    /* Backoff sleep: starts at 1 ms, doubles up to 8 ms on empty queue. */
    const struct timespec backoff_min = { .tv_sec = 0, .tv_nsec =  500000L }; /* 0.5 ms */
    const struct timespec backoff_max = { .tv_sec = 0, .tv_nsec = 8000000L }; /* 8 ms   */
    struct timespec backoff = backoff_min;

    while (!atomic_load_explicit(&engine->stop_requested, memory_order_acquire)) {

        compile_task_t task;
        if (!mpmc_dequeue(&engine->work_queue, &task)) {
            /* Queue empty: sleep with exponential backoff. */
            nanosleep(&backoff, NULL);
            if (backoff.tv_nsec < backoff_max.tv_nsec)
                backoff.tv_nsec *= 2;
            continue;
        }
        backoff = backoff_min; /* Reset backoff on successful dequeue. */

        func_table_entry_t *entry = func_table_get(engine->ftable, task.func_id);
        if (!entry) continue;

        /*
         * Clear in_queue so the monitor thread can re-enqueue this function
         * later if it keeps getting hotter.
         * We clear it BEFORE compiling so that if compilation is slow, the
         * monitor can queue another pass.
         */
        atomic_store_explicit(&entry->in_queue, false, memory_order_relaxed);

        /* Stale-task check: discard if a newer compilation already happened. */
        uint32_t cur_ver = atomic_load_explicit(&entry->version,
                                                 memory_order_relaxed);
        if (cur_ver > task.version_req) {
            if (engine->cfg.verbose) {
                fprintf(stderr,
                        "[cjit/compiler] discard stale task for '%s' "
                        "(ver %u > req %u)\n",
                        entry->name, cur_ver, task.version_req);
            }
            continue;
        }

        /* Check current optimisation level: skip if already at target. */
        opt_level_t cur_level =
            (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                               memory_order_relaxed);
        if (cur_level >= task.target_level) {
            if (engine->cfg.verbose) {
                fprintf(stderr,
                        "[cjit/compiler] skip '%s': already at O%d >= O%d\n",
                        entry->name, (int)cur_level, (int)task.target_level);
            }
            continue;
        }

        /* Try to acquire the per-entry compile lock (non-blocking). */
        if (pthread_mutex_trylock(&entry->compile_lock) != 0) {
            /*
             * Another compiler thread is already compiling this function.
             * Re-enqueue the task so it is not lost, then move on.
             */
            mpmc_enqueue(&engine->work_queue, &task);
            continue;
        }

        /* ── Compilation ──────────────────────────────────────────────── */
        if (engine->cfg.verbose) {
            fprintf(stderr,
                    "[cjit/compiler] compiling '%s' at O%d (call_cnt=%llu)\n",
                    entry->name, (int)task.target_level,
                    (unsigned long long)
                    atomic_load_explicit(&entry->call_cnt, memory_order_relaxed));
        }

        codegen_result_t cres;
        bool ok = codegen_compile(entry->name, entry->ir_source,
                                   task.target_level, &copts, &cres);

        if (!ok) {
            atomic_fetch_add_explicit(&engine->stat_failed, 1,
                                      memory_order_relaxed);
            fprintf(stderr,
                    "[cjit/compiler] FAILED to compile '%s': %s\n",
                    entry->name, cres.errmsg);
            pthread_mutex_unlock(&entry->compile_lock);
            continue;
        }

        atomic_fetch_add_explicit(&engine->stat_compilations, 1,
                                  memory_order_relaxed);

        /* ── Atomic pointer swap ───────────────────────────────────────── */
        /*
         * func_table_swap():
         *   1. Saves entry->dl_handle (old handle).
         *   2. Sets entry->dl_handle = cres.handle.
         *   3. atomic_store(&entry->func_ptr, cres.fn, release).
         *   4. Increments entry->version.
         *   5. Returns old handle.
         *
         * After this store, all runtime threads that subsequently load
         * entry->func_ptr will get cres.fn.  Threads already inside the
         * old function continue executing old code safely because the old
         * shared object is kept alive by the dgc grace period.
         */
        void *old_handle = func_table_swap(engine->ftable, task.func_id,
                                            cres.fn, cres.handle,
                                            task.target_level);

        atomic_fetch_add_explicit(&engine->stat_swaps, 1,
                                  memory_order_relaxed);

        pthread_mutex_unlock(&entry->compile_lock);

        /*
         * Retire the old shared-object handle.
         * dgc_retire() is wait-free (single CAS loop) and does not block.
         * The GC thread will dlclose() it after grace_period_ms.
         */
        dgc_retire(&engine->dgc, old_handle);

        if (engine->cfg.verbose) {
            fprintf(stderr,
                    "[cjit/compiler] swapped '%s' → O%d (ver %lu)\n",
                    entry->name, (int)task.target_level,
                    (unsigned long)atomic_load_explicit(&entry->version, memory_order_relaxed));
        }
    }

    return NULL;
}

/* ══════════════════════════ monitor thread ════════════════════════════════ */

/**
 * Main loop of the hot-function monitor thread.
 *
 * Wakes every monitor_interval_ms and scans the function table looking for
 * functions whose call_cnt has crossed a hot threshold.  Qualifying functions
 * are enqueued for recompilation at the appropriate optimisation tier.
 *
 * The monitor thread does NOT read or write function pointers; it only reads
 * call_cnt (relaxed) and cur_level (relaxed), then enqueues compile tasks.
 *
 * Two tiers:
 *   T1 (OPT_O2): triggered when call_cnt ≥ hot_threshold_t1
 *   T2 (OPT_O3): triggered when call_cnt ≥ hot_threshold_t2
 *
 * De-duplication: in_queue is set atomically before enqueue to prevent
 * duplicate entries in the work queue.
 */
static void *monitor_thread_fn(void *arg)
{
    cjit_engine_t *engine = (cjit_engine_t *)arg;

    struct timespec interval;
    interval.tv_sec  = engine->cfg.monitor_interval_ms / 1000;
    interval.tv_nsec = (long)(engine->cfg.monitor_interval_ms % 1000) * 1000000L;

    while (!atomic_load_explicit(&engine->stop_requested, memory_order_acquire)) {
        nanosleep(&interval, NULL);

        uint32_t n = atomic_load_explicit(&engine->ftable->count,
                                           memory_order_acquire);

        for (uint32_t i = 0; i < n; ++i) {
            func_table_entry_t *entry = &engine->ftable->entries[i];

            uint64_t cnt = atomic_load_explicit(&entry->call_cnt,
                                                 memory_order_relaxed);

            /* Determine the target optimisation tier. */
            opt_level_t target;
            if (cnt >= engine->cfg.hot_threshold_t2)
                target = OPT_O3;
            else if (cnt >= engine->cfg.hot_threshold_t1)
                target = OPT_O2;
            else
                continue; /* Not hot enough yet. */

            /* Skip if already compiled at this tier. */
            opt_level_t cur =
                (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                                   memory_order_relaxed);
            if (cur >= target) continue;

            /*
             * Atomically set in_queue = true.
             * expected = false, desired = true.
             * If it was already true, another enqueue is already pending.
             */
            bool expected = false;
            if (!atomic_compare_exchange_strong_explicit(
                    &entry->in_queue, &expected, true,
                    memory_order_relaxed, memory_order_relaxed)) {
                continue; /* Already queued. */
            }

            compile_task_t task = {
                .func_id      = entry->id,
                .target_level = target,
                .priority     = (target == OPT_O3) ? 2 : 1,
                .version_req  = atomic_load_explicit(&entry->version,
                                                      memory_order_relaxed),
            };

            if (!mpmc_enqueue(&engine->work_queue, &task)) {
                /* Queue full: clear in_queue flag so we can retry later. */
                atomic_store_explicit(&entry->in_queue, false,
                                      memory_order_relaxed);
            } else if (engine->cfg.verbose) {
                fprintf(stderr,
                        "[cjit/monitor] enqueued '%s' for O%d "
                        "(call_cnt=%llu)\n",
                        entry->name, (int)target,
                        (unsigned long long)cnt);
            }
        }
    }

    return NULL;
}

/* ══════════════════════════ public API ════════════════════════════════════ */

cjit_config_t cjit_default_config(void)
{
    cjit_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_functions        = CJIT_MAX_FUNCTIONS;
    cfg.compiler_threads     = CJIT_COMPILER_THREADS;
    cfg.hot_threshold_t1     = CJIT_HOT_THRESHOLD_T1;
    cfg.hot_threshold_t2     = CJIT_HOT_THRESHOLD_T2;
    cfg.grace_period_ms      = CJIT_GRACE_PERIOD_MS;
    cfg.monitor_interval_ms  = 50;   /* Check for hot functions every 50 ms */
    cfg.enable_inlining      = true;
    cfg.enable_vectorization = true;
    cfg.enable_loop_unroll   = true;
    cfg.enable_const_fold    = true;
    cfg.enable_native_arch   = true;
    cfg.verbose              = false;
    return cfg;
}

cjit_engine_t *cjit_create(const cjit_config_t *config)
{
    cjit_engine_t *e = calloc(1, sizeof(*e));
    if (!e) return NULL;

    e->cfg = config ? *config : cjit_default_config();

    /* Clamp compiler_threads to the compile-time maximum. */
    if (e->cfg.compiler_threads > CJIT_COMPILER_THREADS)
        e->cfg.compiler_threads = CJIT_COMPILER_THREADS;
    if (e->cfg.compiler_threads == 0)
        e->cfg.compiler_threads = 1;

    e->ftable = func_table_create(e->cfg.max_functions);
    if (!e->ftable) { free(e); return NULL; }

    mpmc_init(&e->work_queue);
    dgc_init(&e->dgc, e->cfg.grace_period_ms);

    atomic_init(&e->running,          false);
    atomic_init(&e->stop_requested,   false);
    atomic_init(&e->stat_compilations, 0);
    atomic_init(&e->stat_failed,       0);
    atomic_init(&e->stat_swaps,        0);

    return e;
}

void cjit_start(cjit_engine_t *engine)
{
    if (atomic_load_explicit(&engine->running, memory_order_acquire)) return;

    atomic_store_explicit(&engine->stop_requested, false, memory_order_release);
    atomic_store_explicit(&engine->running,        true,  memory_order_release);

    /* Start the deferred-GC thread. */
    dgc_start(&engine->dgc);

    /* Start monitor thread. */
    pthread_create(&engine->monitor_thread, NULL, monitor_thread_fn, engine);

    /* Start background compiler threads. */
    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i) {
        pthread_create(&engine->compiler_threads[i], NULL,
                       compiler_thread_fn, engine);
    }
}

void cjit_stop(cjit_engine_t *engine)
{
    if (!atomic_load_explicit(&engine->running, memory_order_acquire)) return;

    atomic_store_explicit(&engine->stop_requested, true, memory_order_release);

    /* Wait for all threads to exit. */
    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i) {
        pthread_join(engine->compiler_threads[i], NULL);
    }
    pthread_join(engine->monitor_thread, NULL);

    /* Stop GC (this runs a final forced sweep to free all retired handles). */
    dgc_stop(&engine->dgc);

    atomic_store_explicit(&engine->running, false, memory_order_release);
}

void cjit_destroy(cjit_engine_t *engine)
{
    if (!engine) return;

    cjit_stop(engine);

    /*
     * dlclose all remaining handles that are currently active in the table.
     * (The deferred GC has already freed retired ones.)
     */
    uint32_t n = atomic_load_explicit(&engine->ftable->count,
                                       memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        func_table_entry_t *entry = &engine->ftable->entries[i];
        if (entry->dl_handle) {
            dlclose(entry->dl_handle);
            entry->dl_handle = NULL;
        }
    }

    func_table_destroy(engine->ftable);
    free(engine);
}

func_id_t cjit_register_function(cjit_engine_t *engine,
                                  const char    *name,
                                  const char    *ir_source,
                                  jit_func_t     aot_fallback)
{
    if (!engine || !name || !ir_source) return CJIT_INVALID_FUNC_ID;
    return func_table_register(engine->ftable, name, ir_source, aot_fallback);
}

/*
 * cjit_get_func – THE HOT PATH.
 *
 * This is a single atomic load.  On x86-64 with TSO memory model the acquire
 * ordering compiles to a plain MOV instruction (no fence needed because TSO
 * provides load-acquire for free).  On ARMv8 it compiles to LDAR.
 *
 * The returned function pointer is guaranteed valid for at least
 * grace_period_ms milliseconds even if the pointer is swapped concurrently.
 */
jit_func_t cjit_get_func(cjit_engine_t *engine, func_id_t id)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (__builtin_expect(!e, 0)) return NULL;
    return atomic_load_explicit(&e->func_ptr, memory_order_acquire);
}

void cjit_record_call(cjit_engine_t *engine, func_id_t id)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (__builtin_expect(!e, 0)) return;
    /*
     * Relaxed: we only care about the approximate magnitude of the counter.
     * No ordering with respect to any other variable is required.
     */
    atomic_fetch_add_explicit(&e->call_cnt, 1, memory_order_relaxed);
}

void cjit_request_recompile(cjit_engine_t *engine,
                             func_id_t      id,
                             opt_level_t    level)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return;

    /* Prevent duplicate enqueue. */
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(
            &e->in_queue, &expected, true,
            memory_order_relaxed, memory_order_relaxed)) {
        return; /* Already queued. */
    }

    compile_task_t task = {
        .func_id      = id,
        .target_level = level,
        .priority     = 3, /* manual request = highest priority */
        .version_req  = atomic_load_explicit(&e->version, memory_order_relaxed),
    };

    if (!mpmc_enqueue(&engine->work_queue, &task)) {
        /* Queue full; clear in_queue so it can be tried again. */
        atomic_store_explicit(&e->in_queue, false, memory_order_relaxed);
    }
}

cjit_stats_t cjit_get_stats(const cjit_engine_t *engine)
{
    cjit_stats_t s;
    memset(&s, 0, sizeof(s));
    if (!engine) return s;

    s.registered_functions = atomic_load_explicit(&engine->ftable->count,
                                                   memory_order_relaxed);
    s.total_compilations   = atomic_load_explicit(&engine->stat_compilations,
                                                   memory_order_relaxed);
    s.failed_compilations  = atomic_load_explicit(&engine->stat_failed,
                                                   memory_order_relaxed);
    s.total_swaps          = atomic_load_explicit(&engine->stat_swaps,
                                                   memory_order_relaxed);
    s.retired_handles      = atomic_load_explicit(&engine->dgc.total_retired,
                                                   memory_order_relaxed);
    s.freed_handles        = atomic_load_explicit(&engine->dgc.total_freed,
                                                   memory_order_relaxed);
    s.queue_depth          = mpmc_size(&engine->work_queue);
    return s;
}

void cjit_print_stats(const cjit_engine_t *engine)
{
    cjit_stats_t s = cjit_get_stats(engine);
    fprintf(stderr,
            "╔══════════════════════════════════════╗\n"
            "║          CJIT Engine Statistics       ║\n"
            "╠══════════════════════════════════════╣\n"
            "║  Registered functions  : %6u        ║\n"
            "║  Total compilations    : %6llu        ║\n"
            "║  Failed compilations   : %6llu        ║\n"
            "║  Atomic pointer swaps  : %6llu        ║\n"
            "║  Handles retired (GC)  : %6llu        ║\n"
            "║  Handles freed   (GC)  : %6llu        ║\n"
            "║  Work-queue depth now  : %6u        ║\n"
            "╚══════════════════════════════════════╝\n",
            s.registered_functions,
            (unsigned long long)s.total_compilations,
            (unsigned long long)s.failed_compilations,
            (unsigned long long)s.total_swaps,
            (unsigned long long)s.retired_handles,
            (unsigned long long)s.freed_handles,
            s.queue_depth);
}

uint64_t cjit_get_call_count(const cjit_engine_t *engine, func_id_t id)
{
    const func_table_entry_t *e =
        func_table_get((func_table_t *)engine->ftable, id);
    if (!e) return 0;
    return atomic_load_explicit(&e->call_cnt, memory_order_relaxed);
}

opt_level_t cjit_get_current_opt_level(const cjit_engine_t *engine,
                                        func_id_t id)
{
    const func_table_entry_t *e =
        func_table_get((func_table_t *)engine->ftable, id);
    if (!e) return OPT_NONE;
    return (opt_level_t)atomic_load_explicit(&e->cur_level, memory_order_relaxed);
}
