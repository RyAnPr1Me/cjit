/**
 * cjit.c – Non-blocking JIT compiler engine.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                       Architecture Overview                             │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │                                                                         │
 * │   Runtime threads (any number)                                          │
 * │   ────────────────────────────                                          │
 * │   Hot path per call (CJIT_DISPATCH / cjit_get_func_counted):            │
 * │     fn = atomic_load(&table[id].func_ptr);  ← single acquire-load      │
 * │     fn(args…);                              ← indirect call             │
 * │     tls_counts[id]++;                       ← thread-local byte incr   │
 * │     if (tls_counts[id] >= THRESHOLD)        ← flush every N calls      │
 * │         atomic_fetch_add(&call_cnt, N);     ← amortised atomic flush   │
 * │                                                                         │
 * │   Per-thread TLS batch counting                                         │
 * │   ──────────────────────────────                                        │
 * │   Each calling thread owns a private uint8_t tls_counts[MAX_FUNCTIONS] │
 * │   array (~1 KB TLS per thread).  The hot path only writes to this       │
 * │   thread-private array; no shared-memory traffic until the batch fills. │
 * │   func_ptr (cache line 0) and call_cnt (cache line 1) are on separate   │
 * │   cache lines: func_ptr stays in "Shared" state on all cores for the    │
 * │   lifetime of steady-state execution.                                   │
 * │                                                                         │
 * │   Background threads (started by cjit_start)                           │
 * │   ─────────────────────────────────────────                             │
 * │                                                                         │
 * │   ┌──────────────────────────────────────────────────────────────┐  │
 * │   │  Monitor thread (×1)                                         │  │
 * │   │   • Wakes every monitor_interval_ms (nanosleep, low CPU)     │  │
 * │   │   • Maintains per-function EMA of call rate                  │  │
 * │   │   • 5-gate data-driven promotion (all must pass):            │  │
 * │   │     1. Hard recompile cap  (max_recompiles_per_func)         │  │
 * │   │     2. Scaled rate thresh  (+scale_pct% per recompile)       │  │
 * │   │     3. Stability streak    (hot_confirm_cycles + rc×extra)   │  │
 * │   │     4. O3 uptime gate      (min_uptime_for_tier2_ms)         │  │
 * │   │     5. Scaled O3 min-calls (min_calls_for_tier2 × (1+rc))   │  │
 * │   │   • Adaptive cooloff: max(cfg.cooloff, 2×last_compile_ms)    │  │
 * │   │   • Non-blocking async IR prefetch via ir_cache_prefetch()   │  │
 * │   └─────────────────────────────┬────────────────────────────────┘  │
 * │                                 │  (per-thread lock-free MPMC queues)│
 * │   ┌─────────────────────────────▼────────────────────────────────┐  │
 * │   │  Compiler threads (×[1..CJIT_COMPILER_THREADS], dynamic)     │  │
 * │   │   • Each owns one MPMC queue; tasks routed by func_id%%n      │  │
 * │   │   • Work-steals from neighbours when own queue empty         │  │
 * │   │   • Blocks on shared condvar (instant wake on enqueue)       │  │
 * │   │   • Fetches IR via ir_cache_get_ir() (COLD→WARM on miss)     │  │
 * │   │   • Compiles via posix_spawnp(cc) — no shell overhead        │  │
 * │   │   • Atomic func_table_swap + dgc_retire(old_handle)          │  │
 * │   │   • Writes last_compile_duration_ms for adaptive cooloff      │  │
 * │   └─────────────────────────────┬────────────────────────────────┘  │
 * │                                 │  (lock-free retire stack)          │
 * │   ┌─────────────────────────────▼────────────────────────────────┐  │
 * │   │  GC thread (×1, inside deferred_gc)                          │  │
 * │   │   • Pre-allocated pool of retire_entry_t nodes (no malloc)   │  │
 * │   │   • Sleeps exactly until next handle becomes freeable        │  │
 * │   │   • Exponential back-off when retire stack is empty          │  │
 * │   │   • dlclose handles older than grace_period_ms               │  │
 * │   └──────────────────────────────────────────────────────────────┘  │
 * │                                                                       │
 * │   IR cache (per engine, background threads)                           │
 * │   ──────────────────────────────────────                              │
 * │   ┌─────────────────────────────────────────────────────────────┐    │
 * │   │  I/O prefetch pool (×[0..cfg.io_threads])                   │    │
 * │   │   • Drains pf_buf FIFO of prefetch requests                 │    │
 * │   │   • Promotes COLD IR → WARM in background (hides disk I/O)  │    │
 * │   └─────────────────────────────────────────────────────────────┘    │
 * │   ┌─────────────────────────────────────────────────────────────┐    │
 * │   │  Memory-pressure monitor thread (×1)                        │    │
 * │   │   • Polls /proc/meminfo; adjusts HOT/WARM cache capacities  │    │
 * │   │   • Proactively evicts HOT→WARM / WARM→COLD under pressure  │    │
 * │   └─────────────────────────────────────────────────────────────┘    │
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
#include "ir_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#ifdef __linux__
#include <sched.h>    /* sched_getaffinity, CPU_COUNT */
#endif

/* ══════════════════════ per-thread TLS batch counters ════════════════════ */

/*
 * Per-calling-thread call-count batch array.
 *
 * Each calling thread maintains a private uint8_t counter for every
 * registered function.  On every call, only this thread-local byte is
 * incremented (no shared-memory write, no atomic, no cache-line ownership
 * transfer).  When the counter reaches CJIT_TLS_FLUSH_THRESHOLD, the
 * accumulated count is flushed to the global atomic call_cnt with a
 * single relaxed fetch_add and the local counter is reset to zero.
 *
 * uint8_t × CJIT_MAX_FUNCTIONS = 1 024 bytes of TLS per calling thread.
 * TLS variables are zero-initialised by the C runtime.
 */
static __thread uint8_t cjit_tls_counts[CJIT_MAX_FUNCTIONS];

/* ══════════════════════════ engine structure ══════════════════════════════ */

/*
 * Per-compiler-thread argument block.
 *
 * Stable for the lifetime of the engine (allocated alongside compiler_threads
 * in cjit_create).  Each compiler thread receives a pointer to its own entry
 * so it knows its index for work-stealing without querying TLS.
 */
typedef struct {
    cjit_engine_t *engine;
    uint32_t       thread_idx;
} compiler_thread_arg_t;

struct cjit_engine {
    /* Configuration (immutable after cjit_start). */
    cjit_config_t   cfg;

    /* Function table: the heart of the engine. */
    func_table_t   *ftable;

    /* IR LRU cache with memory-pressure awareness. */
    ir_lru_cache_t *ir_cache;

    /*
     * Per-compiler-thread work queues.
     *
     * Heap-allocated array of cfg.compiler_threads MPMC queues.  Tasks are
     * routed by (func_id % num_threads) so the same function always lands on
     * the same queue.  This gives each function compile_lock affinity to one
     * thread, eliminating cross-thread trylock contention in the common case.
     *
     * Each queue has exactly ONE consumer (its owning thread).  A thread whose
     * queue is empty steals from the next queue in round-robin order before
     * sleeping on the shared condition variable.
     */
    mpmc_queue_t   *work_queues;   /* [0 .. cfg.compiler_threads) */

    /*
     * Shared condition variable.
     *
     * Any enqueue into any per-thread queue signals one waiting compiler
     * thread via this condvar.  Protocol:
     *
     *   Producer (engine_enqueue_task):
     *     mpmc_enqueue(work_queues[target]) → lock → signal → unlock
     *
     *   Consumer (compiler_thread_fn):
     *     all-queues-empty? → lock → while (all empty && !stop) wait → unlock
     *     → retry dequeue from own queue (then neighbours)
     *
     * Using a single condvar (not per-thread) keeps the producer fast: one
     * unconditional mutex_lock/signal/unlock per enqueue, regardless of how
     * many threads are sleeping.
     */
    pthread_mutex_t work_cond_mutex;
    pthread_cond_t  work_cond;

    /* Deferred GC for safe dlclose of retired handles. */
    deferred_gc_t   dgc;

    /*
     * Background thread handles and per-thread arguments.
     * Both are heap-allocated arrays of cfg.compiler_threads entries.
     */
    pthread_t              *compiler_threads;
    compiler_thread_arg_t  *thread_args;
    pthread_t               monitor_thread;

    /* Lifecycle control. */
    atomic_bool     running;
    atomic_bool     stop_requested;

    /**
     * Monotonic timestamp (ms) captured by cjit_start().
     *
     * Used by the monitor thread to compute engine uptime, which gates
     * tier-2 (O3) promotions: compilations triggered by startup hotness
     * spikes (before uptime >= min_uptime_for_tier2_ms) are suppressed.
     * Not atomic: written once in cjit_start() (before compiler/monitor
     * threads exist), then read-only by the monitor thread thereafter.
     */
    uint64_t        start_ms;

    /* Global statistics (relaxed atomics; no strict ordering needed). */
    atomic_uint_fast64_t stat_compilations;
    atomic_uint_fast64_t stat_failed;
    atomic_uint_fast64_t stat_swaps;
};

/* ══════════════════════════ internal helpers ══════════════════════════════ */

/* Monotonic timestamp in milliseconds (shared by monitor helpers). */
static uint64_t engine_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/**
 * Route a compile task to the owning thread's queue and signal a waiter.
 *
 * Routing by (func_id % num_threads) gives each function affinity to one
 * compiler thread.  That means the per-entry compile_lock is almost always
 * acquired by the same thread, eliminating cross-thread trylock contention
 * in the common case.
 *
 * The condvar signal is sent inside a brief mutex section so no wake-up can
 * be lost (see engine struct comment for the full protocol).
 *
 * @return true if the task was accepted, false if the target queue is full.
 */
static bool engine_enqueue_task(cjit_engine_t *engine,
                                 const compile_task_t *task)
{
    uint32_t n      = engine->cfg.compiler_threads;
    uint32_t target = task->func_id % n;
    bool ok = mpmc_enqueue(&engine->work_queues[target], task);
    if (ok) {
        pthread_mutex_lock(&engine->work_cond_mutex);
        pthread_cond_signal(&engine->work_cond);
        pthread_mutex_unlock(&engine->work_cond_mutex);
    }
    return ok;
}

/* ══════════════════════════ compiler thread ═══════════════════════════════ */

/**
 * Main loop of a background compiler thread.
 *
 * Dequeue strategy (zero-contention fast path, graceful fallback):
 *   1. Try own queue first (only consumer — never contends with peers).
 *   2. If empty, round-robin work-steal from each other thread's queue.
 *   3. If all queues empty, block on the shared condvar.
 *
 * Stale-task detection:
 *   entry->version > task.version_req  →  a newer compilation already happened;
 *   discard and continue.
 *
 * Concurrent compilation prevention:
 *   compile_lock per entry: trylock(); if taken, re-enqueue to the affinity
 *   queue and move on.  Because tasks are routed by (func_id % nthreads)
 *   this almost always succeeds on the first attempt.
 */
static void *compiler_thread_fn(void *raw_arg)
{
    compiler_thread_arg_t *targ   = (compiler_thread_arg_t *)raw_arg;
    cjit_engine_t         *engine = targ->engine;
    uint32_t               me     = targ->thread_idx;
    uint32_t               n      = engine->cfg.compiler_threads;

    /* Build codegen options once – immutable after cjit_start(). */
    codegen_opts_t copts = {
        .enable_inlining      = engine->cfg.enable_inlining,
        .enable_vectorization = engine->cfg.enable_vectorization,
        .enable_loop_unroll   = engine->cfg.enable_loop_unroll,
        .enable_native_arch   = engine->cfg.enable_native_arch,
        .enable_fast_math     = engine->cfg.enable_fast_math,
        .verbose              = engine->cfg.verbose,
    };

    while (!atomic_load_explicit(&engine->stop_requested, memory_order_acquire)) {

        /* Step 1: own queue – no peer contention. */
        compile_task_t task;
        bool got = mpmc_dequeue(&engine->work_queues[me], &task);

        /* Step 2: work-steal from neighbours round-robin. */
        if (!got) {
            for (uint32_t i = 1; i < n; ++i) {
                if (mpmc_dequeue(&engine->work_queues[(me + i) % n], &task)) {
                    got = true;
                    break;
                }
            }
        }

        /* Step 3: all queues empty – block on condvar. */
        if (!got) {
            pthread_mutex_lock(&engine->work_cond_mutex);
            /*
             * Re-check all queues under the mutex so we cannot miss a signal
             * that arrived after the steal loop above but before the lock.
             */
            bool any = false;
            for (uint32_t i = 0; i < n && !any; ++i)
                any = (mpmc_size(&engine->work_queues[i]) > 0);
            if (!any && !atomic_load_explicit(&engine->stop_requested,
                                              memory_order_relaxed))
                pthread_cond_wait(&engine->work_cond, &engine->work_cond_mutex);
            pthread_mutex_unlock(&engine->work_cond_mutex);
            continue;
        }

        /* Got a task. */
        func_table_entry_t *entry = func_table_get(engine->ftable, task.func_id);
        if (!entry) continue;

        /*
         * Clear in_queue BEFORE compiling so the monitor can re-enqueue if
         * the function stays hot while a slow compilation is in flight.
         */
        atomic_store_explicit(&entry->in_queue, false, memory_order_relaxed);

        /* Discard stale tasks. */
        uint32_t cur_ver = atomic_load_explicit(&entry->version,
                                                 memory_order_relaxed);
        if (cur_ver > task.version_req) {
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] discard stale '%s' (ver %u > req %u)\n",
                        me, entry->name, cur_ver, task.version_req);
            continue;
        }

        /* Skip if already at the requested tier or higher. */
        opt_level_t cur_level =
            (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                               memory_order_relaxed);
        if (cur_level >= task.target_level) {
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] skip '%s': already O%d >= O%d\n",
                        me, entry->name, (int)cur_level, (int)task.target_level);
            continue;
        }

        /* Acquire per-entry compile lock (non-blocking trylock). */
        if (pthread_mutex_trylock(&entry->compile_lock) != 0) {
            /*
             * Another thread is already compiling this function.
             *
             * Problem without a fix: in_queue was cleared above, so the
             * monitor may independently enqueue a fresh copy of this task.
             * If we also re-enqueue our copy, there will be two tasks for
             * the same function → queue bloat → repeated trylock failures
             * until the first compilation finishes and version increments.
             *
             * Fix: try to reclaim in_queue atomically before re-enqueuing.
             *   CAS succeeds → we own in_queue; re-enqueue our copy as the
             *                  canonical pending task; monitor won't add more.
             *   CAS fails    → monitor already set in_queue=true and enqueued
             *                  a fresh copy; our copy is redundant — discard.
             */
            bool exp = false;
            if (atomic_compare_exchange_strong_explicit(
                    &entry->in_queue, &exp, true,
                    memory_order_relaxed, memory_order_relaxed)) {
                /* We reclaimed the slot; re-enqueue to the affinity queue.
                 * Routing uses (func_id % n), identical to engine_enqueue_task,
                 * so tasks for the same function always land on the same queue. */
                if (!mpmc_enqueue(&engine->work_queues[task.func_id % n], &task)) {
                    /* Queue full: clear in_queue so future enqueues can proceed. */
                    atomic_store_explicit(&entry->in_queue, false,
                                          memory_order_relaxed);
                }
            }
            /* else: monitor's fresh copy is already in the queue; drop ours. */
            continue;
        }

        /* Compilation. */
        if (engine->cfg.verbose)
            fprintf(stderr,
                    "[cjit/compiler#%u] compiling '%s' O%d "
                    "(call_cnt=%llu rate=%llucps)\n",
                    me, entry->name, (int)task.target_level,
                    (unsigned long long)
                    atomic_load_explicit(&entry->call_cnt, memory_order_relaxed),
                    (unsigned long long)task.call_rate);

        /* Fetch IR (promotes COLD->WARM; loads from disk if cold). */
        const char *ir_to_use    = NULL;
        char       *ir_cache_copy = NULL;
        if (engine->ir_cache) {
            ir_cache_copy = ir_cache_get_ir(engine->ir_cache, task.func_id);
            ir_to_use     = ir_cache_copy;
        }
        if (!ir_to_use) ir_to_use = entry->ir_source;
        if (!ir_to_use) {
            fprintf(stderr, "[cjit/compiler#%u] no IR for '%s'\n",
                    me, entry->name);
            pthread_mutex_unlock(&entry->compile_lock);
            free(ir_cache_copy);
            continue;
        }

        uint64_t t0_compile = engine_now_ms();
        codegen_result_t cres;
        bool ok = codegen_compile(entry->name, ir_to_use,
                                   task.target_level, &copts, &cres);
        free(ir_cache_copy);

        /* Record how long this compilation took (relaxed store, read by monitor
         * for adaptive cooloff — zero hot-path overhead). */
        uint32_t dur_ms = (uint32_t)(engine_now_ms() - t0_compile);
        atomic_store_explicit(&entry->last_compile_duration_ms, dur_ms,
                              memory_order_relaxed);

        if (!ok) {
            atomic_fetch_add_explicit(&engine->stat_failed, 1,
                                      memory_order_relaxed);
            fprintf(stderr, "[cjit/compiler#%u] FAILED '%s': %s\n",
                    me, entry->name, cres.errmsg);
            pthread_mutex_unlock(&entry->compile_lock);
            continue;
        }

        atomic_fetch_add_explicit(&engine->stat_compilations, 1,
                                  memory_order_relaxed);

        /* Atomic RCU-style pointer swap. */
        void *old_handle = func_table_swap(engine->ftable, task.func_id,
                                            cres.fn, cres.handle,
                                            task.target_level);
        atomic_fetch_add_explicit(&engine->stat_swaps, 1,
                                  memory_order_relaxed);
        pthread_mutex_unlock(&entry->compile_lock);

        /* Retire old handle – GC thread dlclose's after the grace period. */
        dgc_retire(&engine->dgc, old_handle);

        if (engine->cfg.verbose)
            fprintf(stderr,
                    "[cjit/compiler#%u] swapped '%s' -> O%d (ver %lu)\n",
                    me, entry->name, (int)task.target_level,
                    (unsigned long)atomic_load_explicit(&entry->version,
                                                        memory_order_relaxed));
    }

    return NULL;
}

/* ══════════════════════════ monitor thread ════════════════════════════════ */

/**
 * Main loop of the hot-function monitor thread.
 *
 * Algorithm: EMA-smoothed confidence-based tier promotion
 * ────────────────────────────────────────────────────────
 * All state is LOCAL to this function (zero hot-path overhead):
 *
 *   prev_cnt[i]       – call_cnt at the previous scan
 *   ema_rate[i]       – exponential moving average of calls/sec
 *                       α = 2 / (hot_confirm_cycles + 1)
 *   cnt_at_compile[i] – call_cnt when the last compile task was enqueued
 *   last_queued_ms[i] – monotonic timestamp of the last enqueue
 *   prefetch_done[i]  – true once a prefetch request has been submitted
 *
 * Per-scan decision for function i:
 *
 *   delta = call_cnt[i] - prev_cnt[i]
 *   rate  = delta * 1000 / interval_ms       (instantaneous calls/sec)
 *   ema   = α × rate + (1 − α) × ema         (smoothed)
 *
 *   Warm-up prefetch (zero-overhead, one-shot):
 *     if ema >= hot_rate_t1/10 AND ir is COLD AND !prefetch_done[i]:
 *       ir_cache_prefetch(…)     ← non-blocking, returns immediately
 *       prefetch_done[i] = true
 *     By the time hot_confirm_cycles cycles pass and a compile task fires,
 *     the IR is already in memory.
 *
 *   T1 (OPT_O2):
 *     if ema >= hot_rate_t1:
 *       if (now − last_queued_ms[i]) >= effective_cooloff:  enqueue O2
 *     else: ema decays naturally; no explicit reset needed
 *
 *   T2 (OPT_O3 upgrade from O2):
 *     same gate, plus (call_cnt[i] − cnt_at_compile[i]) >= min_calls_for_tier2
 *
 *   Adaptive cooloff:
 *     effective_cooloff = max(cfg.compile_cooloff_ms,
 *                             2 × entry->last_compile_duration_ms)
 *     Re-enqueuing before the previous compile likely completed is prevented.
 */
static void *monitor_thread_fn(void *arg)
{
    cjit_engine_t *engine    = (cjit_engine_t *)arg;
    uint32_t       max_funcs = engine->cfg.max_functions;
    uint32_t       itvl_ms   = engine->cfg.monitor_interval_ms;
    uint32_t       itvl_safe = itvl_ms > 0 ? itvl_ms : 1;

    /* EMA coefficient: α = 2 / (hot_confirm_cycles + 1).
     * hot_confirm_cycles >= 1 enforced by cjit_create. */
    float alpha = 2.0f / (float)(engine->cfg.hot_confirm_cycles + 1);

    /*
     * Per-function monitoring state, private to this thread.
     *
     * Allocated as ONE contiguous block so that the monitor scan loop
     * (which steps through all arrays in lockstep) benefits from spatial
     * locality.  Layout (descending alignment to avoid padding gaps):
     *
     *   [uint64_t × max_funcs] prev_cnt
     *   [uint64_t × max_funcs] cnt_at_compile
     *   [uint64_t × max_funcs] last_queued_ms
     *   [float    × max_funcs] ema_rate
     *   [uint32_t × max_funcs] hot_scan_streak
     *   [bool     × max_funcs] prefetch_done
     *
     * hot_scan_streak[i] counts consecutive monitor scans where
     * ema_rate[i] was >= the scaled rate threshold for function i.
     * Promotion only happens once the streak reaches:
     *   hot_confirm_cycles + recompile_count[i] × extra_streak_per_recompile
     * This ensures each successive recompile requires a proportionally
     * longer observation window, preventing promotions based on
     * insufficient or transient data.
     */
    size_t monitor_block_sz =
        (size_t)max_funcs * (3 * sizeof(uint64_t) + sizeof(float)
                             + sizeof(uint32_t) + sizeof(bool));
    void *monitor_block = calloc(1, monitor_block_sz);
    if (!monitor_block) {
        fprintf(stderr, "[cjit/monitor] FATAL: cannot allocate monitor state (%zu B)\n", monitor_block_sz);
        return NULL;
    }

    uint64_t *prev_cnt        = (uint64_t *)monitor_block;
    uint64_t *cnt_at_compile  = prev_cnt        + max_funcs;
    uint64_t *last_queued_ms  = cnt_at_compile  + max_funcs;
    float    *ema_rate        = (float   *)(void *)(last_queued_ms  + max_funcs);
    uint32_t *hot_scan_streak = (uint32_t *)(void *)(ema_rate       + max_funcs);
    bool     *prefetch_done   = (bool    *)(void *)(hot_scan_streak + max_funcs);

    struct timespec interval;
    interval.tv_sec  = itvl_ms / 1000;
    interval.tv_nsec = (long)(itvl_ms % 1000) * 1000000L;

    /* Low-watermark rate at which we prefetch COLD IR: 10 % of the T1 gate.
     * This is intentionally generous — prefetch is cheap and cancels itself
     * if the function never becomes truly hot. */
    uint64_t prefetch_rate_threshold = engine->cfg.hot_rate_t1 / 10;
    if (prefetch_rate_threshold == 0) prefetch_rate_threshold = 1;

    while (!atomic_load_explicit(&engine->stop_requested, memory_order_acquire)) {
        nanosleep(&interval, NULL);

        uint64_t now = engine_now_ms();
        /* Engine uptime, used to gate O3 promotions during startup. */
        uint64_t uptime_ms = (now >= engine->start_ms)
                                 ? (now - engine->start_ms) : 0;
        uint32_t n   = atomic_load_explicit(&engine->ftable->count,
                                             memory_order_acquire);

        for (uint32_t i = 0; i < n; ++i) {
            func_table_entry_t *entry = &engine->ftable->entries[i];

            /* ── Update EMA ────────────────────────────────────────────── */
            uint64_t cur_cnt = atomic_load_explicit(&entry->call_cnt,
                                                     memory_order_relaxed);
            uint64_t delta   = cur_cnt - prev_cnt[i];
            prev_cnt[i]      = cur_cnt;

            /* Guard against overflow in delta * 1000ULL and catch counter
             * anomalies (e.g. bugs that produce runaway counters).
             * 1e12 calls/sec over any interval is physically impossible;
             * clamping here makes the rate saturate gracefully rather than
             * wrap to a misleadingly small value. */
            if (delta > (uint64_t)1000000000000ULL)
                delta = (uint64_t)1000000000000ULL;
            uint64_t inst_rate = delta * 1000ULL / itvl_safe;  /* calls/sec */
            ema_rate[i] = ema_rate[i] + alpha * ((float)inst_rate - ema_rate[i]);

            opt_level_t cur_level =
                (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                                   memory_order_relaxed);

            /* Already at maximum tier. */
            if (cur_level >= OPT_O3) continue;

            /* ── Warm-up prefetch (one-shot, non-blocking) ─────────────── */
            if (!prefetch_done[i] && engine->ir_cache &&
                (uint64_t)ema_rate[i] >= prefetch_rate_threshold &&
                ir_cache_get_generation(engine->ir_cache, (func_id_t)i)
                    == IRC_GEN_COLD) {
                if (ir_cache_prefetch(engine->ir_cache, (func_id_t)i))
                    prefetch_done[i] = true;
            }

            /* ── Tier promotion gate ────────────────────────────────────── */

            /*
             * Load how many times this function has already been JIT-compiled.
             * Used below to scale thresholds so each successive recompile
             * requires progressively stronger evidence of sustained benefit.
             */
            uint32_t rc = atomic_load_explicit(&entry->recompile_count,
                                               memory_order_relaxed);

            /*
             * Gate 1 – Hard recompile cap.
             *
             * Once a function has been recompiled max_recompiles_per_func
             * times the monitor stops considering it, preventing infinite
             * recompilation loops and cache-thrashing for diminishing gains.
             * Reset the streak so a later inspection (e.g. after a cfg change)
             * sees a clean slate.
             */
            if (rc >= engine->cfg.max_recompiles_per_func) {
                hot_scan_streak[i] = 0;
                continue;
            }

            opt_level_t target;
            uint64_t    rate_thresh;
            if (cur_level < OPT_O2) {
                target      = OPT_O2;
                rate_thresh = engine->cfg.hot_rate_t1;
            } else {
                target      = OPT_O3;
                rate_thresh = engine->cfg.hot_rate_t2;
            }

            /*
             * Gate 2 – Scaled rate threshold.
             *
             * Effective threshold = base × (1 + rc × scale_pct / 100).
             *
             * Each additional recompile raises the required EMA rate by
             * scale_pct percent, ensuring the monitor only triggers again
             * when the function is proportionally hotter than the previous
             * trigger level.  This prevents re-promotions for marginal gains
             * (e.g. 0.00001% faster) that do not justify the compilation cost
             * or the I-cache pressure of swapping in new code.
             *
             * Example with scale_pct=50:
             *   rc=0 → 1.0× base  (first recompile: standard gate)
             *   rc=1 → 1.5× base  (second: must be 50% hotter)
             *   rc=2 → 2.0× base
             *   rc=4 → 3.0× base
             *
             * Cap at 64× base to avoid overflow; the hard cap above terminates
             * the loop long before rc would reach such values.
             */
            uint64_t scaled_rate_thresh = rate_thresh;
            if (engine->cfg.recompile_rate_scale_pct > 0 && rc > 0) {
                uint64_t bump = rate_thresh
                                * (uint64_t)rc
                                * engine->cfg.recompile_rate_scale_pct
                                / 100ULL;
                uint64_t max_bump = rate_thresh * 63ULL;
                if (bump > max_bump) bump = max_bump;
                scaled_rate_thresh = rate_thresh + bump;
            }

            /* EMA must be above the scaled threshold.
             * If not, this scan is "cold" — reset the streak. */
            if ((uint64_t)ema_rate[i] < scaled_rate_thresh) {
                hot_scan_streak[i] = 0;
                continue;
            }

            /* Increment the consecutive-above-threshold scan streak. */
            if (hot_scan_streak[i] < UINT32_MAX) hot_scan_streak[i]++;

            /*
             * Gate 3 – Stability streak (data sufficiency).
             *
             * Required streak = hot_confirm_cycles
             *                 + rc × extra_streak_per_recompile.
             *
             * A function that has already been recompiled rc times must
             * demonstrate sustained hotness over a proportionally longer
             * observation window.  This is the primary "enough data" gate:
             * a transient spike that dies after a few scans resets the streak
             * to 0 and must prove itself again from scratch.
             *
             * Saturating arithmetic throughout to handle extreme rc values.
             */
            uint32_t req_streak = engine->cfg.hot_confirm_cycles;
            {
                uint32_t extra = engine->cfg.extra_streak_per_recompile;
                uint32_t rc_extra = (extra == 0 || rc == 0)
                                        ? 0
                                        : ((rc <= UINT32_MAX / extra)
                                               ? rc * extra : UINT32_MAX);
                req_streak = (req_streak <= UINT32_MAX - rc_extra)
                                 ? req_streak + rc_extra : UINT32_MAX;
            }
            if (hot_scan_streak[i] < req_streak) continue;

            /* Adaptive cooloff: max(cfg.compile_cooloff_ms,
             *                       2 × last_compile_duration_ms). */
            uint32_t last_dur = atomic_load_explicit(
                &entry->last_compile_duration_ms, memory_order_relaxed);
            uint32_t effective_cooloff = engine->cfg.compile_cooloff_ms;
            /* Saturating multiply: last_dur * 2, clamped to UINT32_MAX. */
            uint32_t last_dur_x2 = (last_dur <= UINT32_MAX / 2)
                                       ? last_dur * 2 : UINT32_MAX;
            if (last_dur_x2 > effective_cooloff)
                effective_cooloff = last_dur_x2;

            if ((now - last_queued_ms[i]) < effective_cooloff) continue;

            if (target == OPT_O3) {
                /*
                 * Gate 4 – Uptime gate for O3.
                 *
                 * Suppress O3 promotions until the engine has been running
                 * for at least min_uptime_for_tier2_ms milliseconds.  Startup
                 * call spikes (module init, JIT warmup, memory layout settling)
                 * do not represent steady-state behaviour; compiling to O3
                 * based on them wastes CPU and I-cache for code that will
                 * rarely be called at the same intensity again.
                 */
                if (uptime_ms < engine->cfg.min_uptime_for_tier2_ms) continue;

                /*
                 * Gate 5 – Scaled min-calls for O3.
                 *
                 * Each successive recompile multiplies the minimum required
                 * call count since the last promotion by (1 + rc), ensuring
                 * the engine has collected proportionally more real-world
                 * usage data before attempting another expensive upgrade.
                 *
                 * Saturating multiply to handle extreme rc values.
                 */
                uint64_t calls_since    = cur_cnt - cnt_at_compile[i];
                uint64_t scaled_min_calls = engine->cfg.min_calls_for_tier2;
                uint64_t mult = (uint64_t)(1 + rc);
                if (scaled_min_calls <= UINT64_MAX / mult)
                    scaled_min_calls *= mult;
                else
                    scaled_min_calls = UINT64_MAX;
                if (calls_since < scaled_min_calls) continue;
            }

            /* ── Enqueue ────────────────────────────────────────────────── */
            bool expected = false;
            if (!atomic_compare_exchange_strong_explicit(
                    &entry->in_queue, &expected, true,
                    memory_order_relaxed, memory_order_relaxed))
                continue;

            compile_task_t task = {
                .func_id      = entry->id,
                .target_level = target,
                .priority     = (target == OPT_O3) ? 2 : 1,
                .version_req  = atomic_load_explicit(&entry->version,
                                                      memory_order_relaxed),
                .call_rate    = (uint64_t)ema_rate[i],
            };

            if (!engine_enqueue_task(engine, &task)) {
                atomic_store_explicit(&entry->in_queue, false,
                                      memory_order_relaxed);
            } else {
                cnt_at_compile[i]   = cur_cnt;
                last_queued_ms[i]   = now;
                hot_scan_streak[i]  = 0;  /* fresh evidence required for next tier */
                prefetch_done[i]    = false;
                if (engine->cfg.verbose)
                    fprintf(stderr,
                            "[cjit/monitor] enqueued '%s' for O%d "
                            "(ema=%.0fcps streak=%u/%u rc=%u cooloff=%ums)\n",
                            entry->name, (int)target,
                            (double)ema_rate[i],
                            hot_scan_streak[i], req_streak,
                            rc, effective_cooloff);
            }
        }
    }

    free(monitor_block);
    return NULL;
}

/* ══════════════════════════ public API ════════════════════════════════════ */

cjit_config_t cjit_default_config(void)
{
    cjit_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_functions = CJIT_MAX_FUNCTIONS;

    /*
     * Default compiler thread count: (online CPUs - 1) reserved for callers.
     *
     * We use sched_getaffinity() first: in containers and cgroup-restricted
     * environments it returns the number of CPUs actually *available* to this
     * process, which may be less than the total online count reported by
     * sysconf(_SC_NPROCESSORS_ONLN).  Falls back to sysconf on error.
     */
    long ncpu = -1;
#ifdef __linux__
    {
        cpu_set_t cs;
        if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
            ncpu = (long)CPU_COUNT(&cs);
    }
#endif
    if (ncpu <= 0)
        ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    uint32_t nthreads = (ncpu > 1) ? (uint32_t)(ncpu - 1) : 1;
    if (nthreads > CJIT_COMPILER_THREADS) nthreads = CJIT_COMPILER_THREADS;
    cfg.compiler_threads = nthreads;

    cfg.hot_threshold_t1    = CJIT_HOT_THRESHOLD_T1;
    cfg.hot_threshold_t2    = CJIT_HOT_THRESHOLD_T2;
    cfg.grace_period_ms     = CJIT_GRACE_PERIOD_MS;
    cfg.monitor_interval_ms = 50;

    /* Hot-function detection (EMA-based confidence). */
    cfg.hot_rate_t1         = CJIT_DEFAULT_HOT_RATE_T1;
    cfg.hot_rate_t2         = CJIT_DEFAULT_HOT_RATE_T2;
    cfg.hot_confirm_cycles  = CJIT_DEFAULT_HOT_CONFIRM_CYCLES;
    cfg.min_calls_for_tier2 = CJIT_DEFAULT_MIN_CALLS_T2;
    cfg.compile_cooloff_ms  = CJIT_DEFAULT_COMPILE_COOLOFF_MS;
    cfg.io_threads          = CJIT_DEFAULT_IO_THREADS;

    cfg.max_recompiles_per_func     = CJIT_DEFAULT_MAX_RECOMPILES;
    cfg.recompile_rate_scale_pct    = CJIT_DEFAULT_RECOMPILE_RATE_SCALE_PCT;
    cfg.min_uptime_for_tier2_ms     = CJIT_DEFAULT_MIN_UPTIME_T2_MS;
    cfg.extra_streak_per_recompile  = CJIT_DEFAULT_EXTRA_STREAK_PER_RECOMPILE;

    cfg.enable_inlining      = true;
    cfg.enable_vectorization = true;
    cfg.enable_loop_unroll   = true;
    cfg.enable_const_fold    = true;
    cfg.enable_native_arch   = true;
    cfg.enable_fast_math     = false;
    cfg.verbose              = false;

    cfg.hot_ir_cache_size         = 64;
    cfg.warm_ir_cache_size        = 128;
    cfg.mem_pressure_check_ms     = 500;
    cfg.mem_pressure_low_pct      = 20;
    cfg.mem_pressure_high_pct     = 10;
    cfg.mem_pressure_critical_pct = 5;
    return cfg;
}

cjit_engine_t *cjit_create(const cjit_config_t *config)
{
    cjit_engine_t *e = calloc(1, sizeof(*e));
    if (!e) return NULL;

    e->cfg = config ? *config : cjit_default_config();

    /* Clamp / floor compiler_threads. */
    if (e->cfg.compiler_threads > CJIT_COMPILER_THREADS)
        e->cfg.compiler_threads = CJIT_COMPILER_THREADS;
    if (e->cfg.compiler_threads == 0)
        e->cfg.compiler_threads = 1;

    /* Clamp hot-detection config to sane ranges. */
    if (e->cfg.hot_confirm_cycles == 0) e->cfg.hot_confirm_cycles = 1;
    if (e->cfg.compile_cooloff_ms == 0) e->cfg.compile_cooloff_ms = 1;

    e->ftable = func_table_create(e->cfg.max_functions);
    if (!e->ftable) { free(e); return NULL; }

    ir_cache_config_t icc = {
        .max_funcs             = e->cfg.max_functions,
        .hot_cap               = e->cfg.hot_ir_cache_size,
        .warm_cap              = e->cfg.warm_ir_cache_size,
        .ir_dir                = e->cfg.ir_disk_dir[0] ? e->cfg.ir_disk_dir : NULL,
        .mem_check_interval_ms = e->cfg.mem_pressure_check_ms,
        .mem_low_pct           = e->cfg.mem_pressure_low_pct,
        .mem_high_pct          = e->cfg.mem_pressure_high_pct,
        .mem_critical_pct      = e->cfg.mem_pressure_critical_pct,
        .num_io_threads        = e->cfg.io_threads,
    };
    e->ir_cache = ir_cache_create(&icc);
    if (!e->ir_cache) { func_table_destroy(e->ftable); free(e); return NULL; }

    /* Per-thread work queues. */
    e->work_queues = calloc(e->cfg.compiler_threads, sizeof(mpmc_queue_t));
    if (!e->work_queues) {
        ir_cache_destroy(e->ir_cache);
        func_table_destroy(e->ftable);
        free(e);
        return NULL;
    }
    for (uint32_t i = 0; i < e->cfg.compiler_threads; ++i)
        mpmc_init(&e->work_queues[i]);

    /* Condition variable for instant compiler wake-up. */
    pthread_mutex_init(&e->work_cond_mutex, NULL);
    pthread_cond_init(&e->work_cond, NULL);

    /* Dynamic thread arrays. */
    e->compiler_threads = calloc(e->cfg.compiler_threads, sizeof(pthread_t));
    e->thread_args      = calloc(e->cfg.compiler_threads,
                                  sizeof(compiler_thread_arg_t));
    if (!e->compiler_threads || !e->thread_args) {
        free(e->compiler_threads); free(e->thread_args);
        free(e->work_queues);
        pthread_cond_destroy(&e->work_cond);
        pthread_mutex_destroy(&e->work_cond_mutex);
        ir_cache_destroy(e->ir_cache);
        func_table_destroy(e->ftable);
        free(e);
        return NULL;
    }

    dgc_init(&e->dgc, e->cfg.grace_period_ms);

    atomic_init(&e->running,           false);
    atomic_init(&e->stop_requested,    false);
    atomic_init(&e->stat_compilations,  0);
    atomic_init(&e->stat_failed,        0);
    atomic_init(&e->stat_swaps,         0);

    return e;
}

void cjit_start(cjit_engine_t *engine)
{
    if (atomic_load_explicit(&engine->running, memory_order_acquire)) return;

    engine->start_ms = engine_now_ms();

    atomic_store_explicit(&engine->stop_requested, false, memory_order_release);
    atomic_store_explicit(&engine->running,        true,  memory_order_release);

    dgc_start(&engine->dgc);

    pthread_create(&engine->monitor_thread, NULL, monitor_thread_fn, engine);

    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i) {
        engine->thread_args[i].engine     = engine;
        engine->thread_args[i].thread_idx = i;
        pthread_create(&engine->compiler_threads[i], NULL,
                       compiler_thread_fn, &engine->thread_args[i]);
    }
}

void cjit_stop(cjit_engine_t *engine)
{
    if (!atomic_load_explicit(&engine->running, memory_order_acquire)) return;

    atomic_store_explicit(&engine->stop_requested, true, memory_order_release);

    /* Wake all sleeping compiler threads so they observe stop_requested. */
    pthread_mutex_lock(&engine->work_cond_mutex);
    pthread_cond_broadcast(&engine->work_cond);
    pthread_mutex_unlock(&engine->work_cond_mutex);

    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i)
        pthread_join(engine->compiler_threads[i], NULL);
    pthread_join(engine->monitor_thread, NULL);

    dgc_stop(&engine->dgc);

    atomic_store_explicit(&engine->running, false, memory_order_release);
}

void cjit_destroy(cjit_engine_t *engine)
{
    if (!engine) return;

    cjit_stop(engine);

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
    ir_cache_destroy(engine->ir_cache);

    pthread_cond_destroy(&engine->work_cond);
    pthread_mutex_destroy(&engine->work_cond_mutex);

    free(engine->work_queues);
    free(engine->compiler_threads);
    free(engine->thread_args);
    free(engine);
}

func_id_t cjit_register_function(cjit_engine_t *engine,
                                   const char    *name,
                                   const char    *ir_source,
                                   jit_func_t     aot_fallback)
{
    if (!engine || !name || !ir_source) return CJIT_INVALID_FUNC_ID;

    func_id_t id = func_table_register(engine->ftable, name, ir_source,
                                        aot_fallback);
    if (id == CJIT_INVALID_FUNC_ID) return id;

    if (engine->ir_cache)
        ir_cache_register(engine->ir_cache, id, name, ir_source);

    return id;
}

/*
 * cjit_get_func – THE HOT PATH.
 *
 * Single acquire-load on x86-64 compiles to plain MOV (TSO provides load-
 * acquire for free); on ARMv8 it compiles to LDAR.
 */
jit_func_t cjit_get_func(cjit_engine_t *engine, func_id_t id)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (__builtin_expect(!e, 0)) return NULL;
    return atomic_load_explicit(&e->func_ptr, memory_order_acquire);
}

void cjit_record_call(cjit_engine_t *engine, func_id_t id)
{
    if (__builtin_expect(id >= CJIT_MAX_FUNCTIONS, 0)) return;
    /*
     * Increment the thread-local batch counter (a plain byte write —
     * zero shared-memory traffic).  When it reaches CJIT_TLS_FLUSH_THRESHOLD,
     * flush the accumulated batch to the global atomic call_cnt and reset.
     *
     * func_table_get() on the flush path is bounded by the same id-valid
     * check above, so the array access is always in bounds.
     */
    if (__builtin_expect(++cjit_tls_counts[id] >= CJIT_TLS_FLUSH_THRESHOLD, 0)) {
        cjit_tls_counts[id] = 0;
        func_table_entry_t *e = func_table_get(engine->ftable, id);
        if (__builtin_expect(e != NULL, 1))
            atomic_fetch_add_explicit(&e->call_cnt, CJIT_TLS_FLUSH_THRESHOLD,
                                      memory_order_relaxed);
    }
}

/*
 * cjit_get_func_counted – combined single-lookup overhead-free hot path.
 *
 * Performs ONE func_table_get (one atomic bounds-check load + one
 * stable-pointer dereference).  Call-count accounting uses the TLS batch
 * counter: the only shared-memory operation is the acquire-load of func_ptr
 * (on cache line 0, now independent of call_cnt on cache line 1).
 *
 * Atomic ordering:
 *   call_cnt flush : relaxed (approximation; once per CJIT_TLS_FLUSH_THRESHOLD
 *                   calls per thread)
 *   func_ptr load  : acquire (function body visible before indirect call)
 */
jit_func_t cjit_get_func_counted(cjit_engine_t *engine, func_id_t id)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (__builtin_expect(!e, 0)) return NULL;
    /*
     * id < ftable->count ≤ CJIT_MAX_FUNCTIONS is guaranteed by func_table_get
     * returning non-NULL, so cjit_tls_counts[id] is always in bounds here.
     */
    if (__builtin_expect(++cjit_tls_counts[id] >= CJIT_TLS_FLUSH_THRESHOLD, 0)) {
        cjit_tls_counts[id] = 0;
        atomic_fetch_add_explicit(&e->call_cnt, CJIT_TLS_FLUSH_THRESHOLD,
                                  memory_order_relaxed);
    }
    return atomic_load_explicit(&e->func_ptr, memory_order_acquire);
}

void cjit_flush_local_counts(cjit_engine_t *engine)
{
    /*
     * Flush any partial batch accumulated since the last automatic flush.
     *
     * For each registered function, cjit_tls_counts[i] holds the number of
     * calls accumulated since the last flush (in [0, CJIT_TLS_FLUSH_THRESHOLD)).
     * Full batches of CJIT_TLS_FLUSH_THRESHOLD have already been added to
     * call_cnt by the threshold-compare path.  Only the sub-threshold remainder
     * is pending here.
     *
     * We scan only registered functions (count ≤ CJIT_MAX_FUNCTIONS), so the
     * TLS array access is always in bounds.
     */
    uint32_t n = atomic_load_explicit(&engine->ftable->count,
                                       memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t rem = cjit_tls_counts[i];
        if (rem == 0) continue;            /* nothing pending */
        cjit_tls_counts[i] = 0;
        atomic_fetch_add_explicit(&engine->ftable->entries[i].call_cnt,
                                  rem, memory_order_relaxed);
    }
}

void cjit_request_recompile(cjit_engine_t *engine,
                              func_id_t      id,
                              opt_level_t    level)
{
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return;

    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(
            &e->in_queue, &expected, true,
            memory_order_relaxed, memory_order_relaxed))
        return;   /* already queued */

    compile_task_t task = {
        .func_id      = id,
        .target_level = level,
        .priority     = 3,   /* manual request = highest priority */
        .version_req  = atomic_load_explicit(&e->version, memory_order_relaxed),
        .call_rate    = 0,   /* manual; rate unknown */
    };

    if (!engine_enqueue_task(engine, &task))
        atomic_store_explicit(&e->in_queue, false, memory_order_relaxed);
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

    /* Sum queue depths and find highest recompile count across all functions. */
    uint32_t qd  = 0;
    uint32_t max_rc = 0;
    uint32_t nf  = s.registered_functions;
    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i)
        qd += mpmc_size(&engine->work_queues[i]);
    for (uint32_t i = 0; i < nf; ++i) {
        uint32_t rc = atomic_load_explicit(
            &engine->ftable->entries[i].recompile_count, memory_order_relaxed);
        if (rc > max_rc) max_rc = rc;
    }
    s.queue_depth        = qd;
    s.max_recompile_count = max_rc;

    if (engine->ir_cache) {
        ir_cache_stats_t cs = ir_cache_get_stats(engine->ir_cache);
        s.ir_hot_count          = cs.hot_count;
        s.ir_warm_count         = cs.warm_count;
        s.ir_cold_count         = cs.cold_count;
        s.ir_disk_writes        = cs.disk_writes;
        s.ir_disk_reads         = cs.disk_reads;
        s.ir_evictions          = cs.evictions;
        s.ir_promotions         = cs.promotions;
        s.ir_cache_hits         = cs.cache_hits;
        s.ir_cache_misses       = cs.cache_misses;
        s.ir_pressure_evictions = cs.pressure_evictions;
        s.mem_pressure          = cs.pressure;
        s.mem_available_mb      = cs.mem_available_mb;
        s.mem_total_mb          = cs.mem_total_mb;
    }
    return s;
}

void cjit_print_stats(const cjit_engine_t *engine)
{
    cjit_stats_t s = cjit_get_stats(engine);
    static const char *const pnames[] =
        { "NORMAL", "MEDIUM", "HIGH", "CRITICAL" };
    fprintf(stderr,
            "╔══════════════════════════════════════════╗\n"
            "║           CJIT Engine Statistics          ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Registered functions  : %6u            ║\n"
            "║  Total compilations    : %6llu            ║\n"
            "║  Failed compilations   : %6llu            ║\n"
            "║  Atomic pointer swaps  : %6llu            ║\n"
            "║  Handles retired (GC)  : %6llu            ║\n"
            "║  Handles freed   (GC)  : %6llu            ║\n"
            "║  Work-queue depth now  : %6u            ║\n"
            "║  Max recompile count   : %6u            ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Memory pressure       : %-8s          ║\n"
            "║  Mem available         : %6llu MB         ║\n"
            "║  Mem total             : %6llu MB         ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  IR HOT  (in memory)   : %6u            ║\n"
            "║  IR WARM (in memory)   : %6u            ║\n"
            "║  IR COLD (on disk)     : %6u            ║\n"
            "║  IR cache hits         : %6llu            ║\n"
            "║  IR cache misses       : %6llu            ║\n"
            "║  IR disk writes        : %6llu            ║\n"
            "║  IR disk reads         : %6llu            ║\n"
            "║  IR LRU evictions      : %6llu            ║\n"
            "║  IR pressure evictions : %6llu            ║\n"
            "║  IR promotions         : %6llu            ║\n"
            "╚══════════════════════════════════════════╝\n",
            s.registered_functions,
            (unsigned long long)s.total_compilations,
            (unsigned long long)s.failed_compilations,
            (unsigned long long)s.total_swaps,
            (unsigned long long)s.retired_handles,
            (unsigned long long)s.freed_handles,
            s.queue_depth,
            s.max_recompile_count,
            pnames[s.mem_pressure],
            (unsigned long long)s.mem_available_mb,
            (unsigned long long)s.mem_total_mb,
            s.ir_hot_count, s.ir_warm_count, s.ir_cold_count,
            (unsigned long long)s.ir_cache_hits,
            (unsigned long long)s.ir_cache_misses,
            (unsigned long long)s.ir_disk_writes,
            (unsigned long long)s.ir_disk_reads,
            (unsigned long long)s.ir_evictions,
            (unsigned long long)s.ir_pressure_evictions,
            (unsigned long long)s.ir_promotions);
}

uint64_t cjit_get_call_count(const cjit_engine_t *engine, func_id_t id)
{
    const func_table_entry_t *e =
        func_table_get(engine->ftable, id);
    if (!e) return 0;
    return atomic_load_explicit(&e->call_cnt, memory_order_relaxed);
}

opt_level_t cjit_get_current_opt_level(const cjit_engine_t *engine,
                                        func_id_t id)
{
    const func_table_entry_t *e =
        func_table_get(engine->ftable, id);
    if (!e) return OPT_NONE;
    return (opt_level_t)atomic_load_explicit(&e->cur_level, memory_order_relaxed);
}

uint32_t cjit_get_recompile_count(const cjit_engine_t *engine, func_id_t id)
{
    const func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return 0;
    return atomic_load_explicit(&e->recompile_count, memory_order_relaxed);
}
