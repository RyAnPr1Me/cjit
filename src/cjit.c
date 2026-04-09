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
#include "codegen_cache.h"
#include "ir_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>
#include <sys/stat.h>  /* mkdir */
#include <ftw.h>       /* nftw for PGO dir cleanup */
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
 * Declared extern in cjit.h so that the CJIT_SHOULD_SAMPLE / CJIT_SAMPLE_ARGS
 * macros can read it inline without a function call.  The cjit_tls_elapsed
 * array remains private to this translation unit.
 *
 * uint8_t × CJIT_MAX_FUNCTIONS = 1 024 bytes of TLS per calling thread.
 * TLS variables are zero-initialised by the C runtime.
 */
__thread uint8_t  cjit_tls_counts[CJIT_MAX_FUNCTIONS];

/*
 * Per-calling-thread elapsed-nanoseconds accumulator.
 *
 * When CJIT_DISPATCH_TIMED or cjit_record_timed_call() is used, the
 * measured nanoseconds for each call are added here.  The buffer is
 * flushed to the global atomic total_elapsed_ns alongside cjit_tls_counts
 * — both share the same CJIT_TLS_FLUSH_THRESHOLD trigger — so only one
 * additional atomic_fetch_add per THRESHOLD calls is required beyond the
 * normal call_cnt flush.
 *
 * uint64_t × CJIT_MAX_FUNCTIONS = 8 192 bytes of TLS per calling thread.
 * Zero-initialised; stays zero for functions not dispatched via timed paths.
 */
static __thread uint64_t cjit_tls_elapsed[CJIT_MAX_FUNCTIONS];

/* ══════════════════════════ PGO helpers ═══════════════════════════════════ */

/*
 * PGO state constants (stored in func_table_entry_t::pgo_state).
 * Kept as plain ints so they can be used with atomic_int without casts.
 */
#define PGO_STATE_NONE    0   /* PGO not started              */
#define PGO_STATE_RUNNING 1   /* Instrumented version running */
#define PGO_STATE_DONE    2   /* PGO cycle complete           */

/*
 * Source snippet prepended to every PGO_GENERATE compilation.
 *
 * Injects _cjit_pgo_flush() into the instrumented .so so the engine can
 * flush the gcov counters to disk (via __gcov_dump / __gcov_flush) at a
 * convenient point WITHOUT waiting for process exit.
 *
 * Both symbols are declared weak so that:
 *   • On GCC with -lgcov: __gcov_dump is linked into the .so directly and
 *     _cjit_pgo_flush() calls it.
 *   • On Clang or when gcov is unavailable: both resolve to NULL; the
 *     function becomes a safe no-op (profile data will not be written, and
 *     the PGO_USE compilation falls back gracefully via -fprofile-correction).
 *
 * __attribute__((visibility("default"))) ensures that dlsym(handle, …) can
 * always locate _cjit_pgo_flush even if the .so was compiled with -fvisibility=
 * hidden (it isn't by default, but be explicit).
 */
static const char CJIT_PGO_FLUSH_HELPER[] =
    "extern void __gcov_dump(void)  __attribute__((weak));\n"
    "extern void __gcov_flush(void) __attribute__((weak));\n"
    "__attribute__((visibility(\"default\")))\n"
    "void _cjit_pgo_flush(void) {\n"
    "    if (__gcov_dump)  { __gcov_dump();  return; }\n"
    "    if (__gcov_flush)   __gcov_flush();\n"
    "}\n";

/*
 * nftw callback for pgo_rmdir: remove files and directories depth-first.
 */
static int pgo_rm_cb(const char *path, const struct stat *sb,
                     int typeflag, struct FTW *ftwbuf)
{
    (void)sb; (void)ftwbuf;
    return (typeflag == FTW_F || typeflag == FTW_SL) ? unlink(path) : rmdir(path);
}

/*
 * Recursively remove a PGO profile-data directory.
 * Silently ignores errors (the directory may already be gone).
 */
static void pgo_rmdir(const char *dir)
{
    if (dir && dir[0])
        nftw(dir, pgo_rm_cb, 8, FTW_DEPTH | FTW_PHYS);
}

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

    /**
     * Persistent compiled-artifact cache.
     *
     * NULL when cfg.cache_dir is empty (cache disabled).
     * When non-NULL, compiler threads check this cache before spawning the
     * system C compiler, and store newly compiled artifacts for future hits.
     */
    codegen_cache_t *artifact_cache;

    /*
     * Per-compiler-thread work queues (2 per thread).
     *
     * Layout: work_queues[2 × thread_idx + lane]
     *   lane 0 = normal queue   (priority 1 — background monitoring promotions)
     *   lane 1 = priority queue (priority ≥ 2 — T2 upgrades, manual requests)
     *
     * A compiler thread always drains its own priority queue first, then its
     * normal queue.  Work-stealing first targets priority queues of idle peers
     * (to get urgent tasks done as fast as possible), then normal queues.
     *
     * Using two separate queues instead of a sorted queue preserves the O(1)
     * lock-free MPMC properties of the underlying Vyukov ring-buffer: there is
     * no reordering or search, just two FIFO queues with a fixed priority rule.
     *
     * Total memory: 2 × cfg.compiler_threads × sizeof(mpmc_queue_t).
     */
    mpmc_queue_t   *work_queues;   /* [2 × cfg.compiler_threads]: [0]=normal [1]=prio */

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
    atomic_uint_fast64_t stat_timeouts;          /**< Compiler subprocess timeouts.  */
    atomic_uint_fast64_t stat_swaps;
    atomic_uint_fast64_t stat_tier_skips;        /**< O0→O3 tier-skip promotions. */
    atomic_uint_fast64_t stat_predictive_promos; /**< Slope-lookahead early promotions. */
    atomic_uint_fast64_t stat_pgo_cycles;

    /**
     * Count of compiler threads that have acquired compile_lock and are
     * actively compiling (between compile_lock acquire and release).
     *
     * Incremented by every compiler thread (or cjit_compile_sync) just
     * before acquiring compile_lock; decremented immediately after
     * releasing it.  Used by cjit_drain_queue() to detect the window
     * between in_queue=false (cleared before the lock) and the actual
     * end of compilation.  Written and read with relaxed ordering since
     * cjit_drain_queue() polls with nanosleep pauses between reads.
     */
    atomic_uint_fast32_t active_compilations;

    /*
     * Condition variable signaled after every compilation attempt (success or
     * failure).  Used by cjit_wait_compiled() to block efficiently without
     * busy-waiting until a function's func_ptr becomes non-NULL.
     *
     * Protocol:
     *   Compiler thread: after func_table_swap (or on failed compile):
     *     lock → broadcast → unlock
     *   cjit_wait_compiled:
     *     lock → while (func_ptr == NULL && !timedout) timedwait → unlock
     */
    pthread_mutex_t compile_done_mutex;
    pthread_cond_t  compile_done_cond;

    /*
     * Compile-event callback (optional).
     *
     * Called by compiler threads after every compilation attempt (success,
     * failure, or timeout) with a cjit_compile_event_t snapshot.
     *
     * Both fields are protected by a single mutex so that the callback + its
     * userdata are always updated and read atomically.  The mutex is acquired
     * once per compilation event — an infrequent operation, never on the hot
     * dispatch path.
     */
    pthread_mutex_t         cb_mutex;
    cjit_compile_callback_t compile_cb;
    void                   *compile_cb_userdata;
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
 * Two-level priority:
 *   priority >= 2 → priority queue (lane 1): T2 upgrades, manual requests.
 *   priority  < 2 → normal queue   (lane 0): background monitoring promotions.
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
    uint32_t lane   = (task->priority >= 2) ? 1u : 0u;
    bool ok = mpmc_enqueue(&engine->work_queues[2 * target + lane], task);
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
        .enable_const_fold    = engine->cfg.enable_const_fold,
        .verbose              = engine->cfg.verbose,
        .extra_cflags         = engine->cfg.extra_cflags[0] ? engine->cfg.extra_cflags : NULL,
        .cc_binary            = engine->cfg.cc_binary[0]    ? engine->cfg.cc_binary    : NULL,
        .cache                = engine->artifact_cache,   /* may be NULL */
        .compile_timeout_ms   = engine->cfg.compile_timeout_ms,
    };

    while (!atomic_load_explicit(&engine->stop_requested, memory_order_acquire)) {

        /*
         * Two-level dequeue: always drain the priority queue first.
         *
         * Step 1a: own priority queue (lane 1) — T2 upgrades, manual requests.
         * Step 1b: own normal queue   (lane 0) — background monitoring.
         * Step 2a: steal priority queues from neighbours (urgent tasks get
         *          processed as fast as possible, across threads if needed).
         * Step 2b: steal normal queues from neighbours (load balancing).
         * Step 3:  all queues empty — block on condvar.
         */
        compile_task_t task;
        bool got;

        /* Step 1a: own priority queue. */
        got = mpmc_dequeue(&engine->work_queues[2 * me + 1], &task);

        /* Step 1b: own normal queue. */
        if (!got)
            got = mpmc_dequeue(&engine->work_queues[2 * me + 0], &task);

        /* Step 2a: work-steal priority queues round-robin. */
        if (!got) {
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t peer = (me + i) % n;
                if (mpmc_dequeue(&engine->work_queues[2 * peer + 1], &task)) {
                    got = true;
                    break;
                }
            }
        }

        /* Step 2b: work-steal normal queues round-robin. */
        if (!got) {
            for (uint32_t i = 1; i < n; ++i) {
                uint32_t peer = (me + i) % n;
                if (mpmc_dequeue(&engine->work_queues[2 * peer + 0], &task)) {
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
            for (uint32_t i = 0; i < 2 * n && !any; ++i)
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

        /* Skip if already at the requested tier or higher.
         * Exception: PGO tasks (pgo_mode != PGO_MODE_NONE) always proceed —
         * PGO_GENERATE re-instruments a function at the same O2 level, and
         * PGO_USE may legitimately target a level already reached via a
         * non-PGO path in a race. */
        opt_level_t cur_level =
            (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                               memory_order_relaxed);
        if (task.pgo_mode == PGO_MODE_NONE && cur_level >= task.target_level) {
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] skip '%s': already O%d >= O%d\n",
                        me, entry->name, (int)cur_level, (int)task.target_level);
            continue;
        }

        /* Acquire per-entry compile lock (non-blocking trylock).
         * Increment active_compilations first so cjit_drain_queue() can
         * observe in-flight work even after in_queue is cleared. */
        atomic_fetch_add_explicit(&engine->active_compilations, 1,
                                  memory_order_relaxed);
        if (pthread_mutex_trylock(&entry->compile_lock) != 0) {
            atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                                      memory_order_relaxed);
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
                 * Use the same priority-lane routing as engine_enqueue_task. */
                uint32_t lane = (task.priority >= 2) ? 1u : 0u;
                if (!mpmc_enqueue(&engine->work_queues[2 * (task.func_id % n) + lane],
                                  &task)) {
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
            atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                                      memory_order_relaxed);
            free(ir_cache_copy);
            continue;
        }

        uint64_t t0_compile = engine_now_ms();
        codegen_result_t cres;
        /*
         * Pass the current argument profile snapshot to codegen.  The profile
         * is stored on the cold cache line of func_table_entry_t; this read
         * is lock-free and the compiler thread may see a slightly stale
         * snapshot — this is intentional (statistical data, not requiring
         * coherence).  The arg_profile pointer is only valid until the end of
         * this codegen_compile() call (entry remains live throughout since we
         * hold compile_lock).
         *
         * Also pass the runtime call-rate and average elapsed-time-per-call
         * observed at the time this task was enqueued.  These are injected by
         * codegen_compile() as preprocessor defines (CJIT_CALL_RATE,
         * CJIT_AVG_ELAPSED_NS) so user IR can make compile-time decisions
         * based on the function's actual runtime profile.
         */
        copts.arg_profile = &entry->arg_profile;
        copts.call_rate   = task.call_rate;
        {
            uint64_t cnt_snap = atomic_load_explicit(&entry->call_cnt,
                                                      memory_order_relaxed);
            uint64_t ns_snap  = atomic_load_explicit(&entry->total_elapsed_ns,
                                                      memory_order_relaxed);
            copts.avg_elapsed_ns = (cnt_snap > 0) ? (ns_snap / cnt_snap) : 0;
        }

        /*
         * PGO: per-task compilation setup.
         *
         * For PGO tasks we need to:
         *   PGO_GENERATE – (a) create the profile directory, (b) augment
         *                  extra_cflags with -fprofile-generate=<dir> -lgcov,
         *                  (c) prepend the _cjit_pgo_flush() helper to the IR.
         *   PGO_USE      – augment extra_cflags with -fprofile-use=<dir>
         *                  -fprofile-correction and skip the artifact cache
         *                  (profile data is run-specific; caching would defeat
         *                  the purpose and produce a wrong cache key).
         *
         * We modify a *per-task copy* of copts so the shared copts template is
         * not mutated between tasks.
         */
        char           pgo_extra_cflags[1100];
        char          *pgo_ir        = NULL;
        codegen_opts_t task_copts    = copts;   /* per-task copy */
        const char    *effective_ir  = ir_to_use;

        if (task.pgo_mode == PGO_MODE_GENERATE) {
            /* Construct (or reuse) the per-function PGO directory. */
            const char *base = (engine->cfg.pgo_base_dir[0])
                                   ? engine->cfg.pgo_base_dir : "/tmp";
            snprintf(entry->pgo_dir, sizeof(entry->pgo_dir),
                     "%s/cjit_pgo_%d_%u", base, (int)getpid(),
                     (unsigned)task.func_id);
            mkdir(entry->pgo_dir, 0700); /* ignore EEXIST */

            /* Augment extra_cflags for instrumented compilation.
             * -fprofile-update=prefer-atomic: use non-atomic counter updates
             * where the compiler can prove no concurrent access (much cheaper
             * than fully-atomic on every edge); fall back to atomic only where
             * needed.  This reduces instrumentation overhead by ~30-50% on
             * typical single-threaded JIT functions compared to =atomic while
             * still producing correct profiles on multi-threaded code. */
            snprintf(pgo_extra_cflags, sizeof(pgo_extra_cflags),
                     "-fprofile-generate=%s -fprofile-update=prefer-atomic -lgcov%s%s",
                     entry->pgo_dir,
                     (copts.extra_cflags && copts.extra_cflags[0]) ? " " : "",
                     copts.extra_cflags ? copts.extra_cflags : "");
            task_copts.extra_cflags = pgo_extra_cflags;

            /* PGO compilations must never hit the artifact cache
             * (profile data is run-specific). */
            task_copts.cache = NULL;

            /* Prepend the gcov flush helper to the IR. */
            size_t hlen = sizeof(CJIT_PGO_FLUSH_HELPER) - 1;
            size_t ilen = strlen(ir_to_use);
            pgo_ir = malloc(hlen + ilen + 1);
            if (pgo_ir) {
                memcpy(pgo_ir, CJIT_PGO_FLUSH_HELPER, hlen);
                memcpy(pgo_ir + hlen, ir_to_use, ilen + 1);
                effective_ir = pgo_ir;
            }
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] PGO_GENERATE '%s' dir=%s\n",
                        me, entry->name, entry->pgo_dir);
        } else if (task.pgo_mode == PGO_MODE_USE) {
            /* Augment extra_cflags for PGO-optimised compilation. */
            snprintf(pgo_extra_cflags, sizeof(pgo_extra_cflags),
                     "-fprofile-use=%s -fprofile-correction%s%s",
                     entry->pgo_dir,
                     (copts.extra_cflags && copts.extra_cflags[0]) ? " " : "",
                     copts.extra_cflags ? copts.extra_cflags : "");
            task_copts.extra_cflags = pgo_extra_cflags;
            task_copts.cache        = NULL; /* same reason as above */
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] PGO_USE '%s' dir=%s\n",
                        me, entry->name, entry->pgo_dir);
        }

        bool ok = codegen_compile(entry->name, effective_ir,
                                   task.target_level, &task_copts, &cres);
        free(pgo_ir);

        /* Restore shared copts fields that the per-task copy may have changed. */
        copts.arg_profile    = NULL; /* clear for next task */
        copts.call_rate      = 0;
        copts.avg_elapsed_ns = 0;
        free(ir_cache_copy);

        /* Record how long this compilation took (relaxed store, read by monitor
         * for adaptive cooloff — zero hot-path overhead). */
        uint32_t dur_ms = (uint32_t)(engine_now_ms() - t0_compile);
        atomic_store_explicit(&entry->last_compile_duration_ms, dur_ms,
                              memory_order_relaxed);

        if (!ok) {
            /*
             * PGO failure handling.
             *
             * If a PGO_GENERATE task fails (e.g. -lgcov not available on a
             * Clang system), mark the function as PGO_DONE so the monitor
             * falls back to a normal O3 compile rather than retrying PGO
             * indefinitely.
             *
             * If a PGO_USE task fails, the function is still running the
             * instrumented O2 version (which is slower).  Set pgo_state=DONE
             * and clean up the profile dir; the monitor will promote to plain
             * O3 on the next hot cycle.
             */
            if (task.pgo_mode == PGO_MODE_GENERATE || task.pgo_mode == PGO_MODE_USE) {
                pgo_rmdir(entry->pgo_dir);
                entry->pgo_dir[0] = '\0';
                entry->pgo_instr_handle = NULL;
                atomic_store_explicit(&entry->pgo_state, PGO_STATE_DONE,
                                      memory_order_relaxed);
            }
            atomic_fetch_add_explicit(&engine->stat_failed, 1,
                                      memory_order_relaxed);
            if (cres.timed_out) {
                atomic_fetch_add_explicit(&engine->stat_timeouts, 1,
                                          memory_order_relaxed);
                if (engine->cfg.verbose)
                    fprintf(stderr, "[cjit/compiler#%u] TIMEOUT '%s' (>%u ms)\n",
                            me, entry->name, engine->cfg.compile_timeout_ms);
            } else {
                fprintf(stderr, "[cjit/compiler#%u] FAILED '%s': %s\n",
                        me, entry->name, cres.errmsg);
            }
            pthread_mutex_unlock(&entry->compile_lock);
            atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                                      memory_order_relaxed);
            /* Wake any cjit_wait_compiled() caller so it can detect failure. */
            pthread_mutex_lock(&engine->compile_done_mutex);
            pthread_cond_broadcast(&engine->compile_done_cond);
            pthread_mutex_unlock(&engine->compile_done_mutex);
            /* Fire compile-event callback (failure path). */
            pthread_mutex_lock(&engine->cb_mutex);
            cjit_compile_callback_t cb = engine->compile_cb;
            void *cb_ud = engine->compile_cb_userdata;
            pthread_mutex_unlock(&engine->cb_mutex);
            if (cb) {
                cjit_compile_event_t ev;
                memset(&ev, 0, sizeof(ev));
                ev.func_id    = task.func_id;
                snprintf(ev.func_name, sizeof(ev.func_name), "%s", entry->name);
                ev.level      = task.target_level;
                ev.success    = false;
                ev.timed_out  = cres.timed_out;
                ev.cache_hit  = false;
                ev.duration_ms = dur_ms;
                snprintf(ev.errmsg, sizeof(ev.errmsg), "%.255s", cres.errmsg);
                cb(&ev, cb_ud);
            }
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

        /*
         * PGO success book-keeping (done BEFORE releasing compile_lock so
         * the monitor observes pgo_state after all writes are visible).
         *
         * PGO_GENERATE:
         *   Store the instrumented .so's handle and the current call count.
         *   Then atomically publish pgo_state=RUNNING (release) so the
         *   monitor can start checking for profile-data sufficiency.
         *
         * PGO_USE:
         *   Profile has been consumed; mark the cycle complete, bump the stat,
         *   clear the handle reference (the old_handle below goes to GC), and
         *   clean up the .gcda files from disk.
         */
        if (task.pgo_mode == PGO_MODE_GENERATE) {
            entry->pgo_instr_handle  = cres.handle; /* not yet in GC */
            entry->pgo_calls_at_start =
                atomic_load_explicit(&entry->call_cnt, memory_order_relaxed);
            /* Snapshot current avg ns/call so the monitor can measure the
             * instrumentation overhead (profiling overhead = new_avg - old_avg). */
            {
                uint64_t snap_cnt = atomic_load_explicit(&entry->call_cnt,
                                                          memory_order_relaxed);
                uint64_t snap_ns  = atomic_load_explicit(&entry->total_elapsed_ns,
                                                          memory_order_relaxed);
                entry->pgo_pre_instr_avg_ns = (snap_cnt > 0)
                                              ? (snap_ns / snap_cnt) : 0;
            }
            atomic_store_explicit(&entry->pgo_state, PGO_STATE_RUNNING,
                                  memory_order_release);
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] PGO instrumented '%s' "
                        "(collect %u calls, pre_avg=%" PRIu64 "ns)\n",
                        me, entry->name, engine->cfg.pgo_profile_calls,
                        entry->pgo_pre_instr_avg_ns);
        } else if (task.pgo_mode == PGO_MODE_USE) {
            entry->pgo_instr_handle = NULL; /* old_handle goes to DGC below */
            atomic_store_explicit(&entry->pgo_state, PGO_STATE_DONE,
                                  memory_order_relaxed);
            atomic_fetch_add_explicit(&engine->stat_pgo_cycles, 1,
                                      memory_order_relaxed);
            /* Clean up profile data directory. */
            pgo_rmdir(entry->pgo_dir);
            entry->pgo_dir[0] = '\0';
            if (engine->cfg.verbose)
                fprintf(stderr,
                        "[cjit/compiler#%u] PGO optimised '%s' O3\n",
                        me, entry->name);
        }

        pthread_mutex_unlock(&entry->compile_lock);
        atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                                  memory_order_relaxed);

        /* Retire old handle – GC thread dlclose's after the grace period. */
        dgc_retire(&engine->dgc, old_handle);

        /* Wake any cjit_wait_compiled() caller now that func_ptr is live. */
        pthread_mutex_lock(&engine->compile_done_mutex);
        pthread_cond_broadcast(&engine->compile_done_cond);
        pthread_mutex_unlock(&engine->compile_done_mutex);

        /* Fire compile-event callback (success path). */
        pthread_mutex_lock(&engine->cb_mutex);
        cjit_compile_callback_t cb2 = engine->compile_cb;
        void *cb2_ud = engine->compile_cb_userdata;
        pthread_mutex_unlock(&engine->cb_mutex);
        if (cb2) {
            cjit_compile_event_t ev;
            memset(&ev, 0, sizeof(ev));
            ev.func_id    = task.func_id;
            snprintf(ev.func_name, sizeof(ev.func_name), "%s", entry->name);
            ev.level      = task.target_level;
            ev.success    = true;
            ev.timed_out  = false;
            ev.cache_hit  = cres.cache_hit;
            ev.duration_ms = dur_ms;
            cb2(&ev, cb2_ud);
        }

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
     *   [uint64_t × max_funcs] prev_elapsed       ← nanosecond baseline
     *   [float    × max_funcs] ema_rate
     *   [float    × max_funcs] prev_ema_rate       ← for slope extrapolation
     *   [float    × max_funcs] ema_ns_per_sec      ← CPU-time EMA (ns/s)
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
     *
     * prev_ema_rate[i] holds the EMA value from the previous scan cycle.
     * The per-cycle delta (ema_rate[i] - prev_ema_rate[i]) is the "slope"
     * used for predictive promotion lookahead.
     */
    size_t monitor_block_sz =
        (size_t)max_funcs * (4 * sizeof(uint64_t) + 3 * sizeof(float)
                             + sizeof(uint32_t) + 2 * sizeof(bool));
    void *monitor_block = calloc(1, monitor_block_sz);
    if (!monitor_block) {
        fprintf(stderr, "[cjit/monitor] FATAL: cannot allocate monitor state (%zu B)\n", monitor_block_sz);
        return NULL;
    }

    uint64_t *prev_cnt          = (uint64_t *)monitor_block;
    uint64_t *cnt_at_compile    = prev_cnt          + max_funcs;
    uint64_t *last_queued_ms    = cnt_at_compile    + max_funcs;
    uint64_t *prev_elapsed      = last_queued_ms    + max_funcs;
    float    *ema_rate          = (float   *)(void *)(prev_elapsed      + max_funcs);
    float    *prev_ema_rate     = (float   *)(void *)(ema_rate          + max_funcs);
    float    *ema_ns_per_sec    = (float   *)(void *)(prev_ema_rate     + max_funcs);
    uint32_t *hot_scan_streak   = (uint32_t *)(void *)(ema_ns_per_sec   + max_funcs);
    bool     *prefetch_done     = (bool    *)(void *)(hot_scan_streak   + max_funcs);
    bool     *tier_skip_pending = (bool    *)(void *)(prefetch_done     + max_funcs);

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

            /*
             * Update EMA and track the per-scan slope for predictive promotion.
             *
             * slope = new_ema − old_ema  (calls/sec gained per scan cycle)
             *
             * A positive slope means the call rate is currently rising.  The
             * predictive promotion feature extrapolates:
             *
             *   predicted_rate = ema_rate + slope × lookahead_cycles
             *
             * and uses predicted_rate for threshold comparisons instead of the
             * raw EMA.  This fires promotions earlier when the rate is trending
             * upward, reducing first-compile latency during fast-ramp workloads.
             */
            float old_ema   = ema_rate[i];
            ema_rate[i]     = old_ema + alpha * ((float)inst_rate - old_ema);
            float ema_slope = ema_rate[i] - prev_ema_rate[i];
            prev_ema_rate[i] = ema_rate[i];

            /*
             * Update the CPU-time EMA (nanoseconds/second).
             *
             * Read the cumulative elapsed nanoseconds for this function, take
             * the delta since the last scan, and convert to ns/sec using the
             * same interval divisor as the call-rate EMA.
             *
             * Clamped to 1e15 ns/sec (> 10^6 × CPU speed) to prevent overflow
             * in the float EMA from impossible counter values.
             */
            uint64_t cur_elapsed = atomic_load_explicit(&entry->total_elapsed_ns,
                                                         memory_order_relaxed);
            uint64_t delta_ns    = cur_elapsed - prev_elapsed[i];
            prev_elapsed[i]      = cur_elapsed;
            if (delta_ns > (uint64_t)1000000000000000ULL)
                delta_ns = (uint64_t)1000000000000000ULL;
            uint64_t inst_ns_per_sec = delta_ns * 1000ULL / itvl_safe;
            ema_ns_per_sec[i] = ema_ns_per_sec[i]
                              + alpha * ((float)inst_ns_per_sec - ema_ns_per_sec[i]);

            opt_level_t cur_level =
                (opt_level_t)atomic_load_explicit(&entry->cur_level,
                                                   memory_order_relaxed);

            /* Already at maximum tier. */
            if (cur_level >= OPT_O3) {
                /* Clear pending flag once the skip target has been reached. */
                tier_skip_pending[i] = false;
                continue;
            }

            /* ── Pinned: skip auto-promotion ────────────────────────────── */
            if (atomic_load_explicit(&entry->pinned, memory_order_relaxed))
                continue;

            /* ── Warm-up prefetch (one-shot, non-blocking) ─────────────── */
            if (!prefetch_done[i] && engine->ir_cache &&
                (uint64_t)ema_rate[i] >= prefetch_rate_threshold &&
                ir_cache_get_generation(engine->ir_cache, (func_id_t)i)
                    == IRC_GEN_COLD) {
                if (ir_cache_prefetch(engine->ir_cache, (func_id_t)i))
                    prefetch_done[i] = true;
            }

            /* ── O1 warm-up tier (optional, fires before tier-skip) ─────── */
            /*
             * When enable_o1_warmup is set, compile to OPT_O1 as soon as
             * the EMA crosses warm_rate_t0 (default: hot_rate_t1 / 4).
             * OPT_O1 is fast to compile (typically < 100 ms) and eliminates
             * the window where the function runs as unoptimised baseline code
             * while waiting for the hot_confirm_cycles streak required for
             * OPT_O2.
             *
             * No streak gate: OPT_O1 is cheap enough that one false-positive
             * is harmless.  The normal OPT_O2 and OPT_O3 gates are unaffected.
             * hot_scan_streak is NOT reset here so O2's streak keeps building.
             */
            if (engine->cfg.enable_o1_warmup && cur_level < OPT_O1) {
                uint64_t t0_wu = engine->cfg.warm_rate_t0;
                if (t0_wu == 0) t0_wu = engine->cfg.hot_rate_t1 / 4;
                if (t0_wu == 0) t0_wu = 1;

                if ((uint64_t)ema_rate[i] >= t0_wu) {
                    uint32_t last_dur_wu = atomic_load_explicit(
                        &entry->last_compile_duration_ms, memory_order_relaxed);
                    uint32_t eff_cooloff_wu = engine->cfg.compile_cooloff_ms;
                    uint32_t last_dur_wu_x2 = (last_dur_wu <= UINT32_MAX / 2)
                                                  ? last_dur_wu * 2 : UINT32_MAX;
                    if (last_dur_wu_x2 > eff_cooloff_wu)
                        eff_cooloff_wu = last_dur_wu_x2;

                    if ((now - last_queued_ms[i]) >= eff_cooloff_wu) {
                        bool exp_wu = false;
                        if (atomic_compare_exchange_strong_explicit(
                                &entry->in_queue, &exp_wu, true,
                                memory_order_relaxed, memory_order_relaxed)) {
                            uint32_t ver_wu = atomic_load_explicit(
                                &entry->version, memory_order_relaxed);
                            compile_task_t wu_task = {
                                .func_id      = (func_id_t)i,
                                .target_level = OPT_O1,
                                .priority     = 1,
                                .version_req  = ver_wu,
                                .call_rate    = (uint64_t)ema_rate[i],
                            };
                            if (engine_enqueue_task(engine, &wu_task)) {
                                last_queued_ms[i] = now;
                                cnt_at_compile[i] = cur_cnt;
                                if (engine->cfg.verbose)
                                    fprintf(stderr,
                                        "[cjit/monitor] O1-warmup '%s'"
                                        " (rate=%.0f thresh=%llu)\n",
                                        entry->name, (double)ema_rate[i],
                                        (unsigned long long)t0_wu);
                            } else {
                                atomic_store_explicit(&entry->in_queue, false,
                                                      memory_order_relaxed);
                            }
                        }
                    }
                    /* Fall through: normal T1/T2 gate runs this same cycle.
                     * If rate also crosses hot_rate_t1 the CAS will fail
                     * (in_queue already true) and the O2 task will be picked
                     * up on the next scan cycle once O1 has been dequeued. */
                }
            }

            /* ── Tier-skip optimization ─────────────────────────────────── */
            /*
             * When tier_skip_multiplier > 0 and the function has not yet
             * been compiled, check whether the call rate already exceeds
             * (hot_rate_t2 × multiplier).  If so, issue an OPT_O3 task
             * directly (skip the intermediate OPT_O2 tier), saving one
             * complete compiler invocation.
             *
             * The same hot_confirm_cycles gate applies, so the function must
             * sustain the elevated rate for the full streak before the skip
             * fires — transient spikes do not trigger this path.
             *
             * When this path fires the normal T1/T2 gate below is skipped.
             */
            if (engine->cfg.tier_skip_multiplier > 0.0f &&
                cur_level < OPT_O2 &&
                !tier_skip_pending[i]) {
                float skip_thresh =
                    (float)engine->cfg.hot_rate_t2 * engine->cfg.tier_skip_multiplier;
                if (ema_rate[i] >= skip_thresh) {
                    /* Reload streak and recompile cap. */
                    uint32_t rc_skip = atomic_load_explicit(&entry->recompile_count,
                                                             memory_order_relaxed);
                    if (rc_skip < engine->cfg.max_recompiles_per_func) {
                        uint32_t req_streak = engine->cfg.hot_confirm_cycles
                            + rc_skip * engine->cfg.extra_streak_per_recompile;
                        hot_scan_streak[i]++;
                        if (hot_scan_streak[i] >= req_streak) {
                            uint64_t now2 = engine_now_ms();
                            uint32_t last_dur =
                                atomic_load_explicit(&entry->last_compile_duration_ms,
                                                     memory_order_relaxed);
                            uint64_t eff_cooloff =
                                engine->cfg.compile_cooloff_ms > 2 * last_dur
                                    ? engine->cfg.compile_cooloff_ms
                                    : 2 * (uint64_t)last_dur;
                            if ((now2 - last_queued_ms[i]) >= eff_cooloff) {
                                uint32_t ver =
                                    atomic_load_explicit(&entry->version,
                                                          memory_order_relaxed);
                                bool already =
                                    atomic_load_explicit(&entry->in_queue,
                                                          memory_order_relaxed);
                                if (!already) {
                                    bool exp2 = false;
                                    if (atomic_compare_exchange_strong_explicit(
                                            &entry->in_queue, &exp2, true,
                                            memory_order_relaxed,
                                            memory_order_relaxed)) {
                                        compile_task_t task = {
                                            .func_id      = (func_id_t)i,
                                            .target_level = OPT_O3,
                                            .priority     = 2,
                                            .version_req  = ver,
                                            .call_rate    = (uint64_t)ema_rate[i],
                                        };
                                        if (engine_enqueue_task(engine, &task)) {
                                            last_queued_ms[i]   = now2;
                                            cnt_at_compile[i]   = cur_cnt;
                                            hot_scan_streak[i]  = 0;
                                            tier_skip_pending[i] = true;
                                            atomic_fetch_add_explicit(
                                                &engine->stat_tier_skips, 1,
                                                memory_order_relaxed);
                                            if (engine->cfg.verbose)
                                                fprintf(stderr,
                                                    "[cjit/monitor] tier-skip '%s' → O3"
                                                    " (rate=%.0f skip_thresh=%.0f)\n",
                                                    entry->name,
                                                    (double)ema_rate[i],
                                                    (double)skip_thresh);
                                            continue;
                                        }
                                        atomic_store_explicit(&entry->in_queue, false,
                                                              memory_order_relaxed);
                                    }
                                }
                            }
                        }
                        continue; /* handled by tier-skip path; skip normal gate */
                    }
                }
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
            uint64_t    ns_thresh;
            if (cur_level < OPT_O2) {
                target      = OPT_O2;
                rate_thresh = engine->cfg.hot_rate_t1;
                ns_thresh   = engine->cfg.cpu_hot_ns_per_sec_t1;
            } else {
                target      = OPT_O3;
                rate_thresh = engine->cfg.hot_rate_t2;
                ns_thresh   = engine->cfg.cpu_hot_ns_per_sec_t2;
            }

            /*
             * Gate 2 – Scaled rate threshold (call rate) AND/OR CPU-time
             * threshold.
             *
             * The call-rate EMA path remains the primary signal and uses
             * the same per-recompile scaled threshold as before.  In
             * addition, when cpu_hot_ns_per_sec_t1/t2 is non-zero, the
             * monitor also evaluates a CPU-time EMA gate: a function that
             * is called infrequently but runs for a long time per call can
             * be promoted based on nanoseconds-per-second even if it never
             * crosses the call-rate threshold.
             *
             * The same recompile-count scale factor is applied to the ns
             * threshold, preventing repeated O3 promotions for diminishing
             * CPU-time gains.
             *
             * Both EMAs use the same alpha (smoothing period), so they
             * track on the same time scale.
             *
             * "Hot" condition: rate_hot OR (ns_thresh > 0 AND ns_hot)
             *
             * Cap at 64× base (same as call-rate path) to avoid overflow.
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

            uint64_t scaled_ns_thresh = ns_thresh;
            if (ns_thresh > 0 && engine->cfg.recompile_rate_scale_pct > 0 && rc > 0) {
                uint64_t bump = ns_thresh
                                * (uint64_t)rc
                                * engine->cfg.recompile_rate_scale_pct
                                / 100ULL;
                uint64_t max_bump = ns_thresh * 63ULL;
                if (bump > max_bump) bump = max_bump;
                scaled_ns_thresh = ns_thresh + bump;
            }

            bool rate_hot = ((uint64_t)ema_rate[i] >= scaled_rate_thresh);
            bool ns_hot   = (scaled_ns_thresh > 0 &&
                             (uint64_t)ema_ns_per_sec[i] >= scaled_ns_thresh);

            /*
             * Predictive promotion: if the call-rate EMA has been rising
             * (positive slope) and prediction_lookahead_cycles > 0, extrapolate
             * the rate forward to see if it will cross the threshold within the
             * lookahead window.  A positive prediction triggers a "predictive
             * hot" condition that counts toward the streak even if the raw EMA
             * has not yet reached the threshold.
             *
             * We clamp the slope contribution to the threshold value to prevent
             * a single extreme spike from masking as a sustained trend.
             *
             * The counter stat_predictive_promos is incremented only when the
             * predictive signal is the deciding factor (i.e., rate_hot was false
             * but predicted_hot would be true) AND the streak gate is met.
             */
            bool predicted_hot = false;
            if (!rate_hot && engine->cfg.prediction_lookahead_cycles > 0
                          && ema_slope > 0.0f) {
                float max_slope_contrib = (float)scaled_rate_thresh;
                float slope_contrib = ema_slope
                    * (float)engine->cfg.prediction_lookahead_cycles;
                if (slope_contrib > max_slope_contrib)
                    slope_contrib = max_slope_contrib;
                predicted_hot = ((ema_rate[i] + slope_contrib) >=
                                 (float)scaled_rate_thresh);
            }

            /* Neither signal crosses its threshold → cold scan, reset streak. */
            if (!rate_hot && !ns_hot && !predicted_hot) {
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

                /*
                 * PGO gate: when enable_pgo is set, intercept the first O3
                 * promotion and redirect it through the PGO cycle instead.
                 *
                 *   PGO_STATE_NONE    → issue a PGO_GENERATE task (O2 +
                 *                       profiling) instead of plain O3.
                 *   PGO_STATE_RUNNING → profile collection is in progress;
                 *                       the second monitor pass below handles
                 *                       the flush + PGO_USE scheduling.  Skip
                 *                       this cycle to avoid a race.
                 *   PGO_STATE_DONE    → PGO cycle completed (or failed and
                 *                       fell back); allow the normal O3 path.
                 */
                if (engine->cfg.enable_pgo) {
                    int pgo_st = atomic_load_explicit(&entry->pgo_state,
                                                      memory_order_relaxed);
                    if (pgo_st == PGO_STATE_RUNNING) {
                        /* Second pass will schedule PGO_USE; skip here. */
                        hot_scan_streak[i] = 0;
                        continue;
                    } else if (pgo_st == PGO_STATE_NONE) {
                        /*
                         * PGO cost/benefit gate.
                         *
                         * Estimate the total overhead of a PGO cycle:
                         *
                         *   profiling_window_ms:
                         *     How long the function will run with gcov counters
                         *     enabled = pgo_profile_calls * 1000 / ema_rate.
                         *     If ema_rate is 0 (shouldn't happen here but be
                         *     safe) we conservatively assume a long window.
                         *
                         *   compile_overhead_ms:
                         *     Two extra compiler invocations on top of the O2
                         *     compile already done = 2 × last_compile_duration_ms.
                         *
                         * If the sum exceeds pgo_max_overhead_ms (and
                         * pgo_max_overhead_ms > 0), skip PGO and fall through
                         * to a direct O3 compile.  This ensures PGO is only
                         * applied to functions that are both hot enough to
                         * amortise the overhead AND can complete the profiling
                         * window quickly.
                         */
                        if (engine->cfg.pgo_max_overhead_ms > 0) {
                            uint32_t need = (engine->cfg.pgo_profile_calls > 0)
                                             ? engine->cfg.pgo_profile_calls : 5000u;
                            /* profiling window: calls / (rate/sec) → ms */
                            uint64_t rate = (uint64_t)ema_rate[i];
                            uint64_t prof_ms = (rate > 0)
                                ? ((uint64_t)need * 1000ULL + rate - 1) / rate
                                : (uint64_t)engine->cfg.pgo_max_overhead_ms + 1;
                            uint32_t last_dur_pgo = atomic_load_explicit(
                                &entry->last_compile_duration_ms,
                                memory_order_relaxed);
                            /* Saturating add of 2 × last_compile_duration */
                            uint64_t compile_oh = (last_dur_pgo <= UINT32_MAX / 2)
                                ? (uint64_t)last_dur_pgo * 2 : (uint64_t)UINT32_MAX;
                            uint64_t total_oh = prof_ms + compile_oh;
                            if (total_oh > engine->cfg.pgo_max_overhead_ms) {
                                /* Overhead too high: skip PGO, allow direct O3. */
                                if (engine->cfg.verbose)
                                    fprintf(stderr,
                                        "[cjit/monitor] PGO skip '%s': "
                                        "cost %" PRIu64 "ms > limit %ums "
                                        "(prof=%" PRIu64 "ms cc=%" PRIu64 "ms)\n",
                                        entry->name, total_oh,
                                        engine->cfg.pgo_max_overhead_ms,
                                        prof_ms, compile_oh);
                                /* fall through to normal O3 enqueue */
                                goto pgo_skip_to_o3;
                            }
                        }

                        /* Redirect: issue PGO_GENERATE instead of plain O3. */
                        bool expected_pgo = false;
                        if (!atomic_compare_exchange_strong_explicit(
                                &entry->in_queue, &expected_pgo, true,
                                memory_order_relaxed, memory_order_relaxed))
                            continue;
                        compile_task_t pgo_gen_task = {
                            .func_id      = entry->id,
                            .target_level = OPT_O2, /* same level, re-instrumented */
                            .priority     = 1,
                            .version_req  = atomic_load_explicit(&entry->version,
                                                                  memory_order_relaxed),
                            .call_rate    = (uint64_t)ema_rate[i],
                            .pgo_mode     = PGO_MODE_GENERATE,
                        };
                        if (!engine_enqueue_task(engine, &pgo_gen_task)) {
                            atomic_store_explicit(&entry->in_queue, false,
                                                  memory_order_relaxed);
                        } else {
                            cnt_at_compile[i]  = cur_cnt;
                            last_queued_ms[i]  = now;
                            hot_scan_streak[i] = 0;
                            if (engine->cfg.verbose)
                                fprintf(stderr,
                                    "[cjit/monitor] PGO_GENERATE '%s' queued "
                                    "(rate=%.0fcps)\n",
                                    entry->name, (double)ema_rate[i]);
                        }
                        continue; /* skip normal O3 enqueue below */
                    }
                    /* pgo_st == PGO_STATE_DONE: fall through to normal O3 */
                }
                pgo_skip_to_o3:;
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
                /* Bookkeeping: was this a predictive promotion? */
                if (predicted_hot && !rate_hot && !ns_hot) {
                    atomic_fetch_add_explicit(&engine->stat_predictive_promos, 1,
                                              memory_order_relaxed);
                }
                if (engine->cfg.verbose)
                    fprintf(stderr,
                            "[cjit/monitor] enqueued '%s' for O%d "
                            "(ema=%.0fcps ema_ns=%.0fns/s streak=%u/%u rc=%u cooloff=%ums%s)\n",
                            entry->name, (int)target,
                            (double)ema_rate[i],
                            (double)ema_ns_per_sec[i],
                            hot_scan_streak[i], req_streak,
                            rc, effective_cooloff,
                            (predicted_hot && !rate_hot) ? " [predictive]" : "");
            }
        }

        /*
         * ── PGO second pass: flush profile data and schedule PGO_USE ──────
         *
         * Walk all registered functions and check whether any instrumented
         * (PGO_STATE_RUNNING) function has accumulated enough profiling calls.
         * When it has:
         *   1. Call _cjit_pgo_flush() on the instrumented handle to write
         *      the .gcda files to disk.
         *   2. Enqueue a PGO_USE task (OPT_O3, high priority).
         *
         * This pass runs every monitor scan cycle but only touches entries
         * whose pgo_state is RUNNING, so it is essentially free in the common
         * case (pgo_state load is relaxed; the branch is almost never taken).
         */
        if (engine->cfg.enable_pgo) {
            uint32_t npgo = (uint32_t)atomic_load_explicit(
                &engine->ftable->count, memory_order_relaxed);
            for (uint32_t pi = 0; pi < npgo; pi++) {
                func_table_entry_t *pe =
                    func_table_get(engine->ftable, (func_id_t)pi);
                if (!pe) continue;
                if (atomic_load_explicit(&pe->pgo_state,
                                          memory_order_relaxed) != PGO_STATE_RUNNING)
                    continue;
                if (atomic_load_explicit(&pe->in_queue,
                                          memory_order_relaxed))
                    continue;

                uint64_t pcnt = atomic_load_explicit(&pe->call_cnt,
                                                      memory_order_relaxed);
                uint64_t need = (engine->cfg.pgo_profile_calls > 0)
                                    ? engine->cfg.pgo_profile_calls : 5000u;
                if (pcnt < pe->pgo_calls_at_start + need) continue;

                /* Enough calls: flush profile data. */
                void *ph = pe->pgo_instr_handle;
                if (ph) {
                    typedef void (*pgo_flush_fn_t)(void);
                    pgo_flush_fn_t flush_fn = (pgo_flush_fn_t)(uintptr_t)
                        dlsym(ph, "_cjit_pgo_flush");
                    if (flush_fn) {
                        flush_fn();
                        if (engine->cfg.verbose) {
                            /* Compute observed profiling overhead:
                             * overhead% = (instr_avg - pre_avg) / pre_avg * 100 */
                            uint64_t now_cnt = atomic_load_explicit(
                                &pe->call_cnt, memory_order_relaxed);
                            uint64_t now_ns  = atomic_load_explicit(
                                &pe->total_elapsed_ns, memory_order_relaxed);
                            uint64_t instr_avg_ns = (now_cnt > 0)
                                                    ? (now_ns / now_cnt) : 0;
                            uint64_t pre = pe->pgo_pre_instr_avg_ns;
                            long overhead_pct = (pre > 0)
                                ? (long)(((int64_t)instr_avg_ns - (int64_t)pre)
                                         * 100 / (int64_t)pre) : 0;
                            fprintf(stderr,
                                    "[cjit/monitor] PGO flushed '%s' "
                                    "(%llu calls, overhead ~%+ld%%)\n",
                                    pe->name,
                                    (unsigned long long)(pcnt - pe->pgo_calls_at_start),
                                    overhead_pct);
                        }
                    }
                }

                /* Enqueue PGO_USE (OPT_O3, high priority). */
                bool exp_pu = false;
                if (atomic_compare_exchange_strong_explicit(
                        &pe->in_queue, &exp_pu, true,
                        memory_order_relaxed, memory_order_relaxed)) {
                    compile_task_t use_task = {
                        .func_id      = (func_id_t)pi,
                        .target_level = OPT_O3,
                        .priority     = 2,
                        .version_req  = atomic_load_explicit(&pe->version,
                                                              memory_order_relaxed),
                        .call_rate    = 0,
                        .pgo_mode     = PGO_MODE_USE,
                    };
                    if (!engine_enqueue_task(engine, &use_task))
                        atomic_store_explicit(&pe->in_queue, false,
                                              memory_order_relaxed);
                    else if (engine->cfg.verbose)
                        fprintf(stderr,
                                "[cjit/monitor] PGO_USE '%s' queued\n",
                                pe->name);
                }
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

    /* Tier-skip and predictive promotion (disabled by default; opt-in). */
    cfg.tier_skip_multiplier          = 0.0f;
    cfg.prediction_lookahead_cycles   = 0;

    /* CPU-time-based tier promotion (disabled by default; opt-in). */
    cfg.cpu_hot_ns_per_sec_t1 = CJIT_DEFAULT_CPU_HOT_NS_T1;
    cfg.cpu_hot_ns_per_sec_t2 = CJIT_DEFAULT_CPU_HOT_NS_T2;

    cfg.enable_inlining      = true;
    cfg.enable_vectorization = true;
    cfg.enable_loop_unroll   = true;
    cfg.enable_const_fold    = true;
    cfg.enable_native_arch   = true;
    cfg.enable_fast_math     = false;
    cfg.verbose              = false;

    /* O1 warm-up tier (disabled by default; opt-in). */
    cfg.enable_o1_warmup = false;
    cfg.warm_rate_t0     = 0;  /* auto: hot_rate_t1 / 4 */

    /* PGO (disabled by default; opt-in). */
    cfg.enable_pgo        = false;
    cfg.pgo_profile_calls = 5000;
    cfg.pgo_max_overhead_ms = 2000;

    /* Compiler thread CPU affinity (disabled by default; opt-in). */
    cfg.pin_compiler_threads = false;

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

    /* Persistent compiled-artifact cache (optional – NULL if disabled). */
    e->artifact_cache = NULL;
    if (e->cfg.cache_dir[0]) {
        e->artifact_cache = codegen_cache_create(e->cfg.cache_dir);
        /*
         * A failed cache_create (e.g. permission denied, disk full) is non-
         * fatal: we continue without caching.  Each compilation falls back to
         * the normal compile path; correctness is unaffected.
         */
    }

    /* Per-thread work queues: 2 per thread (normal lane + priority lane). */
    e->work_queues = calloc(2 * (size_t)e->cfg.compiler_threads, sizeof(mpmc_queue_t));
    if (!e->work_queues) {
        codegen_cache_destroy(e->artifact_cache);
        ir_cache_destroy(e->ir_cache);
        func_table_destroy(e->ftable);
        free(e);
        return NULL;
    }
    for (uint32_t i = 0; i < 2 * e->cfg.compiler_threads; ++i)
        mpmc_init(&e->work_queues[i]);

    /* Condition variable for instant compiler wake-up. */
    pthread_mutex_init(&e->work_cond_mutex, NULL);
    pthread_cond_init(&e->work_cond, NULL);

    /* Condition variable signaled after each compilation attempt. */
    pthread_mutex_init(&e->compile_done_mutex, NULL);
    pthread_cond_init(&e->compile_done_cond, NULL);

    /* Compile-event callback (starts with no callback registered). */
    pthread_mutex_init(&e->cb_mutex, NULL);
    e->compile_cb          = NULL;
    e->compile_cb_userdata = NULL;

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

    atomic_init(&e->running,               false);
    atomic_init(&e->stop_requested,        false);
    atomic_init(&e->stat_compilations,      0);
    atomic_init(&e->stat_failed,            0);
    atomic_init(&e->stat_timeouts,          0);
    atomic_init(&e->stat_swaps,             0);
    atomic_init(&e->stat_tier_skips,        0);
    atomic_init(&e->stat_predictive_promos, 0);
    atomic_init(&e->stat_pgo_cycles,        0);

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
#ifdef __linux__
        /*
         * Optionally pin each compiler thread to a CPU core.
         *
         * Pinning prevents thread migration between cores, which would
         * otherwise cause cold cache misses in the compiler thread's working
         * set (IR text, argv arrays, temp-file paths).  Each compiler thread
         * is assigned to core (i % ncpu) so threads spread across available
         * CPUs when there are more CPUs than compiler threads.
         */
        if (engine->cfg.pin_compiler_threads) {
            long ncpu_pin = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncpu_pin > 0) {
                cpu_set_t cs_pin;
                CPU_ZERO(&cs_pin);
                CPU_SET((int)((unsigned long)i % (unsigned long)ncpu_pin),
                        &cs_pin);
                pthread_setaffinity_np(engine->compiler_threads[i],
                                       sizeof(cs_pin), &cs_pin);
            }
        }
#endif
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
    codegen_cache_destroy(engine->artifact_cache);   /* NULL-safe */

    pthread_cond_destroy(&engine->work_cond);
    pthread_mutex_destroy(&engine->work_cond_mutex);

    pthread_cond_destroy(&engine->compile_done_cond);
    pthread_mutex_destroy(&engine->compile_done_mutex);
    pthread_mutex_destroy(&engine->cb_mutex);

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

size_t cjit_register_from_source(cjit_engine_t *engine,
                                  const char    *source,
                                  size_t         n,
                                  const char *const names[],
                                  func_id_t      ids_out[])
{
    if (!engine || !source || !names || n == 0) {
        if (ids_out) {
            for (size_t k = 0; k < n; ++k)
                ids_out[k] = CJIT_INVALID_FUNC_ID;
        }
        return 0;
    }

    size_t registered = 0;
    for (size_t k = 0; k < n; ++k) {
        func_id_t id = CJIT_INVALID_FUNC_ID;
        if (names[k])
            id = cjit_register_function(engine, names[k], source, NULL);
        if (ids_out)
            ids_out[k] = id;
        if (id != CJIT_INVALID_FUNC_ID)
            registered++;
    }
    return registered;
}

jit_func_t cjit_get_func_by_name(cjit_engine_t *engine, const char *name)
{
    if (!engine || !name) return NULL;
    func_id_t id = cjit_lookup_function(engine, name);
    if (id == CJIT_INVALID_FUNC_ID) return NULL;
    return cjit_get_func(engine, id);
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
     * Full batches have already been added to call_cnt by the threshold-compare
     * path.  Only the sub-threshold remainder is pending here.
     *
     * cjit_tls_elapsed[i] holds accumulated nanoseconds from timed dispatches;
     * it is flushed unconditionally when non-zero regardless of cjit_tls_counts.
     *
     * We scan only registered functions (count ≤ CJIT_MAX_FUNCTIONS), so the
     * TLS array accesses are always in bounds.
     */
    uint32_t n = atomic_load_explicit(&engine->ftable->count,
                                       memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t  rem     = cjit_tls_counts[i];
        uint64_t elapsed = cjit_tls_elapsed[i];
        if (rem == 0 && elapsed == 0) continue;   /* nothing pending */
        func_table_entry_t *e = &engine->ftable->entries[i];
        if (rem > 0) {
            cjit_tls_counts[i] = 0;
            atomic_fetch_add_explicit(&e->call_cnt, rem, memory_order_relaxed);
        }
        if (elapsed > 0) {
            cjit_tls_elapsed[i] = 0;
            atomic_fetch_add_explicit(&e->total_elapsed_ns, elapsed,
                                      memory_order_relaxed);
        }
    }
}

void cjit_record_timed_call(cjit_engine_t *engine,
                              func_id_t      id,
                              uint64_t       elapsed_ns)
{
    if (__builtin_expect(id >= CJIT_MAX_FUNCTIONS, 0)) return;
    /*
     * Accumulate elapsed nanoseconds in TLS (zero shared-memory traffic on the
     * common path).  The count and elapsed buffers share the same flush trigger:
     * when cjit_tls_counts reaches CJIT_TLS_FLUSH_THRESHOLD, both are flushed
     * to the shared atomics with one call_cnt fetch_add and one (conditional)
     * total_elapsed_ns fetch_add.  This adds at most one extra atomic operation
     * per THRESHOLD calls compared to the non-timed path.
     *
     * The histogram bucket for the average per-call latency of this batch is
     * also updated at the flush boundary — one additional atomic per THRESHOLD
     * calls, zero overhead on the common (non-flush) path.
     */
    cjit_tls_elapsed[id] += elapsed_ns;
    if (__builtin_expect(++cjit_tls_counts[id] >= CJIT_TLS_FLUSH_THRESHOLD, 0)) {
        cjit_tls_counts[id] = 0;
        uint64_t acc = cjit_tls_elapsed[id];
        cjit_tls_elapsed[id] = 0;
        func_table_entry_t *e = func_table_get(engine->ftable, id);
        if (__builtin_expect(e != NULL, 1)) {
            atomic_fetch_add_explicit(&e->call_cnt, CJIT_TLS_FLUSH_THRESHOLD,
                                      memory_order_relaxed);
            if (acc > 0) {
                atomic_fetch_add_explicit(&e->total_elapsed_ns, acc,
                                          memory_order_relaxed);
                /* Update histogram: bucket = floor(log₂(avg_ns)).
                 * 63 - clz(x) computes the position of the most-significant set
                 * bit (0-indexed from LSB), which equals floor(log₂(x)) for
                 * any x ≥ 1.  The result is clamped to [0, CJIT_HIST_BUCKETS). */
                uint64_t avg_ns = acc / CJIT_TLS_FLUSH_THRESHOLD;
                int bucket = (avg_ns == 0) ? 0
                    : (int)(63u - (unsigned)__builtin_clzll(avg_ns));
                if (bucket >= CJIT_HIST_BUCKETS)
                    bucket = CJIT_HIST_BUCKETS - 1;
                atomic_fetch_add_explicit(&e->hist_counts[bucket], 1u,
                                          memory_order_relaxed);
            }
        }
    }
}

void cjit_get_histogram(const cjit_engine_t *engine,
                        func_id_t            id,
                        uint64_t             out[CJIT_HIST_BUCKETS])
{
    for (int i = 0; i < CJIT_HIST_BUCKETS; i++) out[i] = 0;
    if (!engine) return;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return;
    for (int i = 0; i < CJIT_HIST_BUCKETS; i++)
        out[i] = (uint64_t)atomic_load_explicit(&e->hist_counts[i],
                                                 memory_order_relaxed);
}

uint64_t cjit_percentile_ns(const cjit_engine_t *engine,
                             func_id_t            id,
                             unsigned             pct)
{
    uint64_t counts[CJIT_HIST_BUCKETS];
    cjit_get_histogram(engine, id, counts);

    uint64_t total = 0;
    for (int i = 0; i < CJIT_HIST_BUCKETS; i++) total += counts[i];
    if (total == 0 || pct > 100) return 0;

    /* Target: ceil(total * pct / 100) to find the bucket that contains pct%. */
    uint64_t target = (total * (uint64_t)pct + 99) / 100;
    uint64_t cum = 0;
    for (int i = 0; i < CJIT_HIST_BUCKETS; i++) {
        cum += counts[i];
        if (cum >= target) {
            /*
             * Return the upper bound of bucket i: 2^i nanoseconds.
             * Bucket 0 covers [0,1) ns → return 1.
             * Bucket k covers [2^(k-1), 2^k) ns → return 2^k.
             */
            return (i == 0) ? 1ULL : (UINT64_C(1) << i);
        }
    }
    return UINT64_C(1) << (CJIT_HIST_BUCKETS - 1);
}

uint64_t cjit_get_elapsed_ns(const cjit_engine_t *engine, func_id_t id)
{
    if (!engine) return 0;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return 0;
    return atomic_load_explicit(&e->total_elapsed_ns, memory_order_relaxed);
}

void cjit_record_arg_samples(cjit_engine_t  *engine,
                               func_id_t       id,
                               uint8_t         n_args,
                               const uint64_t *vals)
{
    if (__builtin_expect(!engine || id >= CJIT_MAX_FUNCTIONS || n_args == 0, 0))
        return;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (__builtin_expect(!e, 0)) return;

    cjit_arg_profile_t *prof = &e->arg_profile;
    if (n_args > CJIT_MAX_PROFILED_ARGS) n_args = CJIT_MAX_PROFILED_ARGS;

    /*
     * Update n_profiled on first observation.  A plain (non-atomic) write is
     * safe here: the value only ever increases and is read by the compiler
     * thread under a roughly once-per-compilation basis, where a stale zero
     * simply means no specialisation is attempted on that recompile.
     */
    if (n_args > prof->n_profiled) prof->n_profiled = n_args;

    for (uint8_t i = 0; i < n_args; ++i)
        cjit_arg_slot_update(&prof->slots[i], vals[i]);
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
    s.compile_timeouts     = atomic_load_explicit(&engine->stat_timeouts,
                                                   memory_order_relaxed);
    s.total_swaps          = atomic_load_explicit(&engine->stat_swaps,
                                                   memory_order_relaxed);
    s.retired_handles      = atomic_load_explicit(&engine->dgc.total_retired,
                                                   memory_order_relaxed);
    s.freed_handles        = atomic_load_explicit(&engine->dgc.total_freed,
                                                   memory_order_relaxed);

    /* Sum queue depths across both lanes for all threads. */
    uint32_t qd       = 0;
    uint32_t qd_prio_depth = 0;
    uint32_t max_rc = 0;
    uint64_t total_elapsed = 0;
    uint32_t nf  = s.registered_functions;
    for (uint32_t i = 0; i < engine->cfg.compiler_threads; ++i) {
        qd           += mpmc_size(&engine->work_queues[2 * i + 0]);
        qd_prio_depth += mpmc_size(&engine->work_queues[2 * i + 1]);
    }
    for (uint32_t i = 0; i < nf; ++i) {
        uint32_t rc = atomic_load_explicit(
            &engine->ftable->entries[i].recompile_count, memory_order_relaxed);
        if (rc > max_rc) max_rc = rc;
        total_elapsed += atomic_load_explicit(
            &engine->ftable->entries[i].total_elapsed_ns, memory_order_relaxed);
    }
    s.queue_depth           = qd;
    s.prio_queue_depth      = qd_prio_depth;
    s.max_recompile_count   = max_rc;
    s.total_elapsed_ns      = total_elapsed;
    s.tier_skips            = atomic_load_explicit(&engine->stat_tier_skips,
                                                    memory_order_relaxed);
    s.predictive_promotions = atomic_load_explicit(&engine->stat_predictive_promos,
                                                    memory_order_relaxed);

    if (engine->artifact_cache) {
        s.artifact_cache_hits   = codegen_cache_hits(engine->artifact_cache);
        s.artifact_cache_misses = codegen_cache_misses(engine->artifact_cache);
    }

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
            "║  Compile timeouts      : %6llu            ║\n"
            "║  Atomic pointer swaps  : %6llu            ║\n"
            "║  Handles retired (GC)  : %6llu            ║\n"
            "║  Handles freed   (GC)  : %6llu            ║\n"
            "║  Normal queue depth    : %6u            ║\n"
            "║  Priority queue depth  : %6u            ║\n"
            "║  Max recompile count   : %6u            ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Tier-skips (O0→O3)    : %6llu            ║\n"
            "║  Predictive promotions : %6llu            ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Artifact cache hits   : %6llu            ║\n"
            "║  Artifact cache misses : %6llu            ║\n"
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
            (unsigned long long)s.compile_timeouts,
            (unsigned long long)s.total_swaps,
            (unsigned long long)s.retired_handles,
            (unsigned long long)s.freed_handles,
            s.queue_depth,
            s.prio_queue_depth,
            s.max_recompile_count,
            (unsigned long long)s.tier_skips,
            (unsigned long long)s.predictive_promotions,
            (unsigned long long)s.artifact_cache_hits,
            (unsigned long long)s.artifact_cache_misses,
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

func_id_t cjit_lookup_function(const cjit_engine_t *engine, const char *name)
{
    if (!engine || !name) return CJIT_INVALID_FUNC_ID;
    uint32_t n = atomic_load_explicit(&engine->ftable->count,
                                       memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
        if (strcmp(engine->ftable->entries[i].name, name) == 0)
            return (func_id_t)i;
    }
    return CJIT_INVALID_FUNC_ID;
}

bool cjit_update_ir(cjit_engine_t *engine,
                    func_id_t      id,
                    const char    *new_ir,
                    opt_level_t    level)
{
    if (!engine || !new_ir) return false;
    func_table_entry_t *entry = func_table_get(engine->ftable, id);
    if (!entry) return false;

    /* Update the IR cache (updates both in-memory copy and on-disk backup). */
    if (engine->ir_cache)
        ir_cache_update_ir(engine->ir_cache, id, entry->name, new_ir);

    /*
     * Bump the version counter so any task already queued against the old IR
     * is detected as stale by compiler threads and silently discarded.
     *
     * Reset cur_level to OPT_NONE so that the compiler thread's tier-skip
     * check (cur_level >= task.target_level) never fires — we always want to
     * recompile after an IR update regardless of the previous tier.
     */
    atomic_fetch_add_explicit(&entry->version, 1, memory_order_release);
    atomic_store_explicit(&entry->cur_level, (int)OPT_NONE, memory_order_release);

    /* Request recompilation at the specified level. */
    cjit_request_recompile(engine, id, level);
    return true;
}

bool cjit_wait_compiled(cjit_engine_t *engine,
                        func_id_t      id,
                        uint32_t       timeout_ms)
{
    if (!engine) return false;
    func_table_entry_t *entry = func_table_get(engine->ftable, id);
    if (!entry) return false;

    /* Fast path: already compiled (or AOT fallback is set). */
    if (atomic_load_explicit(&entry->func_ptr, memory_order_acquire) != NULL)
        return true;

    if (timeout_ms == 0)
        return false;

    /* Compute absolute deadline for pthread_cond_timedwait. */
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec  += (time_t)(timeout_ms / 1000);
    deadline.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&engine->compile_done_mutex);
    while (atomic_load_explicit(&entry->func_ptr, memory_order_acquire) == NULL) {
        int rc = pthread_cond_timedwait(&engine->compile_done_cond,
                                        &engine->compile_done_mutex,
                                        &deadline);
        if (rc == ETIMEDOUT)
            break;
    }
    bool ready = (atomic_load_explicit(&entry->func_ptr, memory_order_acquire)
                  != NULL);
    pthread_mutex_unlock(&engine->compile_done_mutex);
    return ready;
}

/* ══════════════════════════ Feature D: compile-event callback ═════════════ */

void cjit_set_compile_callback(cjit_engine_t           *engine,
                                cjit_compile_callback_t  cb,
                                void                    *userdata)
{
    if (!engine) return;
    pthread_mutex_lock(&engine->cb_mutex);
    engine->compile_cb          = cb;
    engine->compile_cb_userdata = userdata;
    pthread_mutex_unlock(&engine->cb_mutex);
}

/* ══════════════════════════ Feature E: function pinning ════════════════════ */

bool cjit_pin_function(cjit_engine_t *engine, func_id_t id)
{
    if (!engine) return false;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return false;
    atomic_store_explicit(&e->pinned, true, memory_order_relaxed);
    return true;
}

bool cjit_unpin_function(cjit_engine_t *engine, func_id_t id)
{
    if (!engine) return false;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return false;
    atomic_store_explicit(&e->pinned, false, memory_order_relaxed);
    return true;
}

bool cjit_is_pinned(const cjit_engine_t *engine, func_id_t id)
{
    if (!engine) return false;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return false;
    return atomic_load_explicit(&e->pinned, memory_order_relaxed);
}

/* ══════════════════════════ Feature F: IR snapshot export ══════════════════ */

int cjit_snapshot_ir(cjit_engine_t *engine, const char *dir)
{
    if (!engine || !dir) return -1;

    /* Create the directory if it does not exist (mode 0700). */
    if (mkdir(dir, 0700) != 0 && errno != EEXIST)
        return -1;

    uint32_t nf = atomic_load_explicit(&engine->ftable->count,
                                        memory_order_acquire);
    if (nf == 0) return 0;

    /* Open the manifest file first; if this fails return error. */
    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", dir);
    FILE *mf = fopen(manifest_path, "w");
    if (!mf) return -1;

    static const char *gen_names[] = { "HOT", "WARM", "COLD" };

    int written = 0;
    for (uint32_t i = 0; i < nf; i++) {
        func_table_entry_t *entry = &engine->ftable->entries[i];

        /* Retrieve IR from the cache (promotes COLD → WARM as a side effect;
         * returns a heap copy that we must free). */
        char *ir = NULL;
        if (engine->ir_cache)
            ir = ir_cache_get_ir(engine->ir_cache, (func_id_t)i);

        if (!ir) {
            /* Fall back to the pointer in the function-table entry (may be
             * the original registration pointer if not yet cached on disk). */
            ir = engine->ftable->entries[i].ir_source
                     ? strdup(engine->ftable->entries[i].ir_source)
                     : NULL;
        }

        /* Write the .c file. */
        if (ir) {
            char fpath[512];
            snprintf(fpath, sizeof(fpath), "%s/%s.c", dir, entry->name);
            FILE *fp = fopen(fpath, "w");
            if (fp) {
                fputs(ir, fp);
                fputc('\n', fp);
                fclose(fp);
                written++;
            }
            free(ir);
        }

        /* Manifest line: id  name  gen  call_cnt  opt_level */
        uint8_t gen = engine->ir_cache
            ? ir_cache_get_generation(engine->ir_cache, (func_id_t)i)
            : (uint8_t)0;
        if (gen > 2) gen = 2;
        uint64_t calls = atomic_load_explicit(&entry->call_cnt,
                                               memory_order_relaxed);
        int lvl = atomic_load_explicit(&entry->cur_level,
                                        memory_order_relaxed);
        fprintf(mf, "%u\t%s\t%s\t%llu\t%d\n",
                i, entry->name, gen_names[gen],
                (unsigned long long)calls, lvl);
    }

    fclose(mf);
    return written;
}

/* ══════════════════════════ Feature G: per-function stats reset ════════════ */

bool cjit_reset_function_stats(cjit_engine_t *engine, func_id_t id)
{
    if (!engine) return false;
    func_table_entry_t *e = func_table_get(engine->ftable, id);
    if (!e) return false;

    atomic_store_explicit(&e->call_cnt,          0, memory_order_relaxed);
    atomic_store_explicit(&e->total_elapsed_ns,  0, memory_order_relaxed);
    atomic_store_explicit(&e->recompile_count,   0, memory_order_relaxed);

    for (int k = 0; k < CJIT_HIST_BUCKETS; k++)
        atomic_store_explicit(&e->hist_counts[k], 0, memory_order_relaxed);

    return true;
}

/* ══════════════════════════ Feature H: queue drain ════════════════════════ */

bool cjit_drain_queue(cjit_engine_t *engine, uint32_t timeout_ms)
{
    if (!engine) return false;

    uint64_t deadline = (timeout_ms == 0) ? 0 : (engine_now_ms() + timeout_ms);

    do {
        /* Sum all queue depths across both lanes for all compiler threads. */
        uint32_t depth = 0;
        for (uint32_t t = 0; t < engine->cfg.compiler_threads; t++) {
            depth += mpmc_size(&engine->work_queues[2 * t + 0]);
            depth += mpmc_size(&engine->work_queues[2 * t + 1]);
        }

        /* Also check active_compilations: a function may have been dequeued
         * (in_queue cleared) but compilation is still in progress
         * (compile_lock held inside the compiler thread). */
        if (depth == 0 &&
            atomic_load_explicit(&engine->active_compilations,
                                 memory_order_relaxed) == 0)
            return true;

        if (timeout_ms == 0)
            return false;

        struct timespec ts = { .tv_sec = 0, .tv_nsec = 5000000L }; /* 5 ms */
        nanosleep(&ts, NULL);

    } while (engine_now_ms() < deadline);

    /* Final re-check. */
    uint32_t depth = 0;
    for (uint32_t t = 0; t < engine->cfg.compiler_threads; t++) {
        depth += mpmc_size(&engine->work_queues[2 * t + 0]);
        depth += mpmc_size(&engine->work_queues[2 * t + 1]);
    }
    if (depth > 0) return false;
    if (atomic_load_explicit(&engine->active_compilations,
                             memory_order_relaxed) > 0)
        return false;
    return true;
}

/* ══════════════════════════ Feature I: synchronous compile ════════════════ */

bool cjit_compile_sync(cjit_engine_t *engine, func_id_t id, opt_level_t level)
{
    if (!engine) return false;
    func_table_entry_t *entry = func_table_get(engine->ftable, id);
    if (!entry) return false;

    /* Build codegen options identical to the compiler thread. */
    codegen_opts_t copts = {
        .enable_inlining      = engine->cfg.enable_inlining,
        .enable_vectorization = engine->cfg.enable_vectorization,
        .enable_loop_unroll   = engine->cfg.enable_loop_unroll,
        .enable_native_arch   = engine->cfg.enable_native_arch,
        .enable_fast_math     = engine->cfg.enable_fast_math,
        .enable_const_fold    = engine->cfg.enable_const_fold,
        .verbose              = engine->cfg.verbose,
        .extra_cflags         = engine->cfg.extra_cflags[0] ? engine->cfg.extra_cflags : NULL,
        .cc_binary            = engine->cfg.cc_binary[0]    ? engine->cfg.cc_binary    : NULL,
        .cache                = engine->artifact_cache,
        .compile_timeout_ms   = engine->cfg.compile_timeout_ms,
        .arg_profile          = NULL,
    };

    /* Fetch IR (promotes COLD → WARM as a side effect). */
    char       *ir_cache_copy = NULL;
    const char *ir_to_use     = NULL;
    if (engine->ir_cache)
        ir_cache_copy = ir_cache_get_ir(engine->ir_cache, id);
    ir_to_use = ir_cache_copy ? ir_cache_copy : entry->ir_source;

    if (!ir_to_use) {
        free(ir_cache_copy);
        if (engine->cfg.verbose)
            fprintf(stderr, "[cjit/sync] no IR for '%s'\n", entry->name);
        return false;
    }

    /* Take the per-entry compile lock (blocking, unlike trylock in bg thread).
     * This serialises against any concurrent background compilation.
     * Count this as an active compilation so cjit_drain_queue() waits for us. */
    atomic_fetch_add_explicit(&engine->active_compilations, 1,
                              memory_order_relaxed);
    pthread_mutex_lock(&entry->compile_lock);

    /* Attach the current arg profile and zero runtime-profile hints
     * (call_rate and avg_elapsed_ns are unavailable in the sync path). */
    copts.arg_profile    = &entry->arg_profile;
    copts.call_rate      = 0;
    copts.avg_elapsed_ns = 0;

    uint64_t t0 = engine_now_ms();
    codegen_result_t cres;
    bool ok = codegen_compile(entry->name, ir_to_use, level, &copts, &cres);
    copts.arg_profile = NULL;
    free(ir_cache_copy);

    uint32_t dur_ms = (uint32_t)(engine_now_ms() - t0);
    atomic_store_explicit(&entry->last_compile_duration_ms, dur_ms,
                          memory_order_relaxed);

    if (!ok) {
        atomic_fetch_add_explicit(&engine->stat_failed, 1, memory_order_relaxed);
        if (cres.timed_out)
            atomic_fetch_add_explicit(&engine->stat_timeouts, 1,
                                      memory_order_relaxed);
        pthread_mutex_unlock(&entry->compile_lock);
        atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                                  memory_order_relaxed);

        /* Wake cjit_wait_compiled() callers so they can detect the failure. */
        pthread_mutex_lock(&engine->compile_done_mutex);
        pthread_cond_broadcast(&engine->compile_done_cond);
        pthread_mutex_unlock(&engine->compile_done_mutex);

        /* Fire compile-event callback. */
        pthread_mutex_lock(&engine->cb_mutex);
        cjit_compile_callback_t cb = engine->compile_cb;
        void *cb_ud = engine->compile_cb_userdata;
        pthread_mutex_unlock(&engine->cb_mutex);
        if (cb) {
            cjit_compile_event_t ev;
            memset(&ev, 0, sizeof(ev));
            ev.func_id     = id;
            snprintf(ev.func_name, sizeof(ev.func_name), "%s", entry->name);
            ev.level       = level;
            ev.success     = false;
            ev.timed_out   = cres.timed_out;
            ev.cache_hit   = false;
            ev.duration_ms = dur_ms;
            snprintf(ev.errmsg, sizeof(ev.errmsg), "%.255s", cres.errmsg);
            cb(&ev, cb_ud);
        }
        return false;
    }

    atomic_fetch_add_explicit(&engine->stat_compilations, 1, memory_order_relaxed);

    void *old_handle = func_table_swap(engine->ftable, id,
                                        cres.fn, cres.handle, level);
    atomic_fetch_add_explicit(&engine->stat_swaps, 1, memory_order_relaxed);
    pthread_mutex_unlock(&entry->compile_lock);
    atomic_fetch_sub_explicit(&engine->active_compilations, 1,
                              memory_order_relaxed);

    dgc_retire(&engine->dgc, old_handle);

    /* Wake cjit_wait_compiled() callers. */
    pthread_mutex_lock(&engine->compile_done_mutex);
    pthread_cond_broadcast(&engine->compile_done_cond);
    pthread_mutex_unlock(&engine->compile_done_mutex);

    /* Fire compile-event callback (success). */
    pthread_mutex_lock(&engine->cb_mutex);
    cjit_compile_callback_t cb2 = engine->compile_cb;
    void *cb2_ud = engine->compile_cb_userdata;
    pthread_mutex_unlock(&engine->cb_mutex);
    if (cb2) {
        cjit_compile_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.func_id     = id;
        snprintf(ev.func_name, sizeof(ev.func_name), "%s", entry->name);
        ev.level       = level;
        ev.success     = true;
        ev.timed_out   = false;
        ev.cache_hit   = cres.cache_hit;
        ev.duration_ms = dur_ms;
        cb2(&ev, cb2_ud);
    }

    if (engine->cfg.verbose)
        fprintf(stderr,
                "[cjit/sync] compiled '%s' O%d in %u ms\n",
                entry->name, (int)level, dur_ms);

    return true;
}

/* ══════════════════════════ IR cache stats / prefetch ═════════════════════ */

void cjit_print_ir_cache_stats(const cjit_engine_t *engine)
{
    if (!engine || !engine->ir_cache) return;
    ir_cache_print_stats(engine->ir_cache);
}

bool cjit_ir_cache_prefetch(cjit_engine_t *engine, func_id_t id)
{
    if (!engine || !engine->ir_cache) return false;
    if (!func_table_get(engine->ftable, id)) return false;

    /* Try the async path first (io_threads > 0 in the IR cache). */
    if (ir_cache_prefetch(engine->ir_cache, id))
        return true;

    /* Fall back to synchronous: read IR into warm memory now. */
    char *ir = ir_cache_get_ir(engine->ir_cache, id);
    if (!ir) return false;
    free(ir);
    return true;
}
