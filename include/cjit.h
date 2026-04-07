/**
 * cjit.h – Public API for the non-blocking JIT compiler engine.
 *
 * Design overview
 * ───────────────
 * The engine maintains a fully-atomic function-pointer table (func_table).
 * Runtime threads read function pointers with a SINGLE atomic load and then
 * dispatch directly – no locks, no reference counts, no memory barriers
 * beyond that one load.
 *
 * A pool of CJIT_COMPILER_THREADS background threads continuously picks
 * compile_task items from a lock-free MPMC work-queue, invokes the
 * system C compiler on the function's IR (C source), loads the resulting
 * shared object with dlopen/dlsym, and atomically swaps the entry in the
 * function table.
 *
 * One monitoring thread samples per-entry call-counters, detects hot
 * functions, and enqueues them at increasing optimization levels.
 *
 * Memory safety for retired function handles is provided by a grace-period
 * deferred-free scheme: handles are placed on a lock-free retire stack
 * together with a retirement timestamp and are only dlclose'd after a
 * configurable grace period has elapsed.  This guarantees that no thread
 * still executing through old code touches freed memory.
 *
 * Hot path (per call):
 *   jit_func_t f = atomic_load_explicit(ptr, memory_order_acquire);  // (1)
 *   f(args…);                                                          // (2)
 *
 * Thread safety
 * ─────────────
 *   • Function pointer reads  : atomic_load  (acquire)
 *   • Function pointer writes : atomic_store (release) performed only by
 *                               compiler threads after full compilation.
 *   • Call-count increments   : atomic_fetch_add (relaxed) by runtime
 *                               threads via cjit_record_call().
 *   • Work queue              : lock-free MPMC (Dmitry Vyukov's algorithm).
 *   • Retire stack            : lock-free LIFO (CAS on head pointer).
 *
 * Written in C11; requires POSIX threads and dlopen.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════ tuneable constants ═══════════════════════════ */

/** Maximum number of concurrently registered functions. */
#define CJIT_MAX_FUNCTIONS      1024

/** Number of background compilation threads. */
#define CJIT_COMPILER_THREADS   3

/** Number of calls before a function is considered "hot" (tier-1). */
#define CJIT_HOT_THRESHOLD_T1   500ULL

/** Number of calls before a function is escalated to tier-2 (aggressive). */
#define CJIT_HOT_THRESHOLD_T2   5000ULL

/** Milliseconds to keep a retired function handle alive before dlclose. */
#define CJIT_GRACE_PERIOD_MS    100

/* ══════════════════════════════ public types ══════════════════════════════ */

/** Opaque JIT engine handle. */
typedef struct cjit_engine cjit_engine_t;

/**
 * Generic zero-argument function pointer stored in the table.
 *
 * In practice callers cast to a concrete prototype; the table itself stores
 * only void(*)(void) to keep the table type-uniform.
 */
typedef void (*jit_func_t)(void);

/** Stable numeric identifier for a registered function (0-based index). */
typedef uint32_t func_id_t;

/** Sentinel value indicating an invalid function ID. */
#define CJIT_INVALID_FUNC_ID ((func_id_t)UINT32_MAX)

/**
 * System memory pressure level, computed from /proc/meminfo by the IR cache's
 * background pressure-monitor thread.
 *
 * Affects the effective HOT/WARM capacity of the IR LRU cache:
 *   NORMAL   → 100% of configured capacity
 *   MEDIUM   →  75%
 *   HIGH     →  50%
 *   CRITICAL →  25% (never below 1)
 */
typedef enum {
    MEM_PRESSURE_NORMAL   = 0,  /**< Plenty of memory available.              */
    MEM_PRESSURE_MEDIUM   = 1,  /**< Some pressure; cache capacity reduced.   */
    MEM_PRESSURE_HIGH     = 2,  /**< High pressure; aggressive eviction.      */
    MEM_PRESSURE_CRITICAL = 3,  /**< Very low memory; nearly all IR on disk.  */
} mem_pressure_t;

/**
 * Optimisation tier requested for recompilation.
 *
 * Maps directly to compiler -O flags; higher tiers also enable additional
 * flags such as -funroll-loops, -ftree-vectorize, -march=native.
 */
typedef enum {
    OPT_NONE = 0,   /**< -O0  – unoptimised baseline / AOT fall-back         */
    OPT_O1   = 1,   /**< -O1  – basic optimisations                          */
    OPT_O2   = 2,   /**< -O2  – standard optimisations + inlining            */
    OPT_O3   = 3,   /**< -O3  – aggressive: unroll, vectorise, march=native  */
} opt_level_t;

/**
 * Configuration passed to cjit_create().
 *
 * All fields have sensible defaults; use cjit_default_config() to obtain
 * them and then override only the fields you care about.
 */
typedef struct {
    uint32_t max_functions;       /**< Maximum functions (≤ CJIT_MAX_FUNCTIONS). */
    uint32_t compiler_threads;    /**< Background compiler threads.              */
    uint64_t hot_threshold_t1;    /**< Calls before tier-1 optimisation.         */
    uint64_t hot_threshold_t2;    /**< Calls before tier-2 optimisation.         */
    uint32_t grace_period_ms;     /**< Grace period for deferred dlclose (ms).   */
    uint32_t monitor_interval_ms; /**< How often monitor thread wakes (ms).      */
    bool     enable_inlining;     /**< Pass -finline-functions to compiler.       */
    bool     enable_vectorization;/**< Pass -ftree-vectorize to compiler.         */
    bool     enable_loop_unroll;  /**< Pass -funroll-loops to compiler.           */
    bool     enable_const_fold;   /**< Constant folding (enabled at -O1+).       */
    bool     enable_native_arch;  /**< Pass -march=native at OPT_O3.             */
    bool     verbose;             /**< Print compilation events to stderr.        */

    /* ── IR LRU cache settings ──────────────────────────────────────────── */
    uint32_t hot_ir_cache_size;   /**< Max HOT-gen IR entries in memory (def 64). */
    uint32_t warm_ir_cache_size;  /**< Max WARM-gen IR entries in memory (def 128).*/
    char     ir_disk_dir[256];    /**< On-disk IR directory (empty = auto-create). */

    /* ── Memory-pressure monitoring ─────────────────────────────────────── */
    uint32_t mem_pressure_check_ms;     /**< /proc/meminfo poll interval (ms).  */
    uint32_t mem_pressure_low_pct;      /**< % avail → MEDIUM  (def 20).        */
    uint32_t mem_pressure_high_pct;     /**< % avail → HIGH    (def 10).        */
    uint32_t mem_pressure_critical_pct; /**< % avail → CRITICAL (def  5).       */
} cjit_config_t;

/* ══════════════════════════════ runtime stats ═════════════════════════════ */

/** Snapshot of JIT-engine statistics. */
typedef struct {
    /* ── JIT engine core ──────────────────────────────────────────────── */
    uint32_t registered_functions;  /**< Total functions registered.            */
    uint64_t total_compilations;    /**< Total successful compilations.         */
    uint64_t failed_compilations;   /**< Total failed compilations.             */
    uint64_t total_swaps;           /**< Total atomic pointer swaps performed.  */
    uint64_t retired_handles;       /**< Total handles enqueued for deferred GC.*/
    uint64_t freed_handles;         /**< Total handles already freed.           */
    uint32_t queue_depth;           /**< Current depth of the compile queue.    */

    /* ── IR LRU cache ────────────────────────────────────────────────── */
    uint32_t ir_hot_count;          /**< Entries in HOT  generation (memory).  */
    uint32_t ir_warm_count;         /**< Entries in WARM generation (memory).  */
    uint32_t ir_cold_count;         /**< Entries in COLD generation (disk).    */
    uint64_t ir_disk_writes;        /**< IR files written to disk.             */
    uint64_t ir_disk_reads;         /**< IR files loaded from disk.            */
    uint64_t ir_evictions;          /**< Total LRU evictions (HOT/WARM→lower). */
    uint64_t ir_promotions;         /**< Total promotions (COLD/WARM→higher).  */
    uint64_t ir_cache_hits;         /**< get_ir() satisfied from memory.       */
    uint64_t ir_cache_misses;       /**< get_ir() required disk load.          */
    uint64_t ir_pressure_evictions; /**< Evictions triggered by memory pressure*/

    /* ── Memory pressure ─────────────────────────────────────────────── */
    mem_pressure_t mem_pressure;    /**< Current pressure level.               */
    uint64_t mem_available_mb;      /**< Last observed MemAvailable (MB).      */
    uint64_t mem_total_mb;          /**< Last observed MemTotal (MB).          */
} cjit_stats_t;

/* ══════════════════════════════ public API ════════════════════════════════ */

/**
 * Return a cjit_config_t pre-filled with the default values.
 *
 * Defaults mirror the CJIT_* compile-time constants above.
 */
cjit_config_t cjit_default_config(void);

/**
 * Create and initialise a JIT engine.
 *
 * Does NOT start background threads; call cjit_start() to begin compilation
 * and monitoring.
 *
 * @param config  Engine configuration; may be NULL to use defaults.
 * @return        Pointer to the new engine, or NULL on failure.
 */
cjit_engine_t *cjit_create(const cjit_config_t *config);

/**
 * Destroy the JIT engine.
 *
 * Calls cjit_stop() if threads are still running, waits for all background
 * threads to exit, frees all memory, and dlclose's any loaded handles
 * (bypassing the grace period for shutdown).
 */
void cjit_destroy(cjit_engine_t *engine);

/**
 * Start the background compiler and monitor threads.
 *
 * Must be called after cjit_create() and before the runtime dispatch loop.
 * Safe to call only once per engine.
 */
void cjit_start(cjit_engine_t *engine);

/**
 * Signal background threads to stop and wait for them to exit.
 *
 * After cjit_stop() returns the engine is quiescent and can be inspected or
 * destroyed.  It is not safe to restart a stopped engine.
 */
void cjit_stop(cjit_engine_t *engine);

/**
 * Register a function with the JIT engine.
 *
 * @param engine       The engine.
 * @param name         Unique function name (used as the symbol name in dlsym).
 * @param ir_source    C source code of the function (the JIT's IR).
 * @param aot_fallback Compiled-AOT function pointer used before any JIT
 *                     compilation has completed (may be NULL).
 * @return             Stable func_id_t for this function, or
 *                     CJIT_INVALID_FUNC_ID on failure.
 *
 * Thread safety: safe to call before cjit_start(); do not call concurrently
 * with itself (registration is single-threaded setup).
 */
func_id_t cjit_register_function(cjit_engine_t *engine,
                                  const char    *name,
                                  const char    *ir_source,
                                  jit_func_t     aot_fallback);

/**
 * Retrieve the current function pointer for the given ID.
 *
 * This is the HOT PATH.  The implementation is a single
 * atomic_load_explicit(..., memory_order_acquire).
 *
 * The returned pointer is guaranteed to remain valid for at least
 * CJIT_GRACE_PERIOD_MS milliseconds even if it is replaced concurrently.
 *
 * @param engine  The engine.
 * @param id      Function ID returned by cjit_register_function().
 * @return        Current function pointer; never NULL after registration.
 */
jit_func_t cjit_get_func(cjit_engine_t *engine, func_id_t id);

/**
 * Record a call to a function (updates the hot-function counter).
 *
 * Uses a single relaxed atomic_fetch_add – effectively free on modern CPUs.
 * Call this immediately after cjit_get_func() in the dispatch loop if you
 * want the monitor thread to track call frequency.
 *
 * @param engine  The engine.
 * @param id      Function ID.
 */
void cjit_record_call(cjit_engine_t *engine, func_id_t id);

/**
 * Convenience macro: atomic dispatch with a single load + indirect call.
 *
 * Usage:
 *   typedef int (*add_fn_t)(int, int);
 *   int result = CJIT_DISPATCH(engine, id, add_fn_t, a, b);
 *
 * The macro expands to exactly:
 *   ((CastType)cjit_get_func(engine, id))(args…)
 * followed by cjit_record_call().
 */
#define CJIT_DISPATCH(engine, id, cast_type, ...)                    \
    ( cjit_record_call((engine), (id)),                              \
      ((cast_type)cjit_get_func((engine), (id)))(__VA_ARGS__) )

/**
 * Forcibly enqueue a function for recompilation at the given tier.
 *
 * If the function is already enqueued at an equal or higher tier, this is a
 * no-op.  Useful for pre-warming the JIT before the hot loop starts.
 *
 * @param engine  The engine.
 * @param id      Function ID.
 * @param level   Target optimisation level.
 */
void cjit_request_recompile(cjit_engine_t *engine,
                             func_id_t      id,
                             opt_level_t    level);

/**
 * Return a snapshot of engine-wide statistics.
 *
 * All fields are read with relaxed atomics; the snapshot is not guaranteed
 * to be self-consistent across fields but is safe to read at any time.
 */
cjit_stats_t cjit_get_stats(const cjit_engine_t *engine);

/**
 * Print a formatted statistics summary to stderr.
 *
 * Convenience wrapper around cjit_get_stats().
 */
void cjit_print_stats(const cjit_engine_t *engine);

/**
 * Return the current call count for a registered function.
 */
uint64_t cjit_get_call_count(const cjit_engine_t *engine, func_id_t id);

/**
 * Return the optimisation level that is currently active for the function.
 */
opt_level_t cjit_get_current_opt_level(const cjit_engine_t *engine,
                                        func_id_t            id);

#ifdef __cplusplus
}
#endif
