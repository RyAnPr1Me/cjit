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
#include <time.h>       /* struct timespec, clock_gettime (for cjit_timestamp_ns) */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════ tuneable constants ═══════════════════════════ */

/** Maximum number of concurrently registered functions. */
#define CJIT_MAX_FUNCTIONS      1024

/**
 * Maximum number of background compilation threads.
 *
 * The actual number started is set by cjit_config_t.compiler_threads and
 * defaults to (nproc - 1) as detected by cjit_default_config().  This
 * constant is the hard upper bound; values above it are clamped.
 */
#define CJIT_COMPILER_THREADS   16

/** Number of calls before a function is considered "hot" (tier-1). */
#define CJIT_HOT_THRESHOLD_T1   500ULL

/** Number of calls before a function is escalated to tier-2 (aggressive). */
#define CJIT_HOT_THRESHOLD_T2   5000ULL

/** Milliseconds to keep a retired function handle alive before dlclose. */
#define CJIT_GRACE_PERIOD_MS    100

/**
 * Maximum length (including NUL) of the extra_cflags string in cjit_config_t.
 * Used to size both the config field and CLI accumulation buffers so that all
 * places that reference this limit stay in sync.
 */
#define CJIT_MAX_EXTRA_CFLAGS   512

/**
 * Maximum length (including NUL) of the cc_binary string in cjit_config_t.
 */
#define CJIT_MAX_CC_BINARY      64

/**
 * Per-calling-thread TLS call-counter flush threshold.
 *
 * Each calling thread maintains a private per-function byte counter in
 * thread-local storage (TLS).  The counter is incremented on every call
 * with no shared-memory traffic.  When it reaches CJIT_TLS_FLUSH_THRESHOLD,
 * the accumulated count is flushed to the global atomic call_cnt in the
 * function table and the local counter is reset to zero.
 *
 * Benefits
 * ────────
 * • Hot path (common case): one plain byte increment to thread-private memory
 *   — no atomics, no cache-line ownership transfer, zero coherence traffic.
 * • func_ptr (cache line 0) stays in "Shared" state on all cores at all times.
 *   It is only invalidated during an actual function-pointer swap, which is
 *   rare (compilation events).  Previously, every call_cnt write (same line)
 *   forced a cache-line ownership transfer that also evicted func_ptr.
 * • Atomic flush to call_cnt (cache line 1) occurs once every
 *   CJIT_TLS_FLUSH_THRESHOLD calls per thread — a > 96 % reduction in
 *   shared-memory traffic compared with a direct per-call atomic_fetch_add.
 *
 * Accuracy
 * ────────
 * The monitor reads call_cnt with a relaxed load each scan cycle.  With N
 * calling threads, the maximum unobserved lag is
 *   N × (CJIT_TLS_FLUSH_THRESHOLD − 1)  calls.
 * The EMA-based detection algorithm naturally absorbs this small lag: it
 * smooths call rates over multiple monitor scan cycles, so a short delay in
 * observing the last batch never prevents correct tier-promotion decisions.
 *
 * Multi-engine note
 * ─────────────────
 * The TLS array is indexed by func_id_t, which is engine-specific.  Programs
 * that use more than one cjit_engine_t should call cjit_flush_local_counts()
 * for every engine on each thread before switching to a different engine,
 * to avoid cross-engine counter contamination.
 *
 * Must be a power of two in [2, 255].  Default: 32.
 */
#define CJIT_TLS_FLUSH_THRESHOLD  32u

/* ── Hot-function detection defaults ──────────────────────────────────────── */

/** Default minimum sustained calls/sec to trigger O2 compilation. */
#define CJIT_DEFAULT_HOT_RATE_T1        1000ULL

/** Default minimum sustained calls/sec to trigger O3 compilation. */
#define CJIT_DEFAULT_HOT_RATE_T2        5000ULL

/**
 * Default EMA smoothing: number of scan cycles whose equivalent weight is
 * used to compute α = 2 / (CJIT_DEFAULT_HOT_CONFIRM_CYCLES + 1).
 * A value of 3 gives α = 0.5; a single cold scan drops the EMA by 50%.
 */
#define CJIT_DEFAULT_HOT_CONFIRM_CYCLES 3U

/** Default minimum calls since last O2 compilation before trying O3. */
#define CJIT_DEFAULT_MIN_CALLS_T2       2000ULL

/** Default minimum ms between tier promotions of the same function. */
#define CJIT_DEFAULT_COMPILE_COOLOFF_MS 500U

/** Default number of async I/O threads for IR prefetch (inside ir_cache). */
#define CJIT_DEFAULT_IO_THREADS         2U

/**
 * Default hard cap on per-function JIT recompilations.
 *
 * Once a function has been recompiled this many times the monitor stops
 * promoting it entirely, preventing infinite recompilation loops.  The
 * practical maximum useful tier promotions are OPT_O2 and OPT_O3, so the
 * default of 8 leaves significant headroom for manual recompile requests.
 */
#define CJIT_DEFAULT_MAX_RECOMPILES     8U

/**
 * Default rate-threshold scale factor per recompile (percent).
 *
 * The effective EMA rate threshold = base × (1 + recompile_count × pct/100).
 * At the default of 50%:
 *   0 recompiles → 1.0× base    (no extra evidence needed yet)
 *   1 recompile  → 1.5× base    (must be 50% hotter than the original gate)
 *   2 recompiles → 2.0× base
 *   4 recompiles → 3.0× base
 * This ensures each successive recompile is only triggered when the call
 * rate is meaningfully higher than the previous trigger level, preventing
 * oscillation and cache thrashing for tiny potential gains.
 */
#define CJIT_DEFAULT_RECOMPILE_RATE_SCALE_PCT  50U

/**
 * Default minimum engine uptime (ms) before O3 compilations are allowed.
 *
 * Programs exhibit call-pattern spikes in the first few seconds (class
 * loading, JIT-compiler warmup, application init).  Suppressing O3 during
 * this window prevents expensive compilations triggered by transient
 * startup hotness that will never recur in steady state.
 *
 * Default: 5 000 ms (5 seconds).
 */
#define CJIT_DEFAULT_MIN_UPTIME_T2_MS          5000U

/**
 * Default extra required consecutive-above-threshold scan cycles per
 * recompile count.
 *
 * Required stability streak = hot_confirm_cycles + recompile_count × N.
 * At N = 2: after 3 recompiles the monitor needs 6 more consecutive
 * above-threshold scans than the baseline hot_confirm_cycles.  This
 * ensures that each successive promotion requires a proportionally
 * longer window of observed sustained hotness, preventing decisions
 * based on insufficient data.
 */
#define CJIT_DEFAULT_EXTRA_STREAK_PER_RECOMPILE  2U

/**
 * Default CPU-time threshold (nanoseconds per second) to trigger O2
 * compilation based on measured execution time.
 *
 * 0 = disabled: the CPU-time signal is not used for tier promotion unless
 * both cpu_hot_ns_per_sec_t1 and cpu_hot_ns_per_sec_t2 are set non-zero
 * in cjit_config_t.
 *
 * When enabled, this threshold is used alongside (OR with) the call-rate
 * threshold, allowing functions that are called infrequently but execute
 * for a long time each call to be promoted purely on CPU time consumed.
 */
#define CJIT_DEFAULT_CPU_HOT_NS_T1   0ULL  /* disabled */

/** Default CPU-time threshold (ns/sec) to trigger O3.  0 = disabled. */
#define CJIT_DEFAULT_CPU_HOT_NS_T2   0ULL  /* disabled */

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

/* ═══════════════════════════ timing helper ════════════════════════════════ */

/**
 * Returns the current monotonic nanosecond timestamp.
 *
 * Used by CJIT_DISPATCH_TIMED and CJIT_DISPATCH_TIMED_VOID to measure
 * per-call execution time with minimal overhead.  On Linux the call goes
 * through the vDSO (no kernel entry), making the round-trip cost ≈ 25 ns.
 *
 * Two cjit_timestamp_ns() calls bracket a function invocation; the
 * difference gives the elapsed nanoseconds, which are accumulated in a
 * per-thread TLS buffer and flushed to the shared atomic counter every
 * CJIT_TLS_FLUSH_THRESHOLD calls — so the amortised per-call overhead of
 * the flush is negligible.
 */
static inline uint64_t cjit_timestamp_ns(void)
{
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return (uint64_t)_ts.tv_sec * UINT64_C(1000000000)
         + (uint64_t)_ts.tv_nsec;
}

/* ══════════════════════ argument-value sampling helpers ═══════════════════ */

/**
 * @internal  Per-calling-thread call-count batch array.
 *
 * Defined (without static) in cjit.c; exposed here so that the
 * CJIT_SHOULD_SAMPLE and CJIT_SAMPLE_ARGS macros can check the flush
 * boundary without a function call.  Callers MUST NOT modify this array
 * directly — use cjit_record_call() / cjit_get_func_counted() instead.
 */
extern __thread uint8_t cjit_tls_counts[CJIT_MAX_FUNCTIONS];

/**
 * Returns true if the next call to cjit_get_func_counted() (or any
 * CJIT_DISPATCH variant) for function `id` is the flush-boundary call —
 * i.e. it is the one-in-CJIT_TLS_FLUSH_THRESHOLD call that will increment
 * the global atomic call counter.
 *
 * This is the cheapest possible sampling trigger: it reads one byte from
 * the TLS array (already in L1 cache due to the co-located dispatch) and
 * compares it against a compile-time constant.  On the common (non-sample)
 * path the macro is predicted not-taken and contributes < 1 cycle amortised
 * overhead.
 *
 * IMPORTANT: evaluate `id` only once to avoid side effects.
 */
#define CJIT_SHOULD_SAMPLE(id) \
    (__builtin_expect( \
        (id) < CJIT_MAX_FUNCTIONS && \
        cjit_tls_counts[(id)] == (CJIT_TLS_FLUSH_THRESHOLD - 1u), 0))

/**
 * Sample argument values for function `id` on the TLS flush boundary.
 *
 * MUST be placed BEFORE the cjit_get_func_counted() / CJIT_DISPATCH call so
 * that cjit_tls_counts[id] still reflects the pre-flush state.
 *
 * Usage:
 *   CJIT_SAMPLE_ARGS(engine, id, (uint64_t)a, (uint64_t)b);
 *   int r = CJIT_DISPATCH(engine, id, add_fn_t, a, b);
 *
 * The variadic arguments after `id` are the argument values cast to
 * uint64_t.  Exactly as many values as arguments you want to profile should
 * be passed; up to CJIT_MAX_PROFILED_ARGS (8) values are recorded and
 * additional values are silently ignored by cjit_record_arg_samples().
 * Passing more than CJIT_MAX_PROFILED_ARGS arguments does NOT cause a
 * buffer overrun — the count is clamped before any array access.
 *
 * Hot-path overhead (non-sample case):
 *   • CJIT_SHOULD_SAMPLE check: 1 TLS byte load + compare + branch ≈ 1 cycle
 *   • No other overhead
 *
 * Sample-path overhead (1 in CJIT_TLS_FLUSH_THRESHOLD calls per thread):
 *   • Argument values already computed for the real call — no re-evaluation
 *   • One call to cjit_record_arg_samples() ≈ 50–100 ns (cold cache miss
 *     on the arg_profile cold cache line; amortised over THRESHOLD calls ≈
 *     50 ns / 32 = 1.5 ns average additional overhead per call)
 */
#define CJIT_SAMPLE_ARGS(engine, id, ...)                                      \
    do {                                                                        \
        if (CJIT_SHOULD_SAMPLE(id)) {                                          \
            uint64_t _cjit_sa_[] = {__VA_ARGS__};                              \
            cjit_record_arg_samples((engine), (id),                            \
                (uint8_t)(sizeof(_cjit_sa_) / sizeof(_cjit_sa_[0])),          \
                _cjit_sa_);                                                     \
        }                                                                       \
    } while (0)

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
    bool     enable_fast_math;    /**< Pass -ffast-math at OPT_O3 (may change
                                       floating-point semantics).                 */
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

    /* ── Hot-function detection tuning ──────────────────────────────────── */

    /**
     * Minimum sustained call rate (calls/second) to consider a function
     * ready for tier-1 (O2) compilation.
     *
     * The monitor computes rate = delta_calls / interval_ms * 1000 each
     * scan cycle.  Using rate instead of a raw total count means a function
     * that was heavily used last hour but has since cooled is NOT promoted,
     * preventing wasteful O3 compilations of effectively cold code.
     *
     * Default: 1 000 calls/sec.
     */
    uint64_t hot_rate_t1;

    /**
     * Minimum sustained call rate (calls/second) to consider a function
     * ready for tier-2 (O3) compilation.  Must be ≥ hot_rate_t1.
     *
     * Default: 5 000 calls/sec.
     */
    uint64_t hot_rate_t2;

    /**
     * Number of consecutive monitor scan cycles the function must stay above
     * the rate threshold before a promotion is issued.
     *
     * This acts as a confidence filter: brief call spikes (e.g. a single
     * burst loop that finishes quickly) do not trigger expensive O3
     * compilation.  Only functions that are *continuously* hot over at least
     * hot_confirm_cycles × monitor_interval_ms milliseconds are promoted.
     *
     * Default: 3 cycles (e.g. 150 ms at the 50 ms default interval).
     */
    uint32_t hot_confirm_cycles;

    /**
     * Minimum number of calls that must have occurred since the previous
     * promotion before an O2 → O3 upgrade is issued.
     *
     * Ensures that O3 is only attempted when there is enough evidence that
     * the function's hot window will persist long enough to recoup the
     * extra compilation cost.  If a function reaches O2 and is then barely
     * called again, upgrading to O3 would be pure overhead.
     *
     * Default: 2 000 calls.
     */
    uint64_t min_calls_for_tier2;

    /**
     * Minimum milliseconds that must elapse between consecutive promotion
     * attempts for the same function (cooloff period).
     *
     * The monitor also enforces max(compile_cooloff_ms, 2 × last_compile_duration_ms)
     * so that re-enqueuing cannot happen before the previous compilation has
     * likely completed.  last_compile_duration_ms is written by the background
     * compiler thread into the function-table entry — zero hot-path overhead.
     *
     * Default: CJIT_DEFAULT_COMPILE_COOLOFF_MS (500 ms).
     */
    uint32_t compile_cooloff_ms;

    /**
     * Number of background I/O threads dedicated to async IR prefetch.
     *
     * When the monitor observes a function's call rate crossing hot_rate_t1/10
     * for the first time and the function's IR is COLD (on disk only), it
     * submits a non-blocking prefetch request.  An I/O thread loads the IR
     * from disk into the WARM generation so that by the time hot_confirm_cycles
     * scan cycles later the compile task fires, the IR is already in memory —
     * hiding disk-read latency from the compiler thread entirely.
     *
     * Setting this to 0 disables async prefetch (reads happen synchronously
     * in the compiler thread, as before).
     *
     * Default: CJIT_DEFAULT_IO_THREADS (2).
     */
    uint32_t io_threads;

    /* ── Data-driven recompile throttling ───────────────────────────────── */

    /**
     * Hard cap on the number of successful JIT recompilations per function.
     *
     * Once a function's recompile_count reaches this limit the monitor
     * stops promoting it entirely, preventing endless recompilation loops
     * and DL-cache thrashing for progressively smaller gains.  Manual
     * cjit_request_recompile() calls are not affected by this limit.
     *
     * Default: CJIT_DEFAULT_MAX_RECOMPILES (8).
     */
    uint32_t max_recompiles_per_func;

    /**
     * Percentage by which the effective EMA rate threshold increases for
     * each additional recompile already performed on the function.
     *
     * Effective threshold = base_thresh × (1 + recompile_count × pct / 100).
     *
     * Higher values make the monitor demand proportionally stronger evidence
     * of sustained hotness before attempting another recompile, preventing
     * the engine from repeatedly recompiling a function for tiny gains.
     * Setting this to 0 disables the per-recompile threshold scaling.
     *
     * Default: CJIT_DEFAULT_RECOMPILE_RATE_SCALE_PCT (50).
     */
    uint32_t recompile_rate_scale_pct;

    /**
     * Minimum engine uptime (ms) before tier-2 (O3) compilations are issued.
     *
     * Programs exhibit artificial call-rate spikes during initialisation
     * (module loading, internal warmup, memory layout settling).  Suppressing
     * O3 promotions until the engine has been running for at least this long
     * prevents expensive O3 compilations from being triggered by transient
     * startup hotness that will never recur once the program reaches steady
     * state.
     *
     * Default: CJIT_DEFAULT_MIN_UPTIME_T2_MS (5 000 ms).
     */
    uint32_t min_uptime_for_tier2_ms;

    /**
     * Extra consecutive above-threshold scan cycles required per additional
     * recompile already performed.
     *
     * Required stability streak = hot_confirm_cycles
     *                           + recompile_count × extra_streak_per_recompile.
     *
     * A function that has been recompiled multiple times must demonstrate
     * sustained hotness over a proportionally longer observation window before
     * the next recompile is issued.  Setting this to 0 keeps the streak
     * requirement constant (equal to hot_confirm_cycles) regardless of how
     * many times the function has been recompiled.
     *
     * Default: CJIT_DEFAULT_EXTRA_STREAK_PER_RECOMPILE (2).
     */
    uint32_t extra_streak_per_recompile;

    /* ── CPU-time-based tier promotion ─────────────────────────────────── */

    /**
     * Minimum sustained CPU time (nanoseconds per second) to consider a
     * function ready for tier-1 (O2) compilation.
     *
     * When non-zero this threshold is checked in parallel with (OR alongside)
     * the call-rate gate: a function is promoted to O2 if either:
     *   • ema_rate       ≥ hot_rate_t1, OR
     *   • ema_ns_per_sec ≥ cpu_hot_ns_per_sec_t1  (and this field > 0)
     *
     * This catches functions that are invoked infrequently but spend a
     * significant wall-clock time each call (e.g. large parsers, image
     * decoders) — functions whose call rate never crosses the call-rate
     * threshold but whose CPU footprint clearly warrants optimisation.
     *
     * The EMA smoothing coefficient (alpha) is the same as for the call-rate
     * EMA, so the two signals update at the same pace.  The per-recompile
     * scaled threshold and streak gates also apply to the CPU-time path,
     * preventing premature promotions based on transient spikes.
     *
     * Timing data is only accumulated when CJIT_DISPATCH_TIMED or
     * CJIT_DISPATCH_TIMED_VOID is used for dispatch.  Functions dispatched
     * through CJIT_DISPATCH or cjit_get_func_counted() contribute zero
     * elapsed time; the CPU-time gate will never fire for them regardless
     * of this threshold.
     *
     * 0 = disabled (default).  Units: nanoseconds/second.
     */
    uint64_t cpu_hot_ns_per_sec_t1;

    /**
     * Minimum sustained CPU time (nanoseconds per second) to consider a
     * function ready for tier-2 (O3) compilation.  Must be ≥
     * cpu_hot_ns_per_sec_t1 when both are non-zero.
     *
     * 0 = disabled (default).  Units: nanoseconds/second.
     */
    uint64_t cpu_hot_ns_per_sec_t2;

    /**
     * Extra compiler flags (space-separated) passed verbatim to cc(1).
     *
     * Applied after all CJIT-generated flags so they can override defaults.
     * Useful for -I include paths, -D preprocessor defines, -l libraries, etc.
     * Example: "-I/usr/local/include -DNDEBUG -lm"
     *
     * Maximum length: CJIT_MAX_EXTRA_CFLAGS - 1 characters.
     */
    char extra_cflags[CJIT_MAX_EXTRA_CFLAGS];

    /**
     * Compiler binary override.
     *
     * Empty string (default) → "cc" found on PATH.
     * Set to e.g. "gcc", "clang", or an absolute path like "/usr/bin/gcc-14".
     *
     * Maximum length: CJIT_MAX_CC_BINARY - 1 characters.
     */
    char cc_binary[CJIT_MAX_CC_BINARY];
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

    /**
     * Highest recompile_count seen across all registered functions.
     *
     * Useful for diagnosing whether the hard cap is being hit or whether
     * the scaled thresholds are having the desired effect.
     */
    uint32_t max_recompile_count;   /**< Highest per-function recompile count.  */

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

    /* ── Execution timing ────────────────────────────────────────────── */

    /**
     * Sum of per-function total_elapsed_ns across all registered functions.
     *
     * Only non-zero when CJIT_DISPATCH_TIMED / CJIT_DISPATCH_TIMED_VOID
     * are used for at least some dispatches.  Useful for computing the
     * overall fraction of wall-clock time spent inside JIT-compiled code.
     */
    uint64_t total_elapsed_ns;
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
 * This is a single atomic_load_explicit(..., memory_order_acquire).
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
 * Uses a per-calling-thread TLS batch counter (see CJIT_TLS_FLUSH_THRESHOLD).
 * The hot-path cost is a single byte increment to thread-private memory — no
 * atomics, no shared-memory traffic.  A flush to the global atomic call_cnt
 * occurs automatically every CJIT_TLS_FLUSH_THRESHOLD calls per thread.
 *
 * Call this immediately after cjit_get_func() in the dispatch loop if you
 * want the monitor thread to track call frequency.
 *
 * @param engine  The engine.
 * @param id      Function ID.
 */
void cjit_record_call(cjit_engine_t *engine, func_id_t id);

/**
 * HOT PATH: retrieve the current function pointer AND increment the call
 * counter in a single function-table lookup.
 *
 * This is strictly faster than calling cjit_get_func() + cjit_record_call()
 * separately because it performs only ONE table lookup (one atomic bounds-check
 * load + one stable-pointer dereference) instead of two.
 *
 * Call-count accounting uses the TLS batch counter (CJIT_TLS_FLUSH_THRESHOLD),
 * so the counter increment is always overhead-free on the hot path.
 *
 * Atomicity guarantees:
 *   • call_cnt flush : relaxed atomic_fetch_add (once per THRESHOLD calls;
 *                      approximation is sufficient; no ordering needed).
 *   • func_ptr load  : acquire (ensures the function body is fully visible
 *                      before the indirect call executes).
 *
 * @param engine  The engine.
 * @param id      Function ID returned by cjit_register_function().
 * @return        Current function pointer; never NULL after registration.
 */
jit_func_t cjit_get_func_counted(cjit_engine_t *engine, func_id_t id);

/**
 * Convenience macro: single-lookup atomic dispatch.
 *
 * Uses cjit_get_func_counted() so only ONE table lookup is performed per
 * dispatch, compared to the naive separate get_func + record_call pattern.
 *
 * Usage:
 *   typedef int (*add_fn_t)(int, int);
 *   int result = CJIT_DISPATCH(engine, id, add_fn_t, a, b);
 *
 * The macro uses a GCC/Clang statement expression (__extension__({...})) to
 * ensure that `engine` and `id` are evaluated exactly once even if the
 * arguments have side effects (e.g. CJIT_DISPATCH(get_eng(), next_id++, ...)).
 * Note: `cast_type` is used only as a type token (no side effects possible)
 * and the variadic function arguments are passed through directly and evaluated
 * once in the underlying function call — this is the expected C behaviour.
 * This extension is available in all GCC and Clang versions that support the
 * rest of this codebase.
 */
#define CJIT_DISPATCH(engine, id, cast_type, ...)                             \
    __extension__({                                                            \
        cjit_engine_t *_cjit_e = (engine);                                    \
        func_id_t      _cjit_i = (id);                                        \
        ((cast_type)cjit_get_func_counted(_cjit_e, _cjit_i))(__VA_ARGS__);   \
    })

/**
 * HOT PATH: timed single-lookup dispatch for non-void functions.
 *
 * Identical to CJIT_DISPATCH but additionally measures the wall-clock time
 * of the function call and accumulates it in the per-thread TLS elapsed
 * buffer.  The buffer is flushed to the shared atomic
 * func_table_entry_t::total_elapsed_ns every CJIT_TLS_FLUSH_THRESHOLD calls
 * per thread — the same batch size used for call_cnt — so only one extra
 * atomic operation (per THRESHOLD calls) is needed beyond the normal
 * CJIT_DISPATCH overhead.
 *
 * The monitor reads total_elapsed_ns each scan cycle, computes an EMA of
 * nanoseconds-per-second, and uses it alongside the call-rate EMA for tier-
 * promotion decisions.  Functions that are called infrequently but execute
 * for a long time each call (e.g. parsers, image decoders) will be promoted
 * to O2/O3 based on their CPU time, not just their call frequency.
 *
 * Timer overhead: two clock_gettime(CLOCK_MONOTONIC) calls per dispatch
 * ≈ 50 ns total (vDSO on Linux).  For functions that take < 200 ns to
 * execute the overhead fraction can be significant; for functions ≥ 1 µs
 * it is < 5 %.  Use CJIT_DISPATCH for sub-microsecond functions where
 * call-rate-based promotion is sufficient.
 *
 * Usage (non-void return):
 *   typedef int (*add_fn_t)(int, int);
 *   int result = CJIT_DISPATCH_TIMED(engine, id, add_fn_t, a, b);
 *
 * @note  Uses GCC/Clang __typeof__ to capture the return value without
 *        requiring a user-supplied result variable.  Not suitable for
 *        void-returning functions; use CJIT_DISPATCH_TIMED_VOID instead.
 */
#define CJIT_DISPATCH_TIMED(engine, id, cast_type, ...)                       \
    __extension__({                                                            \
        cjit_engine_t *_cjit_e  = (engine);                                   \
        func_id_t      _cjit_i  = (id);                                       \
        jit_func_t     _cjit_fn = cjit_get_func(_cjit_e, _cjit_i);           \
        uint64_t       _cjit_t0 = cjit_timestamp_ns();                        \
        /* __typeof__ is unevaluated; (cast_type)0 is never dereferenced. */  \
        __typeof__(((cast_type)0)(__VA_ARGS__)) _cjit_r =                     \
            ((cast_type)_cjit_fn)(__VA_ARGS__);                               \
        cjit_record_timed_call(_cjit_e, _cjit_i,                              \
                               cjit_timestamp_ns() - _cjit_t0);               \
        _cjit_r;                                                               \
    })

/**
 * HOT PATH: timed single-lookup dispatch for void-returning functions.
 *
 * Identical to CJIT_DISPATCH_TIMED but for functions with no return value.
 *
 * Usage:
 *   typedef void (*render_fn_t)(scene_t *);
 *   CJIT_DISPATCH_TIMED_VOID(engine, id, render_fn_t, &scene);
 */
#define CJIT_DISPATCH_TIMED_VOID(engine, id, cast_type, ...)                  \
    __extension__({                                                            \
        cjit_engine_t *_cjit_e  = (engine);                                   \
        func_id_t      _cjit_i  = (id);                                       \
        jit_func_t     _cjit_fn = cjit_get_func(_cjit_e, _cjit_i);           \
        uint64_t       _cjit_t0 = cjit_timestamp_ns();                        \
        ((cast_type)_cjit_fn)(__VA_ARGS__);                                   \
        cjit_record_timed_call(_cjit_e, _cjit_i,                              \
                               cjit_timestamp_ns() - _cjit_t0);               \
    })

/**
 * Record a call to function `id` along with `elapsed_ns` nanoseconds spent
 * inside it.
 *
 * Increments the thread-local batch counter (same mechanism as
 * cjit_record_call) and accumulates elapsed_ns in a parallel TLS buffer.
 * Both are flushed to shared atomics every CJIT_TLS_FLUSH_THRESHOLD calls.
 *
 * Callers that prefer manual timing (e.g. when they already have their own
 * high-resolution timer) can call this directly instead of using the
 * CJIT_DISPATCH_TIMED macro.
 *
 * @param engine      The engine.
 * @param id          Function ID.
 * @param elapsed_ns  Nanoseconds spent inside the function on this call.
 */
void cjit_record_timed_call(cjit_engine_t *engine,
                              func_id_t      id,
                              uint64_t       elapsed_ns);

/**
 * Return the cumulative nanoseconds observed for function `id`.
 *
 * Only counts nanoseconds recorded via CJIT_DISPATCH_TIMED,
 * CJIT_DISPATCH_TIMED_VOID, or cjit_record_timed_call().  Returns 0 if
 * the function has only been dispatched through the non-timed paths.
 *
 * Thread safety: relaxed atomic read.  The value may lag by at most
 *   N × (CJIT_TLS_FLUSH_THRESHOLD − 1) × avg_elapsed_ns
 * where N is the number of active calling threads.
 *
 * @param engine  The engine.
 * @param id      Function ID.
 * @return        Total nanoseconds accumulated, or 0 on error.
 */
uint64_t cjit_get_elapsed_ns(const cjit_engine_t *engine, func_id_t id);

/**
 * Record argument values for function `id` at a sampling point.
 *
 * Intended to be called via the CJIT_SAMPLE_ARGS macro, which automatically
 * gates the call to the TLS flush boundary (once per CJIT_TLS_FLUSH_THRESHOLD
 * calls per thread).  Direct calls are also valid — e.g. after obtaining a
 * manual timestamp — but the caller is then responsible for controlling
 * sample frequency to avoid excessive overhead.
 *
 * Updates the per-function argument profile stored in func_table_entry_t
 * using a Boyer-Moore majority-vote algorithm (O(1) per sample, no heap
 * allocation).  The profile is later read by codegen_compile() when a
 * recompilation is triggered; if a confident dominant value is found on any
 * integer-typed argument, a specialised wrapper is generated that lets the
 * compiler constant-fold through the hot path.
 *
 * @param engine  The engine.
 * @param id      Function ID.
 * @param n_args  Number of values in vals[] (≤ CJIT_MAX_PROFILED_ARGS).
 * @param vals    Array of argument values cast to uint64_t.
 */
void cjit_record_arg_samples(cjit_engine_t  *engine,
                               func_id_t       id,
                               uint8_t         n_args,
                               const uint64_t *vals);

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
 * Flush this thread's pending TLS call-count and elapsed-time batches to the
 * engine.
 *
 * Under normal operation calling threads never need to call this; both TLS
 * batches flush automatically every CJIT_TLS_FLUSH_THRESHOLD calls.  However
 * there are two situations where an explicit flush is needed:
 *
 *   1. Before the calling thread exits.  TLS storage is reclaimed by the
 *      C runtime when a thread terminates; any unflushed partial batch is
 *      silently lost.  Calling this function in a thread-exit hook (e.g.
 *      pthread_cleanup_push) preserves those call-count and elapsed-time
 *      observations.
 *
 *   2. Before calling cjit_stop() / cjit_destroy() if you want the final
 *      call-count and elapsed-time snapshots to be accurate (e.g. for
 *      logging / diagnostics).
 *
 * The function iterates over registered functions and flushes the partial
 * remainder for both cjit_tls_counts and cjit_tls_elapsed.  It is
 * O(registered_functions) and touches no shared memory for functions whose
 * local counters are currently zero.
 *
 * Thread safety: safe to call from any thread at any time.  Only the calling
 * thread's TLS state is modified; no other threads are affected.
 *
 * @param engine  The engine whose function table should be updated.
 */
void cjit_flush_local_counts(cjit_engine_t *engine);

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

/**
 * Return the number of successful JIT recompilations performed for the
 * given function.
 *
 * Incremented atomically each time a compiled .so is loaded and the
 * function pointer is swapped.  Useful for diagnosing whether the
 * data-driven recompile throttle (max_recompiles_per_func) is being hit.
 *
 * @param engine  The engine.
 * @param id      Function ID returned by cjit_register_function().
 * @return        Recompile count, or 0 if id is invalid.
 */
uint32_t cjit_get_recompile_count(const cjit_engine_t *engine, func_id_t id);

/**
 * Find a registered function by name.
 *
 * Performs a linear scan of the function table.  Not intended for hot-path
 * use; suitable for setup-time lookups and diagnostics.
 *
 * @param engine  The engine.
 * @param name    Exact function name as passed to cjit_register_function().
 * @return        func_id_t of the matching function, or CJIT_INVALID_FUNC_ID
 *                if not found or if name is NULL.
 */
func_id_t cjit_lookup_function(const cjit_engine_t *engine, const char *name);

/**
 * Replace the IR source for a registered function and trigger recompilation.
 *
 * The new IR is stored in the engine's IR cache and written to the on-disk
 * backup so that a COLD-evicted entry is reloaded with the updated source.
 * Any in-flight compile tasks queued against the old IR are invalidated via
 * the per-entry version counter and silently discarded by compiler threads.
 *
 * Thread safety: safe to call while the engine is running.  Must NOT be
 *                called concurrently with cjit_register_function() for the
 *                same id.
 *
 * @param engine   The engine.
 * @param id       ID of the function to update (from cjit_register_function).
 * @param new_ir   New C source string to use as the function's IR.
 * @param level    Optimisation level to request for the triggered recompile.
 * @return         true on success, false if id is invalid or new_ir is NULL.
 */
bool cjit_update_ir(cjit_engine_t *engine,
                    func_id_t      id,
                    const char    *new_ir,
                    opt_level_t    level);

/**
 * Block until the function at id has been JIT-compiled at least once
 * (i.e. cjit_get_func() returns non-NULL), or until the timeout expires.
 *
 * Uses a condition variable — no busy-waiting.  Suitable for startup
 * warm-up paths where you need the first compilation to complete before
 * serving production traffic.
 *
 * When an AOT fallback was provided at registration time, cjit_get_func()
 * returns the fallback pointer immediately and this function returns true
 * immediately.  Use cjit_get_current_opt_level() to distinguish fallback
 * (OPT_NONE) from JIT-compiled code (OPT_O1 / OPT_O2 / OPT_O3).
 *
 * @param engine      The engine.
 * @param id          Function ID returned by cjit_register_function().
 * @param timeout_ms  Maximum time to wait in milliseconds.
 *                    0 = perform a single non-blocking check.
 * @return            true if cjit_get_func(engine, id) is non-NULL within
 *                    the timeout, false on timeout or invalid id.
 */
bool cjit_wait_compiled(cjit_engine_t *engine,
                        func_id_t      id,
                        uint32_t       timeout_ms);

#ifdef __cplusplus
}
#endif
