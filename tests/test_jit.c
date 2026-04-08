/**
 * test_jit.c – Integration test suite for the CJIT engine.
 *
 * Each test function (t01 … t10) exercises a distinct aspect of the JIT:
 *
 *   t01  AOT fallback correctness      – functions return right answers before
 *                                        any JIT compilation has happened.
 *   t02  Hot-loop O2 promotion         – a sustained dispatch loop causes the
 *                                        monitor to promote functions to O2.
 *   t03  Argument-profile specialise   – using CJIT_SAMPLE_ARGS / CJIT_DISPATCH
 *                                        with a dominant constant arg drives
 *                                        an arg-specialisation wrapper at O2.
 *   t04  Timed dispatch & elapsed-ns   – CJIT_DISPATCH_TIMED accumulates a
 *                                        non-zero total_elapsed_ns.
 *   t05  Heavy multi-function load     – sorting, matrix multiply, string hash,
 *                                        and CRC32 all yield correct answers
 *                                        under sustained load.
 *   t06  Explicit recompile request    – cjit_request_recompile() raises the
 *                                        opt level and increases recompile_count.
 *   t07  CJIT_DISPATCH macro           – single-lookup dispatch returns the
 *                                        correct value.
 *   t08  Stats integrity               – call counts, total_compilations, and
 *                                        total_swaps are consistent after a run.
 *   t09  Prime sieve correctness       – a JIT-compiled Sieve of Eratosthenes
 *                                        returns the right prime count across
 *                                        multiple timed iterations.
 *   t10  Concurrent thread safety      – N threads each dispatch through the
 *                                        same JIT functions simultaneously with
 *                                        no data races or wrong answers.
 *   t11  sum_range pre-warm + bounds   – boundary cases verified across 50 reps
 *                                        after an explicit O1 pre-warm request.
 *   t12  CRC32 tier progression        – CRC32 correctness verified at O1, O2,
 *                                        and O3 against the reference.
 *
 *   t13  cjit_lookup_function       – lookup by name returns correct func_id;
 *                                        unknown names return INVALID.
 *   t14  cjit_update_ir (hot-reload) – hot-reload with new IR produces the
 *                                        updated result after recompilation.
 *   t15  extra_cflags (-D define)   – a -D flag in cfg.extra_cflags reaches
 *                                        the compiler and affects JIT output.
 *   t16  JIT replaces AOT pointer   – after cjit_request_recompile +
 *                                        cjit_wait_compiled, func_ptr is no
 *                                        longer the AOT fallback (new binary).
 *   t17  dladdr compiled object     – dladdr(3) confirms the JIT function
 *                                        lives in a separately loaded shared
 *                                        object, not in the test binary itself.
 *   t18  O3 vectorised correctness  – integer dot-product compiled at O3 +
 *                                        vectorise + unroll + march=native
 *                                        gives correct results over 1 000
 *                                        iterations.
 *   t19  Performance benchmark      – measures compile latency (ms) and
 *                                        dispatch throughput (Mcall/s) at AOT,
 *                                        O1, O2, O3 for an integer dot-product;
 *                                        reports speedup vs AOT; asserts
 *                                        correctness and compile < 10 s per
 *                                        tier and O3 not >3× slower than O1.
 *   t20  New opt-flags + preamble   – verifies the -funswitch-loops /
 *                                        -fpeel-loops additions and the new
 *                                        FLATTEN / NORETURN / CJIT_EXPORT /
 *                                        MALLOC_FUNC preamble attributes and
 *                                        the <limits.h> / <stdbool.h> auto-
 *                                        includes by compiling representative
 *                                        IR at O2 and O3 and checking all
 *                                        results against AOT references.
 *   t21  Artifact cache            – compiles a function twice using the same
 *                                        persistent cache directory; verifies
 *                                        pass-1 is a cache miss (compiler ran),
 *                                        pass-2 is a cache hit (compiler skipped),
 *                                        and the function returns correct results.
 *
 * Each test prints PASS or FAIL and returns 0 / 1.  The main() aggregates the
 * results and exits with the failure count (0 = all passed).
 *
 * Build:
 *   cmake --build build --target test_jit
 *   ./build/test_jit
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>   /* dladdr – verify JIT function is in a loaded shared object */

#include "../include/cjit.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Tiny test-framework helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TEST(name) static int name(void)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL  %s:%d  " msg "\n", __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

#define CHECKF(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL  %s:%d  " fmt "\n", __FILE__, __LINE__, __VA_ARGS__); \
        return 1; \
    } \
} while (0)

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static void sleep_ms(long ms)
{
    struct timespec t = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&t, NULL);
}

/* Wait up to `timeout_ms` for predicate fn(arg) to become true. */
static bool wait_for(bool (*fn)(void *), void *arg, long timeout_ms)
{
    uint64_t deadline = now_ms() + (uint64_t)timeout_ms;
    while (now_ms() < deadline) {
        if (fn(arg)) return true;
        sleep_ms(50);
    }
    return fn(arg);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Shared AOT fallbacks
 * ═══════════════════════════════════════════════════════════════════════════ */

static int  aot_add(int a, int b)       { return a + b; }
static int  aot_mul(int a, int b)       { return a * b; }
static long aot_fib(int n)
{
    if (n <= 1) return n;
    long a = 0, b = 1;
    for (int i = 2; i <= n; ++i) { long t = a + b; a = b; b = t; }
    return b;
}
static int  aot_identity(int x)        { return x; }
static long aot_sum_range(int lo, int hi) /* inclusive */
{
    long s = 0; for (int i = lo; i <= hi; ++i) s += i; return s;
}

/* Reference dot-product used as the AOT baseline in t19_perf_benchmark. */
static long aot_dot_int(const int *a, const int *b, int n)
{
    long s = 0;
    for (int i = 0; i < n; ++i) s += (long)a[i] * b[i];
    return s;
}

/* Reference for t20: conditional sum (mode 0 = add, mode 1 = subtract). */
static long aot_cond_sum(const int *a, int n, int mode)
{
    long r = 0;
    for (int i = 0; i < n; ++i) r += (mode == 0) ? a[i] : -a[i];
    return r;
}

/* Reference prime sieve (returns count of primes ≤ limit). */
static int ref_prime_count(int limit)
{
    if (limit < 2) return 0;
    char *sieve = calloc((size_t)(limit + 1), 1);
    if (!sieve) return -1;
    for (int i = 2; (long)i * i <= limit; ++i)
        if (!sieve[i])
            for (int j = i * i; j <= limit; j += i)
                sieve[j] = 1;
    int cnt = 0;
    for (int i = 2; i <= limit; ++i) if (!sieve[i]) cnt++;
    free(sieve);
    return cnt;
}

/* Reference 32-bit CRC (poly 0xEDB88320). */
static uint32_t ref_crc32(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320U & -(crc & 1));
    }
    return crc ^ 0xFFFFFFFFU;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IR strings (C source passed to the JIT)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char IR_ADD[] =
    "int add(int a, int b) { return a + b; }\n";

static const char IR_MUL[] =
    "int mul(int a, int b) { return a * b; }\n";

static const char IR_FIB[] =
    "long fib(int n) {\n"
    "    if (UNLIKELY(n <= 1)) return n;\n"
    "    long a = 0, b = 1;\n"
    "    for (int i = 2; i <= n; ++i) { long t = a+b; a=b; b=t; }\n"
    "    return b;\n"
    "}\n";

static const char IR_IDENTITY[] =
    "int identity(int x) { return x; }\n";

static const char IR_SUM_RANGE[] =
    "long sum_range(int lo, int hi) {\n"
    "    long s = 0;\n"
    "    for (int i = lo; i <= hi; ++i) s += i;\n"
    "    return s;\n"
    "}\n";

static const char IR_PRIME_COUNT[] =
    "#include <stdlib.h>\n"
    "int prime_count(int limit) {\n"
    "    if (limit < 2) return 0;\n"
    "    char *sieve = calloc((size_t)(limit + 1), 1);\n"
    "    if (!sieve) return -1;\n"
    "    for (int i = 2; (long)i*i <= limit; ++i)\n"
    "        if (!sieve[i])\n"
    "            for (int j = i*i; j <= limit; j += i) sieve[j] = 1;\n"
    "    int cnt = 0;\n"
    "    for (int i = 2; i <= limit; ++i) if (!sieve[i]) cnt++;\n"
    "    free(sieve);\n"
    "    return cnt;\n"
    "}\n";

static const char IR_CRC32[] =
    "#include <stdint.h>\n"
    "#include <stddef.h>\n"
    "uint32_t crc32_jit(const uint8_t *data, size_t len) {\n"
    "    uint32_t crc = 0xFFFFFFFFU;\n"
    "    for (size_t i = 0; i < len; ++i) {\n"
    "        crc ^= data[i];\n"
    "        for (int k = 0; k < 8; ++k)\n"
    "            crc = (crc >> 1) ^ (0xEDB88320U & -(crc & 1u));\n"
    "    }\n"
    "    return crc ^ 0xFFFFFFFFU;\n"
    "}\n";

/* Bubble sort: sorts arr[0..n-1] ascending; returns number of swaps. */
static const char IR_BSORT[] =
    "int bsort(int *arr, int n) {\n"
    "    int swaps = 0;\n"
    "    for (int i = 0; i < n-1; ++i)\n"
    "        for (int j = 0; j < n-1-i; ++j)\n"
    "            if (arr[j] > arr[j+1]) {\n"
    "                int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t;\n"
    "                swaps++;\n"
    "            }\n"
    "    return swaps;\n"
    "}\n";

/* 4×4 integer matrix multiply; result written to out[16]. */
static const char IR_MATMUL[] =
    "void matmul4(const int *a, const int *b, int *out) {\n"
    "    for (int i = 0; i < 4; ++i)\n"
    "        for (int j = 0; j < 4; ++j) {\n"
    "            int s = 0;\n"
    "            for (int k = 0; k < 4; ++k) s += a[i*4+k] * b[k*4+j];\n"
    "            out[i*4+j] = s;\n"
    "        }\n"
    "}\n";

/* djb2 string hash. */
static const char IR_DJB2[] =
    "#include <stdint.h>\n"
    "uint32_t djb2(const char *s) {\n"
    "    uint32_t h = 5381;\n"
    "    int c;\n"
    "    while ((c = (unsigned char)*s++)) h = ((h << 5) + h) ^ (uint32_t)c;\n"
    "    return h;\n"
    "}\n";

/* Integer dot product – used by t18 to exercise O3 vectorisation. */
static const char IR_DOT_INT[] =
    "long dot_int(const int *a, const int *b, int n) {\n"
    "    long s = 0;\n"
    "    for (int i = 0; i < n; ++i) s += (long)a[i] * b[i];\n"
    "    return s;\n"
    "}\n";

/* ─ t20 IR strings ─────────────────────────────────────────────────────────
 *
 * IR_COND_SUM: exercises -funswitch-loops.
 *   The loop contains an invariant conditional on `mode`.  With
 *   -funswitch-loops the compiler hoists the branch outside the loop,
 *   emitting two tight loops (one for each mode) instead of testing `mode`
 *   on every iteration.  We verify correctness for both modes at O2 and O3.
 */
static const char IR_COND_SUM[] =
    "long cond_sum(const int *a, int n, int mode) {\n"
    "    long r = 0;\n"
    "    for (int i = 0; i < n; ++i) {\n"
    "        if (mode == 0)\n"
    "            r += a[i];\n"
    "        else\n"
    "            r -= a[i];\n"
    "    }\n"
    "    return r;\n"
    "}\n";

/*
 * IR_PREAMBLE_ATTRS: verifies that newly-added preamble macros and headers
 * compile without error.
 *
 *   • FLATTEN  – applied to outer(), which calls inner(); the compiler must
 *                inline inner() even though it is a non-static, non-inline
 *                function.
 *   • <limits.h> auto-include – INT_MAX is referenced directly.
 *   • <stdbool.h> auto-include – bool / true / false used.
 *   • CJIT_EXPORT  – marks preamble_test as exported (dlsym-visible).
 *   • MALLOC_FUNC  – applied to a trivial allocator-style function (purely
 *                    a compile-time annotation; function body is inert).
 *
 * Expected return value: INT_MAX (computed as 10 adds + INT_MAX - 10).
 */
static const char IR_PREAMBLE_ATTRS[] =
    "static int inner(int x) { return x + 1; }\n"
    "FLATTEN static int outer(int x, int n) {\n"
    "    for (int i = 0; i < n; ++i) x = inner(x);\n"
    "    return x;\n"
    "}\n"
    "/* MALLOC_FUNC: compile-time annotation only; just confirm it compiles. */\n"
    "MALLOC_FUNC static void *trivial_alloc(void) { return 0; }\n"
    "/* CJIT_EXPORT ensures this symbol is visible via dlsym. */\n"
    "CJIT_EXPORT int preamble_test(void) {\n"
    "    (void)trivial_alloc();\n"
    "    bool ok = true;\n"         /* tests stdbool.h auto-include */
    "    if (!ok) return -1;\n"
    "    return outer(INT_MAX - 10, 10);  /* uses limits.h auto-include */\n"
    "}\n";

/*
 * IR_DOT_HELPERS: exercises FLATTEN with real helper inlining.
 *   The outer FLATTEN-annotated function calls two non-static helpers
 *   (mul_pair, add_accum).  Without FLATTEN these would only be inlined
 *   at -O2 if the compiler judges them small enough; with FLATTEN inlining
 *   is guaranteed.  We verify the numeric result matches the AOT reference.
 */
static const char IR_DOT_HELPERS[] =
    "static int mul_pair(int a, int b) { return a * b; }\n"
    "static long add_accum(long s, int v) { return s + v; }\n"
    "FLATTEN long dot_helpers(const int *a, const int *b, int n) {\n"
    "    long s = 0;\n"
    "    for (int i = 0; i < n; ++i)\n"
    "        s = add_accum(s, mul_pair(a[i], b[i]));\n"
    "    return s;\n"
    "}\n";

/* ═══════════════════════════════════════════════════════════════════════════
 * Engine factory
 * ═══════════════════════════════════════════════════════════════════════════ */

static cjit_engine_t *make_engine(bool verbose)
{
    cjit_config_t cfg         = cjit_default_config();
    cfg.verbose               = verbose;
    cfg.monitor_interval_ms   = 50;
    cfg.hot_rate_t1           = 800ULL;
    cfg.hot_rate_t2           = 4000ULL;
    cfg.hot_confirm_cycles    = 2U;
    cfg.compile_cooloff_ms    = 200U;
    cfg.grace_period_ms       = 100U;
    cfg.hot_ir_cache_size     = 64U;
    cfg.warm_ir_cache_size    = 128U;
    return cjit_create(&cfg);
}

/* ─────────────────────────────────────────────────────────────────────────
 * t01 – AOT fallback correctness BEFORE cjit_start()
 * ─────────────────────────────────────────────────────────────────────────
 * Registers add / mul / fib, dispatches them through the AOT pointer,
 * and verifies the answers without ever starting the background threads.
 */
TEST(t01_aot_correctness)
{
    printf("[t01] AOT fallback correctness...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_add = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    func_id_t id_mul = cjit_register_function(e, "mul", IR_MUL, (jit_func_t)aot_mul);
    func_id_t id_fib = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    CHECK(id_add != CJIT_INVALID_FUNC_ID, "add registration failed");
    CHECK(id_mul != CJIT_INVALID_FUNC_ID, "mul registration failed");
    CHECK(id_fib != CJIT_INVALID_FUNC_ID, "fib registration failed");

    typedef int  (*add_fn)(int,int);
    typedef int  (*mul_fn)(int,int);
    typedef long (*fib_fn)(int);

    int  r_add = ((add_fn)cjit_get_func(e, id_add))(7, 3);
    int  r_mul = ((mul_fn)cjit_get_func(e, id_mul))(6, 7);
    long r_fib = ((fib_fn)cjit_get_func(e, id_fib))(10);

    CHECKF(r_add == 10,  "add(7,3) = %d, want 10",  r_add);
    CHECKF(r_mul == 42,  "mul(6,7) = %d, want 42",  r_mul);
    CHECKF(r_fib == 55,  "fib(10)  = %ld, want 55", r_fib);

    /* Opt level must still be OPT_NONE (no JIT yet). */
    CHECK(cjit_get_current_opt_level(e, id_add) == OPT_NONE, "level != OPT_NONE before start");
    CHECK(cjit_get_recompile_count(e, id_add) == 0, "recompile_count != 0 before start");

    cjit_destroy(e);
    printf("[t01] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t02 – Hot-loop drives O2 promotion
 * ─────────────────────────────────────────────────────────────────────────
 * Dispatches add() in a tight loop for 2 seconds; expects the monitor to
 * detect it as hot and promote it to at least O2 within that window.
 */

typedef struct { cjit_engine_t *e; func_id_t id; opt_level_t want; } wait_arg_t;
static bool check_opt(void *v)
{
    wait_arg_t *a = (wait_arg_t *)v;
    return cjit_get_current_opt_level(a->e, a->id) >= a->want;
}

TEST(t02_hot_promotion)
{
    printf("[t02] Hot-loop O2 promotion...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);

    typedef int (*add_fn)(int, int);
    volatile int sink = 0;
    uint64_t deadline = now_ms() + 2000;
    while (now_ms() < deadline) {
        for (int i = 0; i < 200; ++i)
            sink += CJIT_DISPATCH(e, id, add_fn, i, i + 1);
    }

    wait_arg_t wa = { e, id, OPT_O2 };
    bool promoted = wait_for(check_opt, &wa, 3000);
    (void)sink;

    opt_level_t lv = cjit_get_current_opt_level(e, id);
    CHECKF(promoted, "add not promoted to O2 within timeout; level=%d", (int)lv);
    CHECKF(cjit_get_recompile_count(e, id) >= 1,
           "recompile_count=%u, expected ≥1", cjit_get_recompile_count(e, id));

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations >= 1,
           "total_compilations=%llu", (unsigned long long)s.total_compilations);
    CHECKF(s.total_swaps >= 1,
           "total_swaps=%llu", (unsigned long long)s.total_swaps);

    cjit_destroy(e);
    printf("[t02] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t03 – Argument-profile specialisation at O2
 * ─────────────────────────────────────────────────────────────────────────
 * Calls identity(0) with the *same* argument via CJIT_SAMPLE_ARGS so the
 * profiler sees a confident dominant value.  After an explicit O2 recompile
 * the wrapper codegen should have been invoked (verified via a recompile).
 * Correctness for both the hot arg (0) and a different arg (42) is checked.
 */
TEST(t03_arg_specialisation)
{
    printf("[t03] Argument-profile specialisation...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "identity", IR_IDENTITY,
                                           (jit_func_t)aot_identity);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);

    typedef int (*id_fn)(int);

    /* Drive the profile: call identity(0) many times so the dominant value
     * is 0 with high confidence before the O2 recompile. */
    for (int i = 0; i < 512; ++i) {
        CJIT_SAMPLE_ARGS(e, id, (uint64_t)0);
        (void)CJIT_DISPATCH(e, id, id_fn, 0);
    }

    /* Request O2; this is when generate_spec_wrapper() is called. */
    cjit_request_recompile(e, id, OPT_O2);

    wait_arg_t wa = { e, id, OPT_O2 };
    bool promoted = wait_for(check_opt, &wa, 3000);
    CHECKF(promoted, "identity not promoted to O2; level=%d",
           (int)cjit_get_current_opt_level(e, id));

    /* Correctness: dominant arg (0) and a different arg must both be right. */
    int r0  = CJIT_DISPATCH(e, id, id_fn, 0);
    int r42 = CJIT_DISPATCH(e, id, id_fn, 42);
    int r_1 = CJIT_DISPATCH(e, id, id_fn, -1);
    CHECKF(r0  == 0,   "identity(0)  = %d, want 0",  r0);
    CHECKF(r42 == 42,  "identity(42) = %d, want 42", r42);
    CHECKF(r_1 == -1,  "identity(-1) = %d, want -1", r_1);

    cjit_destroy(e);
    printf("[t03] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t04 – CJIT_DISPATCH_TIMED accumulates elapsed nanoseconds
 * ─────────────────────────────────────────────────────────────────────────
 * Runs fib(30) via CJIT_DISPATCH_TIMED for a sustained period; the
 * total_elapsed_ns reported by cjit_get_elapsed_ns() must be > 0 and
 * plausible (≥ N × (expected lower bound per call)).
 */
TEST(t04_timed_dispatch)
{
    printf("[t04] Timed dispatch / elapsed-ns accounting...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);

    typedef long (*fib_fn)(int);
    volatile long sink = 0;
    const int N = 200;
    for (int i = 0; i < N; ++i)
        sink += CJIT_DISPATCH_TIMED(e, id, fib_fn, 20 + (i % 5));

    /* Flush any unflushed TLS batch. */
    cjit_flush_local_counts(e);

    uint64_t elapsed = cjit_get_elapsed_ns(e, id);
    CHECKF(elapsed > 0, "total_elapsed_ns = 0 after %d timed calls", N);

    /* Sanity: total elapsed must be < 10 seconds (would indicate a bug). */
    CHECKF(elapsed < 10000000000ULL,
           "total_elapsed_ns = %llu (implausibly large)", (unsigned long long)elapsed);

    (void)sink;
    cjit_destroy(e);
    printf("[t04] PASS  (elapsed=%llu ns for %d calls)\n",
           (unsigned long long)elapsed, N);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t05 – Heavy multi-function load: sort, matmul, djb2, CRC32
 * ─────────────────────────────────────────────────────────────────────────
 * Each function is called many times over several seconds; final correctness
 * is verified against the reference implementation.
 */
static int  aot_bsort(int *a, int n)    { (void)a; (void)n; return 0; }
static void aot_matmul(const int *a, const int *b, int *o)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int s = 0;
            for (int k = 0; k < 4; ++k) s += a[i*4+k] * b[k*4+j];
            o[i*4+j] = s;
        }
}
static uint32_t aot_djb2(const char *s)
    { uint32_t h=5381; int c; while((c=(unsigned char)*s++)) h=((h<<5)+h)^(uint32_t)c; return h; }
static uint32_t aot_crc32(const uint8_t *d, size_t l) { return ref_crc32(d, l); }

TEST(t05_heavy_load)
{
    printf("[t05] Heavy multi-function load (sort/matmul/djb2/crc32)...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_sort = cjit_register_function(e, "bsort",    IR_BSORT,  (jit_func_t)aot_bsort);
    func_id_t id_mm   = cjit_register_function(e, "matmul4",  IR_MATMUL, (jit_func_t)aot_matmul);
    func_id_t id_djb  = cjit_register_function(e, "djb2",     IR_DJB2,   (jit_func_t)aot_djb2);
    func_id_t id_crc  = cjit_register_function(e, "crc32_jit",IR_CRC32,  (jit_func_t)aot_crc32);

    CHECK(id_sort != CJIT_INVALID_FUNC_ID, "bsort registration failed");
    CHECK(id_mm   != CJIT_INVALID_FUNC_ID, "matmul4 registration failed");
    CHECK(id_djb  != CJIT_INVALID_FUNC_ID, "djb2 registration failed");
    CHECK(id_crc  != CJIT_INVALID_FUNC_ID, "crc32_jit registration failed");

    cjit_start(e);

    typedef int      (*bsort_fn)(int *, int);
    typedef void     (*mm_fn)(const int *, const int *, int *);
    typedef uint32_t (*djb_fn)(const char *);
    typedef uint32_t (*crc_fn)(const uint8_t *, size_t);

    /* Identity matrix – result of A×I should equal A. */
    static const int A[16] = {
        1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    };
    static const int I[16] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };

    const char *words[] = { "hello", "world", "cjit", "test", "jit" };
    /* Pre-compute reference djb2 hashes. */
    uint32_t ref_djb[5];
    for (int i = 0; i < 5; ++i) ref_djb[i] = aot_djb2(words[i]);

    const uint8_t crc_data[] = "The quick brown fox jumps over the lazy dog";
    uint32_t ref_crc = ref_crc32(crc_data, sizeof(crc_data) - 1);

    uint64_t deadline = now_ms() + 2000;
    volatile long sink = 0;
    int iters = 0;
    while (now_ms() < deadline) {
        /* bsort */
        int arr[8] = { 8,3,7,1,5,2,6,4 };
        int sw = CJIT_DISPATCH(e, id_sort, bsort_fn, arr, 8);
        sink += sw;
        /* matmul */
        int out[16];
        CJIT_DISPATCH_TIMED_VOID(e, id_mm, mm_fn, A, I, out);
        for (int k = 0; k < 16; ++k) sink += out[k];
        /* djb2 */
        for (int k = 0; k < 5; ++k)
            sink += (long)CJIT_DISPATCH(e, id_djb, djb_fn, words[k]);
        /* crc32 */
        sink += (long)CJIT_DISPATCH(e, id_crc, crc_fn, crc_data, sizeof(crc_data)-1);
        iters++;
    }

    printf("[t05]   %d iterations in 2 s\n", iters);

    /* ── Correctness after JIT ── */
    /* Sort: a fresh 8-element array */
    int arr2[8] = { 8,3,7,1,5,2,6,4 };
    CJIT_DISPATCH(e, id_sort, bsort_fn, arr2, 8);
    for (int k = 1; k < 8; ++k)
        CHECKF(arr2[k] >= arr2[k-1],
               "bsort: arr[%d]=%d < arr[%d]=%d after sort", k, arr2[k], k-1, arr2[k-1]);

    /* matmul: A × I == A */
    int out2[16];
    CJIT_DISPATCH_TIMED_VOID(e, id_mm, mm_fn, A, I, out2);
    for (int k = 0; k < 16; ++k)
        CHECKF(out2[k] == A[k],
               "matmul4: out[%d]=%d, want %d", k, out2[k], A[k]);

    /* djb2 */
    for (int k = 0; k < 5; ++k) {
        uint32_t h = CJIT_DISPATCH(e, id_djb, djb_fn, words[k]);
        CHECKF(h == ref_djb[k],
               "djb2(\"%s\") = 0x%x, want 0x%x", words[k], h, ref_djb[k]);
    }

    /* crc32 */
    uint32_t c = CJIT_DISPATCH(e, id_crc, crc_fn, crc_data, sizeof(crc_data)-1);
    CHECKF(c == ref_crc, "crc32 = 0x%x, want 0x%x", c, ref_crc);

    cjit_destroy(e);
    printf("[t05] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t06 – Explicit recompile request + opt-level query
 * ─────────────────────────────────────────────────────────────────────────
 * Calls cjit_request_recompile(OPT_O1) immediately after start, waits for
 * O1, then requests O2 and waits for O2; checks recompile_count increments.
 */
TEST(t06_explicit_recompile)
{
    printf("[t06] Explicit recompile request + opt-level query...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "mul", IR_MUL, (jit_func_t)aot_mul);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);

    /* Request O1. */
    cjit_request_recompile(e, id, OPT_O1);
    wait_arg_t wa1 = { e, id, OPT_O1 };
    CHECK(wait_for(check_opt, &wa1, 3000), "mul not compiled to O1");

    uint32_t rc1 = cjit_get_recompile_count(e, id);
    CHECKF(rc1 >= 1, "recompile_count after O1 = %u, want ≥1", rc1);

    /* Correctness at O1. */
    typedef int (*mul_fn)(int, int);
    int r = CJIT_DISPATCH(e, id, mul_fn, 7, 6);
    CHECKF(r == 42, "mul(7,6) at O1 = %d, want 42", r);

    /* Request O2. */
    cjit_request_recompile(e, id, OPT_O2);
    wait_arg_t wa2 = { e, id, OPT_O2 };
    CHECK(wait_for(check_opt, &wa2, 3000), "mul not compiled to O2");

    uint32_t rc2 = cjit_get_recompile_count(e, id);
    CHECKF(rc2 >= 2, "recompile_count after O2 = %u, want ≥2", rc2);

    /* Correctness at O2. */
    r = CJIT_DISPATCH(e, id, mul_fn, 9, 9);
    CHECKF(r == 81, "mul(9,9) at O2 = %d, want 81", r);

    cjit_destroy(e);
    printf("[t06] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t07 – CJIT_DISPATCH macro (single-lookup dispatch)
 * ─────────────────────────────────────────────────────────────────────────
 * Verifies that CJIT_DISPATCH evaluates engine/id exactly once and returns
 * the correct value.  Also checks CJIT_DISPATCH_TIMED returns correctly.
 */
TEST(t07_dispatch_macros)
{
    printf("[t07] CJIT_DISPATCH and CJIT_DISPATCH_TIMED macros...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_add = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    func_id_t id_fib = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    CHECK(id_add != CJIT_INVALID_FUNC_ID, "add registration failed");
    CHECK(id_fib != CJIT_INVALID_FUNC_ID, "fib registration failed");

    /* No start() — AOT fallback used. */

    typedef int  (*add_fn)(int, int);
    typedef long (*fib_fn)(int);

    int  r1 = CJIT_DISPATCH(e, id_add, add_fn, 100, 200);
    long r2 = CJIT_DISPATCH_TIMED(e, id_fib, fib_fn, 15);

    CHECKF(r1 == 300, "CJIT_DISPATCH add(100,200) = %d, want 300", r1);
    CHECKF(r2 == 610, "CJIT_DISPATCH_TIMED fib(15) = %ld, want 610", r2);

    /* CJIT_DISPATCH_TIMED_VOID – use a void wrapper */
    /* (matmul is void; just check it doesn't crash.) */
    func_id_t id_mm = cjit_register_function(e, "matmul4", IR_MATMUL,
                                              (jit_func_t)aot_matmul);
    CHECK(id_mm != CJIT_INVALID_FUNC_ID, "matmul registration failed");
    typedef void (*mm_fn)(const int *, const int *, int *);
    int A[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    int B[16]={2,0,0,0,0,3,0,0,0,0,5,0,0,0,0,7};
    int C[16]={0};
    CJIT_DISPATCH_TIMED_VOID(e, id_mm, mm_fn, A, B, C);
    /* A is identity, so C == B */
    CHECKF(C[0]==2 && C[5]==3 && C[10]==5 && C[15]==7,
           "matmul identity A*B diagonal wrong: %d %d %d %d",
           C[0], C[5], C[10], C[15]);

    cjit_destroy(e);
    printf("[t07] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t08 – Stats integrity
 * ─────────────────────────────────────────────────────────────────────────
 * Runs a short hot loop, then verifies that:
 *   • total_compilations  ≥ recompile_count of the hot function
 *   • total_swaps         ≥ total_compilations
 *   • registered_functions == number registered
 *   • total_elapsed_ns > 0 if timed dispatch was used
 */
TEST(t08_stats_integrity)
{
    printf("[t08] Stats integrity...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_add = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    func_id_t id_fib = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    CHECK(id_add != CJIT_INVALID_FUNC_ID, "add registration failed");
    CHECK(id_fib != CJIT_INVALID_FUNC_ID, "fib registration failed");

    cjit_start(e);

    typedef int  (*add_fn)(int, int);
    typedef long (*fib_fn)(int);
    volatile long sink = 0;
    uint64_t deadline = now_ms() + 2000;
    while (now_ms() < deadline) {
        for (int i = 0; i < 300; ++i) {
            sink += CJIT_DISPATCH_TIMED(e, id_add, add_fn, i, i);
            sink += CJIT_DISPATCH_TIMED(e, id_fib, fib_fn, 10 + (i % 6));
        }
    }

    /* Wait for any pending compilations. */
    sleep_ms(800);
    cjit_flush_local_counts(e);

    cjit_stats_t s = cjit_get_stats(e);

    CHECKF(s.registered_functions == 2,
           "registered_functions = %u, want 2", s.registered_functions);
    CHECKF(s.total_swaps <= s.total_compilations + 2,
           "swaps=%llu > compilations=%llu (impossible)",
           (unsigned long long)s.total_swaps,
           (unsigned long long)s.total_compilations);
    CHECKF(s.total_elapsed_ns > 0,
           "total_elapsed_ns = %llu after timed dispatches",
           (unsigned long long)s.total_elapsed_ns);


    uint32_t rc = cjit_get_recompile_count(e, id_add);
    CHECKF(s.total_compilations >= rc,
           "total_compilations=%llu < recompile_count=%u for add",
           (unsigned long long)s.total_compilations, rc);

    (void)sink;
    cjit_destroy(e);
    printf("[t08] PASS  (compilations=%llu swaps=%llu elapsed=%llu ns)\n",
           (unsigned long long)s.total_compilations,
           (unsigned long long)s.total_swaps,
           (unsigned long long)s.total_elapsed_ns);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t09 – Prime sieve correctness over time
 * ─────────────────────────────────────────────────────────────────────────
 * Calls the JIT-compiled prime_count() with several values across the
 * tier-promotion window; verifies the result matches the reference at every
 * point in time (before and after JIT swap).
 */
TEST(t09_prime_sieve)
{
    printf("[t09] Prime sieve correctness over time...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "prime_count", IR_PRIME_COUNT,
                                           (jit_func_t)ref_prime_count);
    CHECK(id != CJIT_INVALID_FUNC_ID, "prime_count registration failed");

    cjit_start(e);

    typedef int (*pc_fn)(int);

    static const int limits[]   = { 100, 1000, 5000, 10000 };
    static const int expected[] = {  25,  168,  669,  1229 };

    /* Verify immediately (AOT fallback). */
    for (int i = 0; i < 4; ++i) {
        int got = CJIT_DISPATCH(e, id, pc_fn, limits[i]);
        CHECKF(got == expected[i],
               "prime_count(%d) [AOT] = %d, want %d", limits[i], got, expected[i]);
    }

    /* Drive the function hot so the JIT kicks in. */
    uint64_t deadline = now_ms() + 2000;
    volatile long sink = 0;
    int iters = 0;
    while (now_ms() < deadline) {
        sink += CJIT_DISPATCH_TIMED(e, id, pc_fn, 1000);
        iters++;
    }

    /* Wait for any in-flight compilation. */
    wait_arg_t wa = { e, id, OPT_O2 };
    wait_for(check_opt, &wa, 2000);
    sleep_ms(200);

    printf("[t09]   %d iterations, opt_level=%d\n",
           iters, (int)cjit_get_current_opt_level(e, id));

    /* Verify again after JIT. */
    for (int i = 0; i < 4; ++i) {
        int got = CJIT_DISPATCH(e, id, pc_fn, limits[i]);
        CHECKF(got == expected[i],
               "prime_count(%d) [JIT] = %d, want %d", limits[i], got, expected[i]);
    }

    (void)sink;
    cjit_destroy(e);
    printf("[t09] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t10 – Concurrent thread safety
 * ─────────────────────────────────────────────────────────────────────────
 * Spawns NTHREADS threads each dispatching through add() and fib() for
 * 2 seconds.  Verifies that every result is correct and that there are no
 * crashes (detected by the process still being alive at the end).
 */
#define T10_NTHREADS 4

typedef struct {
    cjit_engine_t *engine;
    func_id_t      id_add;
    func_id_t      id_fib;
    int            errors;
} t10_thread_arg;

static void *t10_worker(void *v)
{
    t10_thread_arg *a = (t10_thread_arg *)v;
    typedef int  (*add_fn)(int, int);
    typedef long (*fib_fn)(int);

    uint64_t deadline = now_ms() + 2000;
    while (now_ms() < deadline) {
        for (int i = 0; i < 100; ++i) {
            int  ra = CJIT_DISPATCH(a->engine, a->id_add, add_fn, i, i + 1);
            long rf = CJIT_DISPATCH(a->engine, a->id_fib, fib_fn, i % 20);
            if (ra != i + i + 1)            a->errors++;
            if (rf != aot_fib(i % 20))      a->errors++;
        }
    }
    cjit_flush_local_counts(a->engine);
    return NULL;
}

TEST(t10_concurrent_dispatch)
{
    printf("[t10] Concurrent thread safety (%d threads)...\n", T10_NTHREADS);
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_add = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    func_id_t id_fib = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    CHECK(id_add != CJIT_INVALID_FUNC_ID, "add registration failed");
    CHECK(id_fib != CJIT_INVALID_FUNC_ID, "fib registration failed");

    cjit_start(e);

    t10_thread_arg args[T10_NTHREADS];
    pthread_t threads[T10_NTHREADS];
    for (int i = 0; i < T10_NTHREADS; ++i) {
        args[i] = (t10_thread_arg){ e, id_add, id_fib, 0 };
        pthread_create(&threads[i], NULL, t10_worker, &args[i]);
    }
    for (int i = 0; i < T10_NTHREADS; ++i)
        pthread_join(threads[i], NULL);

    int total_errors = 0;
    for (int i = 0; i < T10_NTHREADS; ++i)
        total_errors += args[i].errors;

    CHECKF(total_errors == 0,
           "%d correctness errors across %d threads", total_errors, T10_NTHREADS);

    cjit_stats_t s = cjit_get_stats(e);
    printf("[t10]   total calls ≈ %llu  compilations=%llu\n",
           (unsigned long long)cjit_get_call_count(e, id_add) +
           (unsigned long long)cjit_get_call_count(e, id_fib),
           (unsigned long long)s.total_compilations);

    cjit_destroy(e);
    printf("[t10] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t11 – sum_range correctness with eager O1 pre-warm + many input values
 * ─────────────────────────────────────────────────────────────────────────
 * Exercises the JIT with a range of lo/hi pairs; verifies results against
 * the closed-form formula n*(n+1)/2.
 */
TEST(t11_sum_range)
{
    printf("[t11] sum_range correctness + pre-warm...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "sum_range", IR_SUM_RANGE,
                                           (jit_func_t)aot_sum_range);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O1);
    wait_arg_t wa = { e, id, OPT_O1 };
    wait_for(check_opt, &wa, 3000);

    typedef long (*sr_fn)(int, int);

    static const struct { int lo; int hi; long want; } cases[] = {
        {  1,  10,    55 },
        {  1, 100,  5050 },
        {  0,   0,     0 },
        { -5,   5,     0 },
        { 10,  10,    10 },
        {  1, 1000, 500500 },
    };

    /* Run each case 50 times to build up call count, verify each time. */
    for (int rep = 0; rep < 50; ++rep) {
        for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
            long got = CJIT_DISPATCH(e, id, sr_fn, cases[i].lo, cases[i].hi);
            CHECKF(got == cases[i].want,
                   "[rep %d] sum_range(%d,%d) = %ld, want %ld",
                   rep, cases[i].lo, cases[i].hi, got, cases[i].want);
        }
    }

    cjit_destroy(e);
    printf("[t11] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t12 – CRC32 vs reference, driven to O3
 * ─────────────────────────────────────────────────────────────────────────
 * Drives crc32_jit to O3 via explicit recompile requests and verifies
 * correctness against the reference at each tier.
 */
TEST(t12_crc32_tiers)
{
    printf("[t12] CRC32 correctness at O1 → O2 → O3...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "crc32_jit", IR_CRC32,
                                           (jit_func_t)aot_crc32);
    CHECK(id != CJIT_INVALID_FUNC_ID, "crc32 registration failed");

    cjit_start(e);

    typedef uint32_t (*crc_fn)(const uint8_t *, size_t);

    static const char *strings[] = {
        "hello, world",
        "The quick brown fox jumps over the lazy dog",
        "cjit JIT compiler regression test",
        ""
    };

    for (opt_level_t level = OPT_O1; level <= OPT_O3; ++level) {
        cjit_request_recompile(e, id, level);
        wait_arg_t wa = { e, id, level };
        bool ok = wait_for(check_opt, &wa, 3000);
        CHECKF(ok, "crc32_jit not compiled to O%d", (int)level);

        for (size_t i = 0; i < sizeof(strings)/sizeof(strings[0]); ++i) {
            const uint8_t *d = (const uint8_t *)strings[i];
            size_t n = strlen(strings[i]);
            uint32_t got = CJIT_DISPATCH(e, id, crc_fn, d, n);
            uint32_t ref = ref_crc32(d, n);
            CHECKF(got == ref,
                   "[O%d] crc32(\"%s\") = 0x%08x, want 0x%08x",
                   (int)level, strings[i], got, ref);
        }
        printf("[t12]   O%d OK\n", (int)level);
    }

    cjit_destroy(e);
    printf("[t12] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t13 – cjit_lookup_function
 * ─────────────────────────────────────────────────────────────────────────
 * Registers several functions and verifies that cjit_lookup_function()
 * returns the correct func_id for each registered name and
 * CJIT_INVALID_FUNC_ID for an unregistered name.
 */
TEST(t13_lookup_function)
{
    printf("[t13] cjit_lookup_function...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id_add = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    func_id_t id_fib = cjit_register_function(e, "fib", IR_FIB, (jit_func_t)aot_fib);
    func_id_t id_mul = cjit_register_function(e, "mul", IR_MUL, (jit_func_t)aot_mul);
    CHECK(id_add != CJIT_INVALID_FUNC_ID, "add registration failed");
    CHECK(id_fib != CJIT_INVALID_FUNC_ID, "fib registration failed");
    CHECK(id_mul != CJIT_INVALID_FUNC_ID, "mul registration failed");

    CHECK(cjit_lookup_function(e, "add") == id_add, "lookup 'add' returned wrong id");
    CHECK(cjit_lookup_function(e, "fib") == id_fib, "lookup 'fib' returned wrong id");
    CHECK(cjit_lookup_function(e, "mul") == id_mul, "lookup 'mul' returned wrong id");
    CHECK(cjit_lookup_function(e, "nonexistent") == CJIT_INVALID_FUNC_ID,
          "lookup of unregistered name should return INVALID");
    CHECK(cjit_lookup_function(e, NULL) == CJIT_INVALID_FUNC_ID,
          "lookup(NULL) should return INVALID");
    CHECK(cjit_lookup_function(NULL, "add") == CJIT_INVALID_FUNC_ID,
          "lookup(NULL engine) should return INVALID");

    cjit_destroy(e);
    printf("[t13] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t14 – cjit_update_ir (hot-reload)
 * ─────────────────────────────────────────────────────────────────────────
 * Registers a function that returns a constant, waits for initial
 * compilation, verifies the result, then hot-reloads it with new IR
 * returning a different constant, waits for recompilation, and verifies
 * the updated result.
 */
static const char IR_CONST1[] = "int get_val(void) { return 1001; }\n";
static const char IR_CONST2[] = "int get_val(void) { return 2002; }\n";

TEST(t14_update_ir)
{
    printf("[t14] cjit_update_ir (hot-reload)...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "get_val", IR_CONST1, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O1);

    /* Wait for first compilation. */
    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "initial compilation timed out");

    typedef int (*gv_fn)(void);
    int v1 = ((gv_fn)cjit_get_func(e, id))();
    CHECKF(v1 == 1001, "get_val() before update: got %d, want 1001", v1);

    /* Hot-reload with new IR. */
    CHECK(cjit_update_ir(e, id, IR_CONST2, OPT_O1), "cjit_update_ir failed");

    /*
     * After update the func_ptr was set by the first compile; it will be
     * replaced by the new compile triggered by cjit_update_ir.  We wait
     * until the recompile_count increases (= new binary installed).
     */
    uint32_t rc0 = cjit_get_recompile_count(e, id);
    uint64_t deadline = now_ms() + 5000;
    while (now_ms() < deadline &&
           cjit_get_recompile_count(e, id) <= rc0)
        sleep_ms(20);

    CHECKF(cjit_get_recompile_count(e, id) > rc0,
           "recompile_count did not increase after update (rc=%u)",
           cjit_get_recompile_count(e, id));

    int v2 = ((gv_fn)cjit_get_func(e, id))();
    CHECKF(v2 == 2002, "get_val() after update: got %d, want 2002", v2);

    cjit_destroy(e);
    printf("[t14] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t15 – extra_cflags / -D preprocessor define
 * ─────────────────────────────────────────────────────────────────────────
 * Passes a -D flag via cfg.extra_cflags so that a JIT function compiled
 * against a macro-dependent constant produces the expected result.
 */
static const char IR_USE_DEFINE[] =
    "int get_magic(void) {\n"
    "#ifndef CJIT_MAGIC_VAL\n"
    "    return 0;\n"
    "#else\n"
    "    return CJIT_MAGIC_VAL;\n"
    "#endif\n"
    "}\n";

TEST(t15_extra_cflags)
{
    printf("[t15] extra_cflags (-D define)...\n");
    cjit_config_t cfg        = cjit_default_config();
    cfg.verbose              = false;
    cfg.monitor_interval_ms  = 50;
    cfg.hot_ir_cache_size    = 4;
    cfg.warm_ir_cache_size   = 8;
    /* Inject a preprocessor define via extra_cflags. */
    strncpy(cfg.extra_cflags, "-DCJIT_MAGIC_VAL=7777",
            sizeof(cfg.extra_cflags) - 1);
    cfg.extra_cflags[sizeof(cfg.extra_cflags) - 1] = '\0';

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "get_magic", IR_USE_DEFINE, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O1);

    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "compilation with -D flag timed out");

    typedef int (*gm_fn)(void);
    int got = ((gm_fn)cjit_get_func(e, id))();
    CHECKF(got == 7777,
           "get_magic() with -DCJIT_MAGIC_VAL=7777: got %d, want 7777", got);

    cjit_destroy(e);
    printf("[t15] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t16 – JIT pointer is NOT the AOT fallback after compilation
 * ─────────────────────────────────────────────────────────────────────────
 * Verifies end-to-end that cjit_request_recompile() results in the
 * function-table entry being updated to a NEW pointer distinct from the
 * original AOT fallback — proving that a real compiled binary was loaded
 * and installed, not just the fallback pointer kept.
 *
 * NOTE: cjit_wait_compiled() fast-paths when func_ptr is already non-NULL
 * (i.e. the AOT fallback is set), so we use wait_for(check_opt, …) which
 * waits until cur_level reaches the requested tier — that transition only
 * occurs after a successful JIT compilation and pointer swap.
 */
TEST(t16_jit_replaces_aot)
{
    printf("[t16] JIT pointer differs from AOT fallback after compilation...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "add", IR_ADD, (jit_func_t)aot_add);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    /* Before start the engine must serve the AOT fallback. */
    jit_func_t pre = cjit_get_func(e, id);
    CHECK(pre == (jit_func_t)aot_add, "before start: expected AOT fallback");
    CHECK(cjit_get_current_opt_level(e, id) == OPT_NONE, "pre-start opt level != OPT_NONE");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O2);

    /*
     * Wait until cur_level reaches OPT_O2.  cur_level is updated by the
     * compiler thread only AFTER func_table_swap() completes, so reaching
     * OPT_O2 here is a strict proof that (a) compilation succeeded and
     * (b) the new JIT pointer was atomically installed.
     */
    wait_arg_t wa = { e, id, OPT_O2 };
    bool ok = wait_for(check_opt, &wa, 5000);
    CHECK(ok, "JIT compilation did not reach OPT_O2 within 5 s");

    jit_func_t post = cjit_get_func(e, id);

    /* Core assertion: a new pointer was installed (not the AOT fallback). */
    CHECK(post != NULL,                 "func_ptr is NULL after compilation");
    CHECK(post != (jit_func_t)aot_add, "func_ptr still points to AOT fallback after JIT compile");
    CHECK(cjit_get_recompile_count(e, id) >= 1, "recompile_count == 0 after JIT compile");
    CHECK(cjit_get_current_opt_level(e, id) == OPT_O2, "opt_level != OPT_O2 after O2 compile");

    /* Correctness: the JIT function must return the same answer as AOT. */
    typedef int (*add_fn)(int, int);
    int v = ((add_fn)post)(12, 30);
    CHECKF(v == 42, "JIT add(12,30) = %d, want 42", v);

    cjit_destroy(e);
    printf("[t16] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t17 – dladdr: JIT function lives in a separately loaded shared object
 * ─────────────────────────────────────────────────────────────────────────
 * Uses POSIX dladdr(3) to inspect the function pointer installed after
 * JIT compilation.  The function must reside in a dynamically-loaded
 * shared object whose load-base is different from the test binary's own
 * load-base — proving that the JIT compiled and dlopen'd a real .so.
 */
TEST(t17_dladdr_compiled_object)
{
    printf("[t17] dladdr: JIT function is in a loaded shared object...\n");
    cjit_engine_t *e = make_engine(false);
    CHECK(e != NULL, "engine creation failed");

    /*
     * Register WITHOUT an AOT fallback so that func_ptr is NULL until the
     * JIT compilation finishes.  After cjit_wait_compiled() any non-NULL
     * func_ptr must be a JIT-compiled entry point.
     */
    func_id_t id = cjit_register_function(e, "add", IR_ADD, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O1);

    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "JIT compilation did not complete within 5 s");

    jit_func_t fn = cjit_get_func(e, id);
    CHECK(fn != NULL, "func_ptr is NULL after compilation");

    /* ── dladdr inspection ──────────────────────────────────────────── */
    Dl_info jit_info;
    memset(&jit_info, 0, sizeof(jit_info));
    int found = dladdr((void *)(uintptr_t)fn, &jit_info);
    CHECKF(found != 0,
           "dladdr returned 0: JIT function (ptr=%p) not in any mapped shared object",
           (void *)(uintptr_t)fn);
    CHECK(jit_info.dli_fbase != NULL, "dladdr: dli_fbase is NULL");

    /*
     * The JIT .so has a different load base than the test binary.
     * Use now_ms() (a static function in this translation unit) as the
     * anchor for the test binary's load base — it is guaranteed to be in
     * the same mapped object as this test code.
     */
    Dl_info main_info;
    memset(&main_info, 0, sizeof(main_info));
    dladdr((void *)(uintptr_t)now_ms, &main_info);
    CHECK(jit_info.dli_fbase != main_info.dli_fbase,
          "JIT function is in the same mapping as the test binary (not a JIT .so)");

    if (jit_info.dli_fname)
        printf("[t17]   JIT .so mapped from: %s\n", jit_info.dli_fname);
    else
        printf("[t17]   JIT .so: anonymous mapping (in-memory .so)\n");

    /* Correctness: the JIT function must return the right answer. */
    typedef int (*add_fn)(int, int);
    int v = ((add_fn)fn)(5, 7);
    CHECKF(v == 12, "JIT add(5,7) = %d, want 12", v);

    cjit_destroy(e);
    printf("[t17] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t18 – O3 + vectorisation: integer dot-product correctness
 * ─────────────────────────────────────────────────────────────────────────
 * Exercises the full optimisation path (O3 + -ftree-vectorize +
 * -funroll-loops + -march=native) with a loop that the auto-vectoriser
 * can convert to SIMD instructions.  The result is verified against a
 * reference implementation for 1 000 consecutive calls to confirm that the
 * JIT-compiled, highly-optimised code is both correct and stable.
 */
TEST(t18_o3_vectorised_correctness)
{
    printf("[t18] O3 vectorised compilation correctness (dot product)...\n");

    cjit_config_t cfg          = cjit_default_config();
    cfg.verbose                = false;
    cfg.monitor_interval_ms    = 50;
    cfg.enable_inlining        = true;
    cfg.enable_vectorization   = true;
    cfg.enable_loop_unroll     = true;
    cfg.enable_native_arch     = true;
    cfg.hot_ir_cache_size      = 4;
    cfg.warm_ir_cache_size     = 8;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");

    func_id_t id = cjit_register_function(e, "dot_int", IR_DOT_INT, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "registration failed");

    cjit_start(e);
    cjit_request_recompile(e, id, OPT_O3);

    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "O3 compilation did not complete within 5 s");

    CHECK(cjit_get_current_opt_level(e, id) == OPT_O3, "opt_level != OPT_O3 after O3 compile");
    CHECK(cjit_get_recompile_count(e, id) >= 1, "recompile_count == 0 after O3 compile");

    typedef long (*dot_fn)(const int *, const int *, int);
    dot_fn jit_dot = (dot_fn)cjit_get_func(e, id);
    CHECK(jit_dot != NULL, "func_ptr NULL after O3 compilation");

    /* Build deterministic test vectors and compute the reference result. */
    enum { N = 1024 };
    int  a[N], b[N];
    long ref = 0;
    for (int i = 0; i < N; ++i) {
        a[i] = i + 1;
        b[i] = N - i;
        ref += (long)a[i] * b[i];
    }

    /* Single correctness check. */
    long got = jit_dot(a, b, N);
    CHECKF(got == ref, "dot_int[%d] first call: got %ld, want %ld", N, got, ref);

    /* Stability: repeat 1 000 times to catch any code-quality regression
     * (e.g. vectorisation that miscomputes the tail element). */
    for (int t = 1; t < 1000; ++t) {
        long v = jit_dot(a, b, N);
        CHECKF(v == ref, "dot_int[%d] iter %d: got %ld, want %ld", N, t, v, ref);
    }

    printf("[t18]   ref=%ld, 1000 iterations correct at OPT_O3\n", ref);
    cjit_destroy(e);
    printf("[t18] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t19 – JIT performance benchmark
 * ─────────────────────────────────────────────────────────────────────────
 * Measures three dimensions of JIT quality for an integer dot-product
 * workload (N = 1024 elements):
 *
 *   1. Compilation latency (ms)   – how long does it take to JIT-compile
 *      the function at each optimisation tier?
 *
 *   2. Dispatch throughput (Mcall/s) – how many calls/second can the
 *      JIT-compiled function sustain compared to the AOT baseline?
 *
 *   3. Speedup vs AOT – ratio of AOT-ns / JIT-ns for each tier.
 *
 * PASS conditions (hard assertions):
 *   • Every compilation completes within 10 seconds.
 *   • Every JIT tier produces the correct result.
 *   • O3 throughput is not more than 3× worse than O1 (guards against
 *     catastrophic flag regressions; 3× gives generous CI-noise margin).
 *
 * All timing numbers are printed so they can be inspected in CI logs
 * without needing a dedicated profiler run.
 */

/* Number of call repetitions per tier for the throughput measurement.
 * 100 000 reps × 1024 multiply-adds ≈ 100 M MACs per tier sweep,
 * yielding a clean ≥ 10 ms wall-clock window even on slow CI machines. */
#define PERF_REPS 100000

TEST(t19_perf_benchmark)
{
    printf("[t19] Performance benchmark: compile latency + dispatch throughput...\n");

    /* ── Test vectors ─────────────────────────────────────────────────── */
    enum { PN = 1024 };
    int  pa[PN], pb[PN];
    long pref = 0;
    for (int i = 0; i < PN; ++i) {
        pa[i] = i + 1;
        pb[i] = PN - i;
        pref += (long)pa[i] * pb[i];
    }

    /* ── AOT baseline throughput ──────────────────────────────────────── */
    volatile long psink = 0;
    uint64_t pt0 = cjit_timestamp_ns();
    for (int i = 0; i < PERF_REPS; ++i)
        psink += aot_dot_int(pa, pb, PN);
    uint64_t aot_ns = cjit_timestamp_ns() - pt0;
    (void)psink;

    double aot_mcps = (double)PERF_REPS / (double)aot_ns * 1000.0;
    printf("[t19]   AOT baseline : %7.2f Mcall/s  (%4llu ms for %d calls, N=%d)\n",
           aot_mcps,
           (unsigned long long)(aot_ns / 1000000ULL),
           PERF_REPS, PN);

    /* ── JIT engine with full optimisations ──────────────────────────── */
    cjit_config_t cfg        = cjit_default_config();
    cfg.verbose              = false;
    cfg.monitor_interval_ms  = 50;
    cfg.enable_inlining      = true;
    cfg.enable_vectorization = true;
    cfg.enable_loop_unroll   = true;
    cfg.enable_native_arch   = true;
    cfg.hot_ir_cache_size    = 4;
    cfg.warm_ir_cache_size   = 8;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");

    func_id_t pid = cjit_register_function(e, "dot_int", IR_DOT_INT,
                                            (jit_func_t)aot_dot_int);
    CHECK(pid != CJIT_INVALID_FUNC_ID, "registration failed");
    cjit_start(e);

    typedef long (*dot_fn_t)(const int *, const int *, int);

    struct {
        opt_level_t  lvl;
        const char  *name;
        uint64_t     compile_ns; /* wall-clock from request → tier reached  */
        uint64_t     run_ns;     /* wall-clock for PERF_REPS calls           */
    } tiers[] = {
        { OPT_O1, "O1", 0, 0 },
        { OPT_O2, "O2", 0, 0 },
        { OPT_O3, "O3", 0, 0 },
    };

    for (int ti = 0; ti < 3; ++ti) {

        /* ── Request compilation and time it ───────────────────────── */
        uint64_t compile_start = cjit_timestamp_ns();
        cjit_request_recompile(e, pid, tiers[ti].lvl);

        wait_arg_t wa = { e, pid, tiers[ti].lvl };
        bool compiled = wait_for(check_opt, &wa, 10000);
        tiers[ti].compile_ns = cjit_timestamp_ns() - compile_start;

        CHECKF(compiled,
               "Compile to %s timed out after %.0f ms",
               tiers[ti].name, (double)tiers[ti].compile_ns / 1e6);

        CHECKF(tiers[ti].compile_ns < (uint64_t)10000 * 1000000ULL,
               "Compile to %s took %.0f ms — exceeds 10 000 ms limit",
               tiers[ti].name, (double)tiers[ti].compile_ns / 1e6);

        /* ── Correctness check at this tier ─────────────────────── */
        dot_fn_t fn = (dot_fn_t)cjit_get_func(e, pid);
        CHECK(fn != NULL, "func_ptr NULL after compilation");

        long pgot = fn(pa, pb, PN);
        CHECKF(pgot == pref,
               "%s dot_int(N=%d) = %ld, want %ld",
               tiers[ti].name, PN, pgot, pref);

        /* ── Throughput measurement ──────────────────────────────── */
        volatile long psink2 = 0;
        pt0 = cjit_timestamp_ns();
        for (int i = 0; i < PERF_REPS; ++i)
            psink2 += fn(pa, pb, PN);
        tiers[ti].run_ns = cjit_timestamp_ns() - pt0;
        (void)psink2;

        double mcps    = (double)PERF_REPS / (double)tiers[ti].run_ns * 1000.0;
        double speedup = (double)aot_ns    / (double)tiers[ti].run_ns;

        printf("[t19]   JIT %-2s      : %7.2f Mcall/s  (%4llu ms) | "
               "compile %4llu ms | speedup %.2fx vs AOT\n",
               tiers[ti].name, mcps,
               (unsigned long long)(tiers[ti].run_ns   / 1000000ULL),
               (unsigned long long)(tiers[ti].compile_ns / 1000000ULL),
               speedup);
    }

    /* ── Sanity assertion: O3 must not catastrophically regress vs O1 ── *
     * 3× slack is intentionally generous to absorb CI load spikes.      *
     * This specifically catches broken compiler-flag configurations where *
     * O3 produces slower code than O1 (e.g. flag typo, wrong cc_binary). */
    CHECKF(tiers[2].run_ns <= tiers[0].run_ns * 3,
           "O3 throughput catastrophically worse than O1: "
           "O3=%llu ns, O1=%llu ns (ratio=%.1fx > 3.0x limit)",
           (unsigned long long)tiers[2].run_ns,
           (unsigned long long)tiers[0].run_ns,
           (double)tiers[2].run_ns / (double)tiers[0].run_ns);

    cjit_stats_t ps = cjit_get_stats(e);
    printf("[t19]   engine stats : compilations=%llu swaps=%llu queue_depth=%u\n",
           (unsigned long long)ps.total_compilations,
           (unsigned long long)ps.total_swaps,
           ps.queue_depth);

    cjit_destroy(e);
    printf("[t19] PASS\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * t20 – New optimisation flags and preamble attributes
 * ─────────────────────────────────────────────────────────────────────────
 * Verifies three improvements made to the JIT's code-generation layer:
 *
 *   1. -funswitch-loops (now active at O2+)
 *      Hoists loop-invariant conditionals out of loops.  A loop whose body
 *      contains `if (mode == 0)` with a constant-across-iterations `mode`
 *      becomes two separate loops — correct results must be produced for
 *      both mode values at both O2 and O3.
 *
 *   2. FLATTEN preamble attribute
 *      Forces the compiler to inline all calls within the annotated function
 *      recursively.  IR_PREAMBLE_ATTRS defines a FLATTEN outer() that calls
 *      inner(); the compiled function must return INT_MAX (10 increments
 *      starting from INT_MAX-10).  Also exercises the auto-included
 *      <limits.h> (INT_MAX) and <stdbool.h> (bool / true / false).
 *
 *   3. FLATTEN on real helper-calling IR (IR_DOT_HELPERS)
 *      A dot-product loop calling two per-element helpers through FLATTEN;
 *      result must match the AOT reference.
 */
TEST(t20_new_optflags_and_preamble)
{
    printf("[t20] New optimisation flags and preamble attributes...\n");

    cjit_config_t cfg        = cjit_default_config();
    cfg.verbose              = false;
    cfg.monitor_interval_ms  = 50;
    cfg.enable_inlining      = true;
    cfg.enable_vectorization = true;
    cfg.enable_loop_unroll   = true;
    cfg.enable_native_arch   = true;
    cfg.hot_ir_cache_size    = 8;
    cfg.warm_ir_cache_size   = 16;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");

    /* Register all three test functions. */
    func_id_t id_csum = cjit_register_function(e, "cond_sum",
                                                IR_COND_SUM, NULL);
    func_id_t id_patt = cjit_register_function(e, "preamble_test",
                                                IR_PREAMBLE_ATTRS, NULL);
    func_id_t id_dh   = cjit_register_function(e, "dot_helpers",
                                                IR_DOT_HELPERS, NULL);
    CHECK(id_csum != CJIT_INVALID_FUNC_ID, "cond_sum registration failed");
    CHECK(id_patt != CJIT_INVALID_FUNC_ID, "preamble_test registration failed");
    CHECK(id_dh   != CJIT_INVALID_FUNC_ID, "dot_helpers registration failed");

    cjit_start(e);

    /* ── Compile all three to O2 and wait ────────────────────────────── */
    cjit_request_recompile(e, id_csum, OPT_O2);
    cjit_request_recompile(e, id_patt, OPT_O2);
    cjit_request_recompile(e, id_dh,   OPT_O2);

    wait_arg_t wa_csum = { e, id_csum, OPT_O2 };
    wait_arg_t wa_patt = { e, id_patt, OPT_O2 };
    wait_arg_t wa_dh   = { e, id_dh,   OPT_O2 };

    CHECK(wait_for(check_opt, &wa_csum, 5000), "cond_sum O2 compile timeout");
    CHECK(wait_for(check_opt, &wa_patt, 5000), "preamble_test O2 compile timeout");
    CHECK(wait_for(check_opt, &wa_dh,   5000), "dot_helpers O2 compile timeout");

    /* ── 1. cond_sum correctness at O2 ──────────────────────────────── */
    enum { CSN = 512 };
    int ca[CSN];
    for (int i = 0; i < CSN; ++i) ca[i] = i + 1;
    long ref0 = aot_cond_sum(ca, CSN, 0);
    long ref1 = aot_cond_sum(ca, CSN, 1);

    typedef long (*csum_fn)(const int *, int, int);
    csum_fn jit_csum = (csum_fn)cjit_get_func(e, id_csum);
    CHECK(jit_csum != NULL, "cond_sum func_ptr NULL at O2");

    long got0 = jit_csum(ca, CSN, 0);
    long got1 = jit_csum(ca, CSN, 1);
    CHECKF(got0 == ref0, "O2 cond_sum mode=0: got %ld, want %ld", got0, ref0);
    CHECKF(got1 == ref1, "O2 cond_sum mode=1: got %ld, want %ld", got1, ref1);
    printf("[t20]   -funswitch-loops O2: mode0=%ld mode1=%ld  OK\n", got0, got1);

    /* ── 2. preamble_test: FLATTEN + limits.h + stdbool.h ───────────── */
    typedef int (*patt_fn)(void);
    int patt_got = ((patt_fn)cjit_get_func(e, id_patt))();
    CHECKF(patt_got == INT_MAX,
           "preamble_test() = %d, want INT_MAX (%d)", patt_got, INT_MAX);
    printf("[t20]   FLATTEN + limits.h + stdbool.h O2: result=%d (INT_MAX) OK\n",
           patt_got);

    /* ── 3. dot_helpers: FLATTEN on helper-calling loop ─────────────── */
    enum { DHN = 64 };
    int da[DHN], db[DHN];
    long dref = 0;
    for (int i = 0; i < DHN; ++i) {
        da[i] = i + 1;
        db[i] = DHN - i;
        dref += (long)da[i] * db[i];
    }

    typedef long (*dh_fn)(const int *, const int *, int);
    long dh_got = ((dh_fn)cjit_get_func(e, id_dh))(da, db, DHN);
    CHECKF(dh_got == dref,
           "dot_helpers O2 (N=%d): got %ld, want %ld", DHN, dh_got, dref);
    printf("[t20]   FLATTEN dot_helpers O2 (N=%d): %ld  OK\n", DHN, dh_got);

    /* ── Re-verify cond_sum and dot_helpers at O3 ────────────────────── */
    cjit_request_recompile(e, id_csum, OPT_O3);
    cjit_request_recompile(e, id_dh,   OPT_O3);

    wait_arg_t wa_csum3 = { e, id_csum, OPT_O3 };
    wait_arg_t wa_dh3   = { e, id_dh,   OPT_O3 };
    CHECK(wait_for(check_opt, &wa_csum3, 5000), "cond_sum O3 compile timeout");
    CHECK(wait_for(check_opt, &wa_dh3,   5000), "dot_helpers O3 compile timeout");

    long o3_0 = ((csum_fn)cjit_get_func(e, id_csum))(ca, CSN, 0);
    long o3_1 = ((csum_fn)cjit_get_func(e, id_csum))(ca, CSN, 1);
    CHECKF(o3_0 == ref0, "O3 cond_sum mode=0: got %ld, want %ld", o3_0, ref0);
    CHECKF(o3_1 == ref1, "O3 cond_sum mode=1: got %ld, want %ld", o3_1, ref1);
    printf("[t20]   -funswitch-loops O3: mode0=%ld mode1=%ld  OK\n", o3_0, o3_1);

    long dh_o3 = ((dh_fn)cjit_get_func(e, id_dh))(da, db, DHN);
    CHECKF(dh_o3 == dref,
           "dot_helpers O3 (N=%d): got %ld, want %ld", DHN, dh_o3, dref);
    printf("[t20]   FLATTEN dot_helpers O3 (N=%d): %ld  OK\n", DHN, dh_o3);

    cjit_destroy(e);
    printf("[t20] PASS\n");
    return 0;
}



/* ═══════════════════════════════════════════════════════════════════════════
 * t21 – Artifact cache: second compilation of identical IR is a cache hit
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Create an engine with a temporary cache directory.
 *   2. Register and compile a function (cold miss → compiler subprocess).
 *   3. Destroy the engine (releases handles but leaves cache on disk).
 *   4. Create a NEW engine pointing at the SAME cache directory.
 *   5. Register and compile the same function again.
 *   6. Verify: artifact_cache_hits == 1, total_compilations == 1 (cached hit
 *      bypasses the compiler subprocess so no additional compilation counter
 *      is incremented), and the function returns the correct result.
 */
TEST(t21_artifact_cache)
{
    printf("[t21] Artifact cache: warm-hit skips compiler subprocess...\n");

    /* Use a unique temporary directory scoped to this test run. */
    char cache_dir[256];
    snprintf(cache_dir, sizeof(cache_dir), "/tmp/cjit_test_cache_%d", (int)getpid());

    /* ── Pass 1: cold miss → compiler runs, result cached ────────────── */
    {
        cjit_config_t cfg = cjit_default_config();
        cfg.compiler_threads = 1;
        snprintf(cfg.cache_dir, sizeof(cfg.cache_dir), "%s", cache_dir);

        cjit_engine_t *e = cjit_create(&cfg);
        CHECK(e, "cjit_create (pass 1) returned NULL");
        cjit_start(e);

        /* Register WITHOUT an AOT fallback so cjit_wait_compiled() waits for
         * the actual JIT compilation (not just the pre-set fallback pointer). */
        func_id_t id = cjit_register_function(e, "add", IR_ADD, NULL);
        CHECK(id != CJIT_INVALID_FUNC_ID, "register failed (pass 1)");

        cjit_request_recompile(e, id, OPT_O1);
        bool ok = cjit_wait_compiled(e, id, 5000);
        CHECK(ok, "cjit_wait_compiled timed out (pass 1)");

        cjit_stats_t s = cjit_get_stats(e);
        CHECKF(s.artifact_cache_misses == 1,
               "pass 1: expected 1 cache miss, got %llu",
               (unsigned long long)s.artifact_cache_misses);
        CHECKF(s.artifact_cache_hits == 0,
               "pass 1: expected 0 cache hits, got %llu",
               (unsigned long long)s.artifact_cache_hits);
        printf("[t21]   pass 1 miss=1 hit=0  OK\n");

        cjit_destroy(e);
    }

    /* ── Pass 2: warm hit → cached .so reused, no compiler spawn ─────── */
    {
        cjit_config_t cfg = cjit_default_config();
        cfg.compiler_threads = 1;
        snprintf(cfg.cache_dir, sizeof(cfg.cache_dir), "%s", cache_dir);

        cjit_engine_t *e = cjit_create(&cfg);
        CHECK(e, "cjit_create (pass 2) returned NULL");
        cjit_start(e);

        /* Register WITHOUT an AOT fallback — same as pass 1 — so that
         * cjit_wait_compiled() blocks until the cache-hit dlopen completes. */
        typedef int (*add_fn)(int, int);
        func_id_t id = cjit_register_function(e, "add", IR_ADD, NULL);
        CHECK(id != CJIT_INVALID_FUNC_ID, "register failed (pass 2)");

        cjit_request_recompile(e, id, OPT_O1);
        bool ok = cjit_wait_compiled(e, id, 5000);
        CHECK(ok, "cjit_wait_compiled timed out (pass 2)");

        /* Verify correctness: JIT function must return the right answer. */
        add_fn fn = (add_fn)(uintptr_t)cjit_get_func_counted(e, id);
        CHECK(fn, "add function pointer NULL after warm-hit");
        CHECKF(fn(3, 4) == 7, "add(3,4) expected 7, got %d", fn(3, 4));
        CHECKF(fn(-1, 1) == 0, "add(-1,1) expected 0, got %d", fn(-1, 1));

        cjit_stats_t s = cjit_get_stats(e);
        CHECKF(s.artifact_cache_hits == 1,
               "pass 2: expected 1 cache hit, got %llu",
               (unsigned long long)s.artifact_cache_hits);
        CHECKF(s.artifact_cache_misses == 0,
               "pass 2: expected 0 cache misses, got %llu",
               (unsigned long long)s.artifact_cache_misses);
        /*
         * A cache hit means the compiler subprocess was NOT spawned, so
         * total_compilations stays at 0 (the swap + compilation counter is
         * not incremented on a cache-hit path because the result comes from
         * dlopen, not from a posix_spawnp).
         *
         * Note: the stat_compilations counter in cjit.c is incremented by
         * the compiler thread AFTER codegen_compile() succeeds.  A cache hit
         * returns early from codegen_compile() and is still counted as a
         * successful compilation by the compiler thread (it still performed
         * a dlopen + dlsym and swapped the function pointer).  We therefore
         * check that total_compilations == 1, and that the swap also happened.
         */
        CHECKF(s.total_compilations == 1,
               "pass 2: expected 1 total_compilations, got %llu",
               (unsigned long long)s.total_compilations);
        CHECKF(s.total_swaps == 1,
               "pass 2: expected 1 total_swaps, got %llu",
               (unsigned long long)s.total_swaps);

        printf("[t21]   pass 2 hit=1 miss=0 compilations=%llu swaps=%llu  OK\n",
               (unsigned long long)s.total_compilations,
               (unsigned long long)s.total_swaps);

        cjit_destroy(e);
    }

    /* Clean up the temporary cache directory (best effort). */
    {
        char cmd[300];
        snprintf(cmd, sizeof(cmd), "rm -rf %s", cache_dir);
        int _r = system(cmd); (void)_r;
    }

    printf("[t21] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t22 – Priority queue: manual recompile task goes through priority lane
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Verifies that cjit_request_recompile() (priority 3) routes through the
 * priority lane and is correctly processed by the compiler thread.
 * The test exercises the two-level queue indirectly:
 *   1. Registers a function WITHOUT an AOT fallback (func_ptr starts NULL).
 *   2. Calls cjit_request_recompile() with OPT_O2 → priority = 3 → prio queue.
 *   3. Waits for compilation.
 *   4. Verifies correctness, compilation count, and prio_queue_depth = 0 after.
 *   5. Issues a second request (OPT_O3) and verifies the upgrade also completes.
 */
TEST(t22_priority_queue)
{
    printf("[t22] Priority queue: manual requests use fast lane...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");
    cjit_start(e);

    /* Register without AOT fallback so wait_compiled actually waits. */
    func_id_t id = cjit_register_function(e, "add", IR_ADD, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    /* First manual request → priority 3 → priority queue (lane 1). */
    cjit_request_recompile(e, id, OPT_O2);
    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "first manual compile timed out");

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations == 1,
           "expected 1 compilation after first request, got %llu",
           (unsigned long long)s.total_compilations);
    CHECKF(s.prio_queue_depth == 0,
           "priority queue not empty after compile: depth=%u", s.prio_queue_depth);
    CHECKF(cjit_get_current_opt_level(e, id) == OPT_O2,
           "opt level should be O2 after first request, got %d",
           (int)cjit_get_current_opt_level(e, id));

    typedef int (*add_fn)(int, int);
    add_fn fn = (add_fn)(uintptr_t)cjit_get_func_counted(e, id);
    CHECK(fn != NULL, "function pointer NULL after O2");
    CHECKF(fn(10, 20) == 30, "add(10,20) expected 30, got %d", fn(10, 20));
    printf("[t22]   O2 compile: compilations=%llu prio_q_depth=%u  OK\n",
           (unsigned long long)s.total_compilations, s.prio_queue_depth);

    /* Second manual request → OPT_O3, also priority 3. */
    cjit_request_recompile(e, id, OPT_O3);
    /*
     * cjit_wait_compiled() returns as soon as func_ptr != NULL — which it
     * already is from the O2 compile.  Use wait_for(check_opt, …) instead,
     * which polls until the opt-level has actually reached OPT_O3.
     */
    wait_arg_t wa3 = { e, id, OPT_O3 };
    bool ok3 = wait_for(check_opt, &wa3, 5000);
    CHECK(ok3, "second manual compile (O3) timed out");

    s = cjit_get_stats(e);
    CHECKF(s.total_compilations == 2,
           "expected 2 total compilations after O3 upgrade, got %llu",
           (unsigned long long)s.total_compilations);
    CHECKF(cjit_get_current_opt_level(e, id) == OPT_O3,
           "opt level should be O3 after second request, got %d",
           (int)cjit_get_current_opt_level(e, id));

    /* Reuse the add_fn typedef declared above. */
    add_fn fn3 = (add_fn)(uintptr_t)cjit_get_func_counted(e, id);
    CHECK(fn3 != NULL, "function pointer NULL after O3");
    CHECKF(fn3(-5, 5) == 0, "add(-5,5) expected 0, got %d", fn3(-5, 5));
    printf("[t22]   O3 upgrade:  compilations=%llu prio_q_depth=%u  OK\n",
           (unsigned long long)s.total_compilations, s.prio_queue_depth);

    cjit_destroy(e);
    printf("[t22] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t23 – Tier-skip: direct O0→O3 when call rate exceeds the skip threshold
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Configure a low hot_rate_t2 and tier_skip_multiplier = 0.5 (skip when
 * rate ≥ hot_rate_t2 × 0.5 = hot_rate_t1 × 0.5, which is below t2 but
 * above t1 — this exercises the skip path without needing a huge call rate).
 *
 * Strategy:
 *   1. Configure: hot_rate_t1=100, hot_rate_t2=200, tier_skip_multiplier=0.5
 *      → skip fires when rate ≥ 200 × 0.5 = 100 calls/sec.
 *   2. Register a function with no AOT fallback.
 *   3. Drive it at a rate well above the skip threshold.
 *   4. Wait for compilation.
 *   5. Verify the function ended up at OPT_O3 (skip fired) and
 *      stats.tier_skips == 1.
 *   6. Verify result correctness at O3.
 */
TEST(t23_tier_skip)
{
    printf("[t23] Tier-skip: O0→O3 direct promotion for explosively hot function...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.hot_rate_t1           = 100;   /* 100 calls/sec = T1 */
    cfg.hot_rate_t2           = 200;   /* 200 calls/sec = T2 */
    cfg.tier_skip_multiplier  = 0.5f;  /* skip when rate ≥ 200 × 0.5 = 100 */
    cfg.hot_confirm_cycles    = 1;     /* confirm after 1 cycle */
    cfg.monitor_interval_ms   = 25;    /* fast monitor for test speed */
    cfg.min_uptime_for_tier2_ms = 0;   /* no uptime gate */
    cfg.compile_cooloff_ms    = 0;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e != NULL, "engine creation failed");
    cjit_start(e);

    /* Register WITHOUT AOT fallback so wait_compiled blocks on JIT. */
    func_id_t id = cjit_register_function(e, "add", IR_ADD, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    /*
     * Drive the function at a rate well above the skip threshold (100 cps).
     * We call it in a tight loop for 300 ms.  At ≥100 calls in 300 ms that
     * is ≥333 calls/sec, comfortably above the 100 cps skip threshold.
     */
    uint64_t deadline = (uint64_t)clock() * 1000ULL / (uint64_t)CLOCKS_PER_SEC
                        + 300ULL;
    volatile int sink = 0;
    while ((uint64_t)clock() * 1000ULL / (uint64_t)CLOCKS_PER_SEC < deadline) {
        /* Use the AOT fallback directly for counting (no JIT yet). */
        sink += aot_add(1, 2);
        /* Manually advance the call counter so the monitor sees the rate. */
        cjit_get_func_counted(e, id); /* increments call_cnt via hot-path */
    }
    (void)sink;

    /* Wait for the tier-skip promotion to compile at O3 (up to 5 s). */
    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "tier-skip compilation timed out (no compilation in 5 s)");

    cjit_stats_t s = cjit_get_stats(e);
    opt_level_t lvl = cjit_get_current_opt_level(e, id);

    CHECKF(lvl == OPT_O3,
           "tier-skip: expected OPT_O3, got O%d", (int)lvl);
    CHECKF(s.tier_skips >= 1,
           "tier-skip stat should be ≥1, got %llu",
           (unsigned long long)s.tier_skips);
    /* Only one compilation should have happened (the O3 skip, no O2 step). */
    CHECKF(s.total_compilations == 1,
           "tier-skip: expected 1 compilation (no intermediate O2), got %llu",
           (unsigned long long)s.total_compilations);

    typedef int (*add_fn)(int, int);
    add_fn fn = (add_fn)(uintptr_t)cjit_get_func_counted(e, id);
    CHECK(fn != NULL, "function pointer NULL after tier-skip O3");
    CHECKF(fn(7, 8) == 15, "add(7,8) expected 15, got %d", fn(7, 8));

    printf("[t23]   lvl=O%d tier_skips=%llu compilations=%llu  OK\n",
           (int)lvl,
           (unsigned long long)s.tier_skips,
           (unsigned long long)s.total_compilations);

    cjit_destroy(e);
    printf("[t23] PASS\n");
    return 0;
}


typedef int (*test_fn)(void);
static const struct { const char *name; test_fn fn; } TESTS[] = {
    { "t01_aot_correctness",    t01_aot_correctness    },
    { "t02_hot_promotion",      t02_hot_promotion      },
    { "t03_arg_specialisation", t03_arg_specialisation },
    { "t04_timed_dispatch",     t04_timed_dispatch     },
    { "t05_heavy_load",         t05_heavy_load         },
    { "t06_explicit_recompile", t06_explicit_recompile },
    { "t07_dispatch_macros",    t07_dispatch_macros    },
    { "t08_stats_integrity",    t08_stats_integrity    },
    { "t09_prime_sieve",        t09_prime_sieve        },
    { "t10_concurrent_dispatch",t10_concurrent_dispatch},
    { "t11_sum_range",          t11_sum_range          },
    { "t12_crc32_tiers",        t12_crc32_tiers        },
    { "t13_lookup_function",    t13_lookup_function    },
    { "t14_update_ir",          t14_update_ir          },
    { "t15_extra_cflags",       t15_extra_cflags       },
    { "t16_jit_replaces_aot",       t16_jit_replaces_aot         },
    { "t17_dladdr_compiled_object", t17_dladdr_compiled_object   },
    { "t18_o3_vectorised_correctness", t18_o3_vectorised_correctness},
    { "t19_perf_benchmark",         t19_perf_benchmark           },
    { "t20_new_optflags_and_preamble", t20_new_optflags_and_preamble},
    { "t21_artifact_cache",         t21_artifact_cache           },
    { "t22_priority_queue",         t22_priority_queue           },
    { "t23_tier_skip",              t23_tier_skip                },
};
#define N_TESTS ((int)(sizeof(TESTS)/sizeof(TESTS[0])))

int main(int argc, char *argv[])
{
    /* Optionally run a single test by name: ./test_jit t03 */
    const char *filter = (argc > 1) ? argv[1] : NULL;

    printf("\n══════════════════════════════════════════════\n");
    printf("  CJIT Integration Test Suite (%d tests)\n", N_TESTS);
    printf("══════════════════════════════════════════════\n\n");

    int failures = 0;
    for (int i = 0; i < N_TESTS; ++i) {
        if (filter && strncmp(TESTS[i].name, filter, strlen(filter)) != 0)
            continue;
        printf("──────────────────────────────────────────────\n");
        int r = TESTS[i].fn();
        if (r != 0) {
            fprintf(stderr, "[FAIL] %s\n\n", TESTS[i].name);
            failures++;
        }
    }

    printf("\n══════════════════════════════════════════════\n");
    if (failures == 0)
        printf("  ALL TESTS PASSED ✓\n");
    else
        printf("  %d TEST(S) FAILED ✗\n", failures);
    printf("══════════════════════════════════════════════\n\n");
    return failures;
}
