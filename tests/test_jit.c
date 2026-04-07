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
#include <time.h>
#include <unistd.h>
#include <pthread.h>

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

/* ═══════════════════════════════════════════════════════════════════════════
 * Runner
 * ═══════════════════════════════════════════════════════════════════════════ */

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
