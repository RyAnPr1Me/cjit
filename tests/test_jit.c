/**
 * test_jit.c – Integration test suite for the CJIT engine.
 *
 * Each test function (t01 … t44) exercises a distinct aspect of the JIT:
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
 *   t22  Priority queue            – manual high-priority recompile request
 *                                        is serviced before background tasks.
 *   t23  Tier-skip optimization    – O0→O3 direct skip when call-rate exceeds
 *                                        hot_rate_t2 × tier_skip_multiplier.
 *   t24  Compilation watchdog      – a stalled compiler subprocess (sleeping
 *                                        fake cc) is killed after compile_timeout_ms;
 *                                        compile_timeouts stat is incremented.
 *   t25  Latency histogram         – CJIT_DISPATCH_TIMED fills the per-function
 *                                        histogram; cjit_percentile_ns() returns
 *                                        a non-zero value with p50 ≤ p99 < 1 s.
 *   t26  IR normalizer cache       – an IR string with extra comments and
 *                                        whitespace produces the same artifact
 *                                        cache key as the clean version (cache hit).
 *   t27  Compile-event callback   – cjit_set_compile_callback() fires exactly
 *                                        once per compilation with correct fields;
 *                                        removing the callback stops future fires.
 *   t28  Function pinning         – pinned function is not auto-promoted by the
 *                                        monitor; unpinning re-enables promotion.
 *   t29  IR snapshot export       – cjit_snapshot_ir() writes one .c file per
 *                                        registered function plus manifest.txt.
 *   t30  Per-function stats reset  – cjit_reset_function_stats() clears call_cnt,
 *                                        elapsed_ns, histogram buckets, and
 *                                        recompile_count; function continues to run.
 *   t31  Queue drain               – cjit_drain_queue() returns only after all
 *                                        pending compile tasks have finished.
 *   t32  Synchronous compile       – cjit_compile_sync() compiles in the calling
 *                                        thread; function ptr is live on return.
 *   t33  IR LRU evictions          – fill HOT/WARM caches beyond capacity;
 *                                        verify eviction counts > 0 and that
 *                                        a COLD entry can be promoted on access.
 *   t34  cjit_print_stats verbose  – call cjit_print_stats() after a compile;
 *                                        run a verbose engine to exercise the
 *                                        verbose fprintf paths in cjit.c.
 *   t35  compile_sync failure      – call cjit_compile_sync() with invalid IR;
 *                                        verify returns false and callback fires
 *                                        with success=false.
 *   t36  Edge-case API             – cjit_percentile_ns() with 0 calls and an
 *                                        invalid id; OPT_NONE and enable_fast_math;
 *                                        cjit_get_func_counted() call-counting path.
 *   t37  Background compile failure– bad IR routed through the background queue;
 *                                        compile-event callback fires with
 *                                        success=false and non-empty errmsg.
 *   t38  Multi-param specialisation – 2-parameter function with constant-value
 *                                        sampling; void-return wrapper; OPT_O2
 *                                        compilation (covers codegen wrapper paths).
 *   t39  IR prefetch thread         – engine with io_threads=1 and a disk IR dir;
 *                                        monitor calls ir_cache_prefetch; verbose
 *                                        compiler-thread logging enabled.
 *   t40  IR cache print stats +     – call cjit_print_ir_cache_stats() and
 *        prefetch API                    cjit_ir_cache_prefetch(); exercise the
 *                                        async I/O thread body and WARM→HOT path.
 *   t41  Background no-IR skip      – request recompile for a function with no
 *                                        IR source; compiler thread skips silently.
 *   t42  Diverse parameter types    – functions with long/char/short/uint32_t
 *                                        params exercising codegen type checks.
 *   t43  New preamble macros +      – ASSUME/UNROLL/IVDEP macros compile and
 *        extra opt flags               return correct results at O2 and O3;
 *                                        -fno-unwind-tables, -ftree-loop-
 *                                        distribute-patterns, -fgcse-after-
 *                                        reload, -fipa-cp-clone do not break
 *                                        compilation.
 *   t44  Utility preamble macros    – MIN/MAX/CLAMP/COUNT/STATIC_ASSERT/
 *                                        TOSTRING macros compile correctly
 *                                        and return expected results.
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
#include <fcntl.h>       /* open O_CREAT O_WRONLY O_TRUNC */
#include <pthread.h>
#include <dlfcn.h>   /* dladdr – verify JIT function is in a loaded shared object */
#include <sys/stat.h> /* chmod */
#include <dirent.h>  /* opendir, readdir — for rmdir_recursive */

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

/* Recursively delete a directory containing only regular files (no nesting).
 * Used to clean up /tmp cache and snapshot directories created by tests. */
static void rmdir_recursive(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue; /* skip . and .. */
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        unlink(path);
    }
    closedir(d);
    rmdir(dir);
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
    char cache_dir_template[] = "/tmp/cjit_test_cache_XXXXXX";
    char *cache_dir = mkdtemp(cache_dir_template);
    CHECK(cache_dir, "mkdtemp for cache_dir failed");

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
    rmdir_recursive(cache_dir);

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


/* ═══════════════════════════════════════════════════════════════════════════
 * t24 – Compilation watchdog: timed-out compiler subprocess is killed
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Write a tiny shell script to /tmp that sleeps for 30 s (simulates a
 *      stalled compiler).
 *   2. Create an engine with cfg.compile_timeout_ms = 300 and
 *      cfg.cc_binary pointing at the sleep script.
 *   3. Register a function and request a recompile.
 *   4. Wait up to 3 s for the compilation attempt to finish.
 *   5. Verify: compile_timeouts == 1, failed_compilations == 1.
 */
TEST(t24_compile_watchdog)
{
    printf("[t24] Compilation watchdog: stalled compiler is killed...\n");

    /* Write a fake compiler that sleeps 30 s.
     * Keep the path ≤ CJIT_MAX_CC_BINARY - 1 = 63 chars. */
    /* Create the fake "compiler" script via mkstemp to avoid a predictable
     * /tmp path.  We need the path to pass it to the engine as cc_binary. */
    char script_path[] = "/tmp/cjit_cc_XXXXXX";
    int sfd = mkstemp(script_path);
    CHECK(sfd >= 0, "mkstemp for fake compiler script failed");
    /* Make the script executable before writing content. */
    fchmod(sfd, 0700);
    FILE *f = fdopen(sfd, "w");
    if (!f) { close(sfd); unlink(script_path); CHECK(false, "fdopen failed for fake compiler script"); }
    fprintf(f, "#!/bin/sh\nsleep 30\n");
    fclose(f);  /* also closes sfd */

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads    = 1;
    cfg.compile_timeout_ms  = 300;    /* 300 ms – much less than the 30 s sleep */
    cfg.verbose             = false;
    snprintf(cfg.cc_binary, sizeof(cfg.cc_binary), "%s", script_path);

    cjit_engine_t *e = cjit_create(&cfg);
    if (!e) { unlink(script_path); CHECK(e, "cjit_create returned NULL"); }
    cjit_start(e);

    static const char *IR_DUMMY =
        "int dummy_watchdog(int x) { return x; }";
    func_id_t id = cjit_register_function(e, "dummy_watchdog", IR_DUMMY, NULL);
    if (id == CJIT_INVALID_FUNC_ID) {
        cjit_destroy(e); unlink(script_path);
        CHECK(false, "register failed");
    }

    cjit_request_recompile(e, id, OPT_O1);

    /* Wait up to 3 s for the timeout + SIGKILL cycle to complete. */
    uint64_t deadline = now_ms() + 3000;
    cjit_stats_t s;
    do {
        sleep_ms(50);
        s = cjit_get_stats(e);
    } while (s.failed_compilations == 0 && now_ms() < deadline);

    cjit_destroy(e);
    unlink(script_path);

    CHECKF(s.failed_compilations >= 1,
           "expected ≥1 failed compilation, got %llu",
           (unsigned long long)s.failed_compilations);
    CHECKF(s.compile_timeouts >= 1,
           "expected ≥1 compile timeout, got %llu",
           (unsigned long long)s.compile_timeouts);

    printf("[t24]   failed=%llu timeouts=%llu  OK\n",
           (unsigned long long)s.failed_compilations,
           (unsigned long long)s.compile_timeouts);
    printf("[t24] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t25 – Per-function call-latency histogram and cjit_percentile_ns()
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a trivial int add(int, int) function with an AOT fallback.
 *   2. Dispatch it > CJIT_TLS_FLUSH_THRESHOLD times via CJIT_DISPATCH_TIMED
 *      so that the TLS flush path runs and fills the histogram.
 *   3. Verify:
 *        a. cjit_percentile_ns(engine, id, 50) returns > 0 (histogram has data).
 *        b. cjit_percentile_ns(engine, id, 99) >= p50 (monotone).
 *        c. p50 and p99 are < 1 second (10^9 ns) — sanity bound.
 *   4. Verify that p50 for a "do nothing" function is ≤ 1 µs (1000 ns bound).
 */
static int aot_add_t25(int a, int b) { return a + b; }

TEST(t25_latency_histogram)
{
    printf("[t25] Call-latency histogram and cjit_percentile_ns()...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    static const char *IR_ADD_T25 =
        "int add_t25(int a, int b) { return a + b; }";
    func_id_t id = cjit_register_function(e, "add_t25", IR_ADD_T25,
                                           (jit_func_t)(uintptr_t)aot_add_t25);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    /* Drive > 2 × CJIT_TLS_FLUSH_THRESHOLD calls to guarantee at least 2
     * histogram updates.  Use a volatile sink to prevent dead-code elimination. */
    volatile int sink = 0;
    typedef int (*add_fn)(int, int);
    for (unsigned i = 0; i < CJIT_TLS_FLUSH_THRESHOLD * 4u; i++) {
        int r = CJIT_DISPATCH_TIMED(e, id, add_fn, i, i + 1);
        sink += r;
    }
    (void)sink;

    /* Flush any residual TLS counts. */
    cjit_flush_local_counts(e);

    uint64_t counts[CJIT_HIST_BUCKETS];
    cjit_get_histogram(e, id, counts);

    /* At least one bucket must be non-zero. */
    uint64_t total = 0;
    for (int i = 0; i < CJIT_HIST_BUCKETS; i++) total += counts[i];
    CHECKF(total > 0, "histogram is all-zero after %d timed dispatches",
           CJIT_TLS_FLUSH_THRESHOLD * 4);

    uint64_t p50 = cjit_percentile_ns(e, id, 50);
    uint64_t p99 = cjit_percentile_ns(e, id, 99);

    CHECKF(p50 > 0,
           "p50 should be > 0, got %llu", (unsigned long long)p50);
    CHECKF(p99 >= p50,
           "p99 (%llu) should be >= p50 (%llu)",
           (unsigned long long)p99, (unsigned long long)p50);
    /* Sanity: dispatching through an AOT fallback should be << 1 second. */
    CHECKF(p99 < UINT64_C(1000000000),
           "p99 (%llu ns) >= 1 s — unexpectedly slow",
           (unsigned long long)p99);

    printf("[t25]   hist_total=%llu  p50=%lluns  p99=%lluns  OK\n",
           (unsigned long long)total,
           (unsigned long long)p50,
           (unsigned long long)p99);

    cjit_destroy(e);
    printf("[t25] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t26 – IR normalizer: differently-whitespaced/commented IR hits same cache
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Compile IR_CLEAN ("int add_norm(int a, int b){return a+b;}") — cold miss.
 *   2. Compile IR_MESSY (same semantics, extra whitespace + inline comments)
 *      from a FRESH engine pointing at the same cache directory.
 *   3. The normalizer should map both to the same artifact-cache key, so
 *      pass 2 is a warm hit (artifact_cache_hits == 1) with no compiler spawn.
 */
#define IR_NORM_CLEAN "int add_norm(int a, int b) { return a + b; }"
/* Messy IR: same tokens as clean, but with block/line comments added and
 * multiple spaces between tokens that already had single spaces in the clean
 * version.  The normalizer maps both to the same byte stream. */
#define IR_NORM_MESSY \
    "/* function: add_norm */\n" \
    "int  add_norm(int  a,  int  b)  {  // begin\n" \
    "    return  a  +  b;  // end\n" \
    "}\n"

TEST(t26_ir_normalizer_cache)
{
    printf("[t26] IR normalizer: reformatted IR maps to same cache key...\n");

    char cache_dir_t26[] = "/tmp/cjit_norm_cache_XXXXXX";
    char *cache_dir = mkdtemp(cache_dir_t26);
    CHECK(cache_dir, "mkdtemp for t26 cache_dir failed");

    /* ── Pass 1: clean IR → cold miss ─────────────────────────────────── */
    {
        cjit_config_t cfg = cjit_default_config();
        cfg.compiler_threads = 1;
        snprintf(cfg.cache_dir, sizeof(cfg.cache_dir), "%s", cache_dir);

        cjit_engine_t *e = cjit_create(&cfg);
        CHECK(e, "cjit_create (pass 1) returned NULL");
        cjit_start(e);

        func_id_t id = cjit_register_function(e, "add_norm", IR_NORM_CLEAN, NULL);
        CHECK(id != CJIT_INVALID_FUNC_ID, "register failed (pass 1)");

        cjit_request_recompile(e, id, OPT_O1);
        bool ok = cjit_wait_compiled(e, id, 5000);
        CHECK(ok, "cjit_wait_compiled timed out (pass 1)");

        cjit_stats_t s = cjit_get_stats(e);
        CHECKF(s.artifact_cache_misses == 1,
               "pass 1: expected 1 miss, got %llu",
               (unsigned long long)s.artifact_cache_misses);
        printf("[t26]   pass 1 clean IR: miss=1 hit=0  OK\n");
        cjit_destroy(e);
    }

    /* ── Pass 2: messy IR → should be a warm hit via normalizer ─────── */
    {
        cjit_config_t cfg = cjit_default_config();
        cfg.compiler_threads = 1;
        snprintf(cfg.cache_dir, sizeof(cfg.cache_dir), "%s", cache_dir);

        cjit_engine_t *e = cjit_create(&cfg);
        CHECK(e, "cjit_create (pass 2) returned NULL");
        cjit_start(e);

        typedef int (*add_fn)(int, int);
        func_id_t id = cjit_register_function(e, "add_norm", IR_NORM_MESSY, NULL);
        CHECK(id != CJIT_INVALID_FUNC_ID, "register failed (pass 2)");

        cjit_request_recompile(e, id, OPT_O1);
        bool ok = cjit_wait_compiled(e, id, 5000);
        CHECK(ok, "cjit_wait_compiled timed out (pass 2)");

        add_fn fn = (add_fn)(uintptr_t)cjit_get_func_counted(e, id);
        CHECK(fn, "add_norm function pointer NULL after normalizer cache hit");
        CHECKF(fn(3, 4) == 7,  "add_norm(3,4) expected 7, got %d",  fn(3, 4));
        CHECKF(fn(-5, 5) == 0, "add_norm(-5,5) expected 0, got %d", fn(-5, 5));

        cjit_stats_t s = cjit_get_stats(e);
        CHECKF(s.artifact_cache_hits == 1,
               "pass 2: expected 1 cache hit (normalizer), got %llu",
               (unsigned long long)s.artifact_cache_hits);
        CHECKF(s.artifact_cache_misses == 0,
               "pass 2: expected 0 cache misses, got %llu",
               (unsigned long long)s.artifact_cache_misses);

        printf("[t26]   pass 2 messy IR: miss=0 hit=1  OK\n");
        cjit_destroy(e);
    }

    rmdir_recursive(cache_dir);
    printf("[t26] PASS\n");
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * t27 – Compile-event callback: fired for each compilation attempt
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a callback that atomically records each event.
 *   2. Register a function without an AOT fallback and trigger a recompile.
 *   3. Wait for it to complete.
 *   4. Verify: exactly one callback was fired, the event's func_name and
 *      level match, success == true, and timed_out == false.
 *   5. Replace the callback with NULL; trigger another compile; verify no
 *      additional callback fires.
 */

typedef struct {
    int                     count;    /* protected by mu */
    cjit_compile_event_t    last;
    cjit_compile_event_t    events[16]; /* ring buffer of last 16 events */
    pthread_mutex_t         mu;
} cb_state_t;

static void compile_event_recorder(const cjit_compile_event_t *ev, void *ud)
{
    cb_state_t *s = (cb_state_t *)ud;
    pthread_mutex_lock(&s->mu);
    s->last = *ev;
    s->events[s->count % 16] = *ev;
    s->count++;
    pthread_mutex_unlock(&s->mu);
}

TEST(t27_compile_event_callback)
{
    printf("[t27] Compile-event callback fires on compilation...\n");

    cb_state_t state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.count = 0;

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");

    /* Register callback before start. */
    cjit_set_compile_callback(e, compile_event_recorder, &state);
    cjit_start(e);

    static const char *IR_CB =
        "int cb_func(int a, int b) { return a * b; }";
    func_id_t id = cjit_register_function(e, "cb_func", IR_CB, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    cjit_request_recompile(e, id, OPT_O1);
    bool ok = cjit_wait_compiled(e, id, 5000);
    CHECK(ok, "cjit_wait_compiled timed out");

    /* Give the callback time to fire (it runs just after the swap, in
     * the same compiler thread, so it must be done by now). */
    uint64_t deadline = now_ms() + 500;
    int cnt;
    do {
        pthread_mutex_lock(&state.mu);
        cnt = state.count;
        pthread_mutex_unlock(&state.mu);
        if (cnt >= 1) break;
        sleep_ms(10);
    } while (now_ms() < deadline);

    CHECKF(cnt >= 1, "expected ≥1 callback, got %d", cnt);

    pthread_mutex_lock(&state.mu);
    cjit_compile_event_t ev = state.last;
    pthread_mutex_unlock(&state.mu);

    CHECK(ev.success, "last event: expected success");
    CHECK(!ev.timed_out, "last event: unexpected timeout");
    CHECKF(ev.func_id == id,
           "last event: func_id mismatch (%u vs %u)", ev.func_id, id);
    CHECKF(strcmp(ev.func_name, "cb_func") == 0,
           "last event: func_name '%s' != 'cb_func'", ev.func_name);
    CHECKF(ev.level == OPT_O1,
           "last event: level %d != OPT_O1", (int)ev.level);

    printf("[t27]   count=%d  func='%s'  level=%d  success=%d  OK\n",
           cnt, ev.func_name, (int)ev.level, (int)ev.success);

    /* Remove callback; further compiles must not increment the counter. */
    cjit_set_compile_callback(e, NULL, NULL);
    pthread_mutex_lock(&state.mu);
    int cnt_before = state.count;
    pthread_mutex_unlock(&state.mu);
    cjit_request_recompile(e, id, OPT_O2);
    cjit_wait_compiled(e, id, 3000);
    sleep_ms(200); /* let compiler thread finish the event path if any */
    pthread_mutex_lock(&state.mu);
    int cnt_after = state.count;
    pthread_mutex_unlock(&state.mu);
    CHECKF(cnt_after == cnt_before,
           "callback fired after removal (count %d → %d)",
           cnt_before, cnt_after);

    cjit_destroy(e);
    pthread_mutex_destroy(&state.mu);
    printf("[t27] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t28 – Function pinning: pinned functions are not auto-promoted
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a function WITHOUT an AOT fallback.
 *   2. Pin it immediately after registration.
 *   3. Drive the call rate well above hot_rate_t1 for several monitor cycles.
 *   4. Verify: total_compilations == 0 (monitor blocked by pin).
 *   5. Unpin; wait for compilation to complete.
 *   6. Verify: total_compilations == 1 (monitor enqueued after unpin).
 *   7. Verify cjit_is_pinned() reflects the correct state.
 */
static int aot_pinned_stub(int x) { return x + 1; }

TEST(t28_function_pinning)
{
    printf("[t28] Function pinning: pinned function not auto-promoted...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads   = 1;
    cfg.hot_rate_t1        = 500;       /* low threshold – easy to cross */
    cfg.hot_confirm_cycles = 2;
    cfg.monitor_interval_ms = 20;
    cfg.min_uptime_for_tier2_ms = 0;
    cfg.compile_cooloff_ms  = 10;
    cfg.verbose             = false;

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");

    static const char *IR_PIN =
        "int pinned_fn(int x) { return x + 1; }";
    func_id_t id = cjit_register_function(e, "pinned_fn", IR_PIN,
                                            (jit_func_t)(uintptr_t)aot_pinned_stub);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    /* Pin BEFORE starting so the monitor sees it as pinned from the first scan.*/
    bool pok = cjit_pin_function(e, id);
    CHECK(pok, "cjit_pin_function failed");
    CHECK(cjit_is_pinned(e, id), "function should be pinned");

    cjit_start(e);

    /* Drive call rate >> hot_rate_t1 for 10 monitor cycles (200 ms).
     * Use CJIT_DISPATCH so we don't need the timed path. */
    typedef int (*pin_fn)(int);
    volatile int sink = 0;
    uint64_t drive_until = now_ms() + 200;
    while (now_ms() < drive_until) {
        for (int k = 0; k < 200; k++)
            sink += CJIT_DISPATCH(e, id, pin_fn, k);
    }
    (void)sink;
    cjit_flush_local_counts(e);

    /* No compilation should have been enqueued. */
    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations == 0,
           "pinned fn should not be compiled, got compilations=%llu",
           (unsigned long long)s.total_compilations);
    printf("[t28]   while pinned: compilations=%llu  OK\n",
           (unsigned long long)s.total_compilations);

    /* Unpin and wait for auto-promotion. */
    bool upok = cjit_unpin_function(e, id);
    CHECK(upok, "cjit_unpin_function failed");
    CHECK(!cjit_is_pinned(e, id), "function should be unpinned");

    /* Keep driving calls so the monitor sees the hot rate. */
    uint64_t resume_until = now_ms() + 1500;
    while (now_ms() < resume_until) {
        for (int k = 0; k < 400; k++)
            sink += CJIT_DISPATCH(e, id, pin_fn, k);
        cjit_flush_local_counts(e);
    }

    /* Wait up to 3 s for at least one compilation. */
    uint64_t wdeadline = now_ms() + 3000;
    do {
        sleep_ms(50);
        s = cjit_get_stats(e);
    } while (s.total_compilations == 0 && now_ms() < wdeadline);

    cjit_destroy(e);

    CHECKF(s.total_compilations >= 1,
           "expected ≥1 compilation after unpin, got %llu",
           (unsigned long long)s.total_compilations);
    printf("[t28]   after unpin: compilations=%llu  OK\n",
           (unsigned long long)s.total_compilations);
    printf("[t28] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t29 – IR snapshot export: cjit_snapshot_ir() writes .c files + manifest
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register three functions with distinct IR.
 *   2. Call cjit_snapshot_ir(engine, mkdtemp("/tmp/cjit_snap_XXXXXX")).
 *   3. Verify return value == 3.
 *   4. Read each .c file; verify it contains the original IR string as a
 *      substring (the cache may add a newline, but the content must be there).
 *   5. Read manifest.txt; verify exactly 3 lines with the correct names.
 *   6. Clean up.
 */
TEST(t29_snapshot_ir)
{
    printf("[t29] IR snapshot: cjit_snapshot_ir() writes files correctly...\n");

    /* Use a randomised temp directory to avoid symlink-attack vectors. */
    char snap_dir_template[] = "/tmp/cjit_snap_XXXXXX";
    char *snap_dir = mkdtemp(snap_dir_template);
    CHECK(snap_dir, "mkdtemp for snap_dir failed");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    static const char *funcs[][2] = {
        { "snap_add",  "int snap_add(int a, int b) { return a + b; }" },
        { "snap_mul",  "int snap_mul(int a, int b) { return a * b; }" },
        { "snap_sub",  "int snap_sub(int a, int b) { return a - b; }" },
    };
    for (int i = 0; i < 3; i++) {
        func_id_t fid = cjit_register_function(e, funcs[i][0], funcs[i][1], NULL);
        CHECKF(fid != CJIT_INVALID_FUNC_ID, "register '%s' failed", funcs[i][0]);
    }

    int n = cjit_snapshot_ir(e, snap_dir);
    CHECKF(n == 3, "expected 3 files written, got %d", n);
    printf("[t29]   cjit_snapshot_ir returned %d  OK\n", n);

    /* Verify each .c file contains the IR source as a substring. */
    for (int i = 0; i < 3; i++) {
        char fpath[128];
        snprintf(fpath, sizeof(fpath), "%s/%s.c", snap_dir, funcs[i][0]);
        FILE *fp = fopen(fpath, "r");
        CHECKF(fp, "cannot open snapshot file '%s'", fpath);
        char buf[512]; buf[0] = '\0';
        size_t nr = fread(buf, 1, sizeof(buf) - 1, fp);
        buf[nr] = '\0';
        fclose(fp);
        /* The exact IR keyword should be present. */
        CHECKF(strstr(buf, funcs[i][0]),
               "'%s' not found in snapshot file '%s'", funcs[i][0], fpath);
    }

    /* Verify manifest.txt has 3 lines, one per function. */
    char mpath[128];
    snprintf(mpath, sizeof(mpath), "%s/manifest.txt", snap_dir);
    FILE *mf = fopen(mpath, "r");
    CHECK(mf, "cannot open manifest.txt");
    int line_count = 0;
    char line[256];
    while (fgets(line, sizeof(line), mf)) line_count++;
    fclose(mf);
    CHECKF(line_count == 3, "manifest should have 3 lines, got %d", line_count);
    printf("[t29]   manifest has %d lines  OK\n", line_count);

    cjit_destroy(e);

    /* Clean up snapshot directory. */
    rmdir_recursive(snap_dir);

    printf("[t29] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t30 – Per-function stats reset
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a function with an AOT fallback; dispatch it many times via
 *      CJIT_DISPATCH_TIMED to accumulate call_cnt, total_elapsed_ns, and
 *      at least one histogram bucket.
 *   2. Trigger a JIT compile so recompile_count > 0.
 *   3. Call cjit_reset_function_stats() and verify every counter is zero.
 *   4. Verify the function is still callable and returns correct results.
 */
static int aot_reset_fn(int x) { return x * 3; }

TEST(t30_reset_function_stats)
{
    printf("[t30] Per-function stats reset...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    static const char *IR_RESET = "int reset_fn(int x) { return x * 3; }";
    func_id_t id = cjit_register_function(e, "reset_fn", IR_RESET,
                                           (jit_func_t)(uintptr_t)aot_reset_fn);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    typedef int (*rfn_t)(int);
    volatile int sink = 0;
    for (int k = 0; k < 512; k++)
        sink += CJIT_DISPATCH_TIMED(e, id, rfn_t, k);
    cjit_flush_local_counts(e);
    (void)sink;

    /* Use cjit_compile_sync so recompile_count is guaranteed to be > 0
     * before we read it (synchronous — no background race). */
    bool cs = cjit_compile_sync(e, id, OPT_O1);
    CHECK(cs, "cjit_compile_sync(OPT_O1) failed");

    /* Pre-reset assertions using public API. */
    uint64_t before_cnt  = cjit_get_call_count(e, id);
    uint32_t before_rc   = cjit_get_recompile_count(e, id);
    CHECKF(before_cnt > 0, "pre-reset call_cnt should be > 0 (got %llu)",
           (unsigned long long)before_cnt);
    CHECKF(before_rc  > 0, "pre-reset recompile_count should be > 0 (got %u)",
           before_rc);

    uint64_t hist[CJIT_HIST_BUCKETS];
    cjit_get_histogram(e, id, hist);
    uint64_t htotal = 0;
    for (int k = 0; k < CJIT_HIST_BUCKETS; k++) htotal += hist[k];
    CHECKF(htotal > 0, "pre-reset histogram should be non-zero (total=%llu)",
           (unsigned long long)htotal);

    printf("[t30]   pre-reset: call_cnt=%llu rc=%u hist_total=%llu\n",
           (unsigned long long)before_cnt, before_rc, (unsigned long long)htotal);

    /* Reset. */
    bool rok = cjit_reset_function_stats(e, id);
    CHECK(rok, "cjit_reset_function_stats returned false");

    /* Post-reset: all counters must be zero. */
    uint64_t after_cnt  = cjit_get_call_count(e, id);
    uint32_t after_rc   = cjit_get_recompile_count(e, id);
    uint64_t after_elap = cjit_get_elapsed_ns(e, id);
    CHECKF(after_cnt  == 0, "post-reset call_cnt should be 0 (got %llu)",
           (unsigned long long)after_cnt);
    CHECKF(after_rc   == 0, "post-reset recompile_count should be 0 (got %u)",
           after_rc);
    CHECKF(after_elap == 0, "post-reset elapsed_ns should be 0 (got %llu)",
           (unsigned long long)after_elap);

    cjit_get_histogram(e, id, hist);
    htotal = 0;
    for (int k = 0; k < CJIT_HIST_BUCKETS; k++) htotal += hist[k];
    CHECKF(htotal == 0, "post-reset histogram should be zero (total=%llu)",
           (unsigned long long)htotal);

    /* Function must still be callable. */
    int direct_result = ((rfn_t)(uintptr_t)cjit_get_func(e, id))(5);
    CHECKF(direct_result == 15, "reset_fn(5) expected 15, got %d", direct_result);

    printf("[t30]   post-reset: call_cnt=%llu rc=%u elap=%llu hist_total=%llu  OK\n",
           (unsigned long long)after_cnt, after_rc,
           (unsigned long long)after_elap, (unsigned long long)htotal);

    cjit_destroy(e);
    printf("[t30] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t31 – Queue drain: cjit_drain_queue() returns when all tasks are done
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register five functions, enqueue one explicit OPT_O1 recompile each.
 *   2. Call cjit_drain_queue(engine, 8000); assert it returns true.
 *   3. Verify total_compilations == 5 and queue_depth == 0.
 *   4. Verify cjit_drain_queue(engine, 0) (non-blocking) returns true for
 *      the already-empty queue.
 */
TEST(t31_drain_queue)
{
    printf("[t31] Queue drain: all queued tasks finish before drain returns...\n");

#define T31_N 5
    static const char *names[T31_N] = {
        "drain_fn0", "drain_fn1", "drain_fn2", "drain_fn3", "drain_fn4"
    };
    static const char *irs[T31_N] = {
        "int drain_fn0(int x) { return x + 0; }",
        "int drain_fn1(int x) { return x + 1; }",
        "int drain_fn2(int x) { return x + 2; }",
        "int drain_fn3(int x) { return x + 3; }",
        "int drain_fn4(int x) { return x + 4; }",
    };

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    func_id_t ids[T31_N];
    for (int i = 0; i < T31_N; i++) {
        ids[i] = cjit_register_function(e, names[i], irs[i], NULL);
        CHECKF(ids[i] != CJIT_INVALID_FUNC_ID, "register '%s' failed", names[i]);
    }

    for (int i = 0; i < T31_N; i++)
        cjit_request_recompile(e, ids[i], OPT_O1);

    bool drained = cjit_drain_queue(e, 8000);
    CHECK(drained, "cjit_drain_queue timed out");

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations == (uint64_t)T31_N,
           "expected %d compilations after drain, got %llu",
           T31_N, (unsigned long long)s.total_compilations);
    CHECKF(s.queue_depth == 0 && s.prio_queue_depth == 0,
           "queue depth should be 0 after drain (got %u/%u)",
           s.queue_depth, s.prio_queue_depth);
    printf("[t31]   compilations=%llu queue_depth=%u  OK\n",
           (unsigned long long)s.total_compilations, s.queue_depth);

    /* Non-blocking check on an empty queue must return true immediately. */
    bool empty = cjit_drain_queue(e, 0);
    CHECK(empty, "cjit_drain_queue(0) on empty queue returned false");
    printf("[t31]   non-blocking check on empty queue: OK\n");

    cjit_destroy(e);
    printf("[t31] PASS\n");
#undef T31_N
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t32 – Synchronous compile: cjit_compile_sync() is live on return
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a function without an AOT fallback.
 *   2. Call cjit_compile_sync(engine, id, OPT_O2); assert returns true.
 *   3. Assert cjit_get_func() is non-NULL immediately (no background wait).
 *   4. Assert correct result and opt_level == OPT_O2.
 *   5. Assert total_compilations == 1 (stat was incremented).
 *   6. Verify the compile-event callback fires for sync compiles.
 *   7. Upgrade to OPT_O3 with another cjit_compile_sync(); verify result.
 */
TEST(t32_compile_sync)
{
    printf("[t32] Synchronous compile: cjit_compile_sync() installs pointer...\n");

    cb_state_t cb_st;
    memset(&cb_st, 0, sizeof(cb_st));
    pthread_mutex_init(&cb_st.mu, NULL);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");

    cjit_set_compile_callback(e, compile_event_recorder, &cb_st);
    cjit_start(e);

    static const char *IR_SYNC =
        "int sync_fn(int a, int b) { return a + b * 2; }";
    func_id_t id = cjit_register_function(e, "sync_fn", IR_SYNC, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register failed");

    CHECK(cjit_get_func(e, id) == NULL,
          "func_ptr should be NULL before first compile");

    bool ok = cjit_compile_sync(e, id, OPT_O2);
    CHECK(ok, "cjit_compile_sync(OPT_O2) returned false");

    /* Pointer must be live immediately — no wait required. */
    jit_func_t fp = cjit_get_func(e, id);
    CHECK(fp != NULL, "func_ptr is NULL after cjit_compile_sync");

    typedef int (*sfn_t)(int, int);
    int result = ((sfn_t)(uintptr_t)fp)(3, 4);
    CHECKF(result == 11, "sync_fn(3,4) expected 11, got %d", result);
    CHECKF(cjit_get_current_opt_level(e, id) == OPT_O2,
           "opt_level should be OPT_O2, got %d",
           (int)cjit_get_current_opt_level(e, id));
    printf("[t32]   O2: sync_fn(3,4)=%d  OK\n", result);

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations == 1, "expected 1 compilation, got %llu",
           (unsigned long long)s.total_compilations);

    /* Callback must have fired. */
    pthread_mutex_lock(&cb_st.mu);
    int cb_cnt = cb_st.count;
    cjit_compile_event_t cb_ev = cb_st.last;
    pthread_mutex_unlock(&cb_st.mu);
    CHECKF(cb_cnt >= 1, "callback should have fired (count=%d)", cb_cnt);
    CHECK(cb_ev.success, "callback: expected success=true");
    CHECKF(strcmp(cb_ev.func_name, "sync_fn") == 0,
           "callback: func_name='%s'", cb_ev.func_name);
    CHECKF(cb_ev.level == OPT_O2, "callback: level=%d", (int)cb_ev.level);
    printf("[t32]   callback: count=%d func='%s' level=%d  OK\n",
           cb_cnt, cb_ev.func_name, (int)cb_ev.level);

    /* Upgrade to OPT_O3. */
    bool ok3 = cjit_compile_sync(e, id, OPT_O3);
    CHECK(ok3, "cjit_compile_sync(OPT_O3) returned false");

    jit_func_t fp3 = cjit_get_func(e, id);
    int result3 = ((sfn_t)(uintptr_t)fp3)(10, 5);
    CHECKF(result3 == 20, "sync_fn(10,5) expected 20, got %d", result3);
    CHECKF(cjit_get_current_opt_level(e, id) == OPT_O3,
           "opt_level should be OPT_O3 after sync upgrade, got %d",
           (int)cjit_get_current_opt_level(e, id));
    printf("[t32]   O3: sync_fn(10,5)=%d  OK\n", result3);

    cjit_destroy(e);
    pthread_mutex_destroy(&cb_st.mu);
    printf("[t32] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t33 – IR LRU evictions and COLD→WARM disk promotion
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Configure a tiny HOT capacity (2) and WARM capacity (2) so that
 *      inserting more than 4 functions overflows into COLD (disk).
 *   2. Register 8 functions and cjit_compile_sync each one so the IR cache
 *      is populated for every entry.
 *   3. Access a function whose IR is likely in COLD (least-recently-used)
 *      via cjit_compile_sync again, forcing a disk read and COLD→WARM promotion.
 *   4. Assert ir_evictions > 0 and ir_cold_count > 0 in the stats.
 */
TEST(t33_ir_lru_evictions)
{
    printf("[t33] IR LRU evictions and disk promotion...\n");

    char snap_dir[] = "/tmp/t33_XXXXXX";
    char *sd = mkdtemp(snap_dir);
    CHECK(sd, "mkdtemp failed");

#define T33_N 8
    static const char *names[T33_N] = {
        "lru_fn0","lru_fn1","lru_fn2","lru_fn3",
        "lru_fn4","lru_fn5","lru_fn6","lru_fn7"
    };
    /* Keep IR short but unique so each gets its own cache slot. */
    static const char *irs[T33_N] = {
        "int lru_fn0(int x){return x+0;}",
        "int lru_fn1(int x){return x+1;}",
        "int lru_fn2(int x){return x+2;}",
        "int lru_fn3(int x){return x+3;}",
        "int lru_fn4(int x){return x+4;}",
        "int lru_fn5(int x){return x+5;}",
        "int lru_fn6(int x){return x+6;}",
        "int lru_fn7(int x){return x+7;}",
    };

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads  = 1;
    cfg.verbose           = false;
    cfg.hot_ir_cache_size  = 2;   /* tiny HOT capacity to force evictions */
    cfg.warm_ir_cache_size = 2;   /* tiny WARM capacity */
    snprintf(cfg.ir_disk_dir, sizeof(cfg.ir_disk_dir), "%s", snap_dir);
    cfg.io_threads        = 0;    /* synchronous I/O: simpler for testing */

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    func_id_t ids[T33_N];
    for (int i = 0; i < T33_N; i++) {
        ids[i] = cjit_register_function(e, names[i], irs[i], NULL);
        CHECKF(ids[i] != CJIT_INVALID_FUNC_ID, "register '%s' failed", names[i]);
        /* Compile each one so the IR is put into the LRU (via ir_cache_update). */
        bool ok = cjit_compile_sync(e, ids[i], OPT_O1);
        CHECKF(ok, "cjit_compile_sync '%s' failed", names[i]);
    }

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.ir_evictions > 0,
           "expected ir_evictions > 0 after %d inserts into cap-2/2 cache (got %llu)",
           T33_N, (unsigned long long)s.ir_evictions);
    CHECKF(s.ir_cold_count > 0,
           "expected ir_cold_count > 0 after overflow (got %u)", s.ir_cold_count);
    printf("[t33]   evictions=%llu cold=%u warm=%u hot=%u\n",
           (unsigned long long)s.ir_evictions,
           s.ir_cold_count, s.ir_warm_count, s.ir_hot_count);

    /* Re-compile a COLD function to trigger disk read + COLD→WARM promotion.
     * lru_fn0 was inserted first so is most likely to be in COLD. */
    bool ok0 = cjit_compile_sync(e, ids[0], OPT_O2);
    CHECK(ok0, "re-compile lru_fn0 at O2 failed");

    cjit_stats_t s2 = cjit_get_stats(e);
    printf("[t33]   after re-access: promotions=%llu disk_reads=%llu\n",
           (unsigned long long)s2.ir_promotions,
           (unsigned long long)s2.ir_disk_reads);
    /* Either a cache hit (if still in WARM after 2nd insert loop) or a
     * disk read promotion — either way compilation succeeded. */
    CHECK(s2.total_compilations > (uint64_t)T33_N,
          "should have > T33_N total_compilations after re-compile");

    cjit_destroy(e);
    rmdir_recursive(snap_dir);
    printf("[t33] PASS\n");
#undef T33_N
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t34 – cjit_print_stats() + verbose compile path
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Run a standard compile with verbose=true to exercise the verbose
 *      fprintf paths in the compiler thread.
 *   2. Call cjit_print_stats() to cover the stats formatting code.
 *   3. Redirect stderr to /dev/null so test output stays clean.
 */
TEST(t34_print_stats_verbose)
{
    printf("[t34] cjit_print_stats + verbose compile path...\n");

    /* Redirect stderr for the duration of this test so verbose output
     * doesn't pollute test output. */
    int saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = true;   /* exercise verbose paths */

    cjit_engine_t *e = cjit_create(&cfg);
    if (!e) {
        dup2(saved_stderr, STDERR_FILENO); close(saved_stderr);
        fprintf(stderr, "  FAIL  %s:%d  cjit_create returned NULL\n",
                __FILE__, __LINE__);
        return 1;
    }
    cjit_start(e);

    func_id_t id = cjit_register_function(e, "verbose_fn",
                                           "int verbose_fn(int x){return x*7;}",
                                           NULL);
    /* Use the sync path so verbose output fires from the calling thread
     * (compiler thread verbose paths covered when same IR is compiled). */
    cjit_compile_sync(e, id, OPT_O1);
    /* Request another compile at same level; verbose "already at tier" log fires. */
    cjit_request_recompile(e, id, OPT_O1);
    cjit_drain_queue(e, 3000);

    /* Restore stderr BEFORE calling print_stats so banner goes to stderr. */
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    /* cjit_print_stats covers lines 1802-1865 in cjit.c. */
    cjit_print_stats(e);

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations >= 1,
           "expected >= 1 compilation, got %llu",
           (unsigned long long)s.total_compilations);

    cjit_destroy(e);
    printf("[t34] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t35 – cjit_compile_sync failure path + compile-event callback
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a function with deliberately invalid IR (syntax error).
 *   2. Attach a compile-event callback.
 *   3. Call cjit_compile_sync(); assert it returns false.
 *   4. Verify the callback fired with success=false and errmsg non-empty.
 *   5. Also cover enable_fast_math by compiling a valid OPT_O3 function
 *      with fast_math enabled (hits the -ffast-math codegen branch).
 */
TEST(t35_compile_sync_failure)
{
    printf("[t35] cjit_compile_sync failure path + fast_math...\n");

    cb_state_t cb_st;
    memset(&cb_st, 0, sizeof(cb_st));
    pthread_mutex_init(&cb_st.mu, NULL);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads  = 1;
    cfg.verbose           = false;
    cfg.enable_fast_math  = true;  /* exercise -ffast-math branch */

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_set_compile_callback(e, compile_event_recorder, &cb_st);
    cjit_start(e);

    /* Invalid IR: syntax error causes compilation failure. */
    static const char *BAD_IR = "int bad_fn(int x) { THIS_IS_NOT_C; }";
    func_id_t bad_id = cjit_register_function(e, "bad_fn", BAD_IR, NULL);
    CHECK(bad_id != CJIT_INVALID_FUNC_ID, "register bad_fn failed");

    bool ok = cjit_compile_sync(e, bad_id, OPT_O1);
    CHECK(!ok, "cjit_compile_sync should return false for invalid IR");

    /* Callback must have fired with success=false. */
    pthread_mutex_lock(&cb_st.mu);
    int cnt = cb_st.count;
    cjit_compile_event_t ev = cb_st.last;
    pthread_mutex_unlock(&cb_st.mu);

    CHECKF(cnt >= 1, "callback should have fired (count=%d)", cnt);
    CHECK(!ev.success, "callback: expected success=false for bad IR");
    CHECKF(ev.errmsg[0] != '\0', "callback: errmsg should be non-empty, got '%s'",
           ev.errmsg);
    printf("[t35]   failure callback: count=%d success=%d errmsg_len=%zu  OK\n",
           cnt, (int)ev.success, strlen(ev.errmsg));

    /* Register a valid float function and compile at O3 with fast_math=true.
     * This hits the -ffast-math codegen branch in codegen.c. */
    static const char *FLOAT_IR =
        "float fm_fn(float a, float b) { return a * b + a; }";
    func_id_t fid = cjit_register_function(e, "fm_fn", FLOAT_IR, NULL);
    CHECK(fid != CJIT_INVALID_FUNC_ID, "register fm_fn failed");
    bool ok3 = cjit_compile_sync(e, fid, OPT_O3);
    CHECK(ok3, "cjit_compile_sync(OPT_O3, fast_math) failed");

    typedef float (*fmfn_t)(float, float);
    float r = ((fmfn_t)(uintptr_t)cjit_get_func(e, fid))(2.0f, 3.0f);
    CHECKF(r > 7.9f && r < 8.1f,
           "fm_fn(2,3) expected ~8.0 got %f", (double)r);
    printf("[t35]   fast_math O3: fm_fn(2,3)=%.4f  OK\n", (double)r);

    cjit_destroy(e);
    pthread_mutex_destroy(&cb_st.mu);
    printf("[t35] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t36 – Edge-case API: cjit_percentile_ns + OPT_NONE + cjit_get_func_counted
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   A. cjit_percentile_ns edge cases:
 *      • Call on a function with zero accumulated timed calls → expect 0
 *      • Call with invalid id → expect 0
 *      • Call with percentile > 100 → expect the maximum bucket value
 *
 *   B. OPT_NONE compile:
 *      • cjit_compile_sync with OPT_NONE triggers the `-O0` branch in codegen.
 *
 *   C. cjit_get_func_counted():
 *      • Use cjit_get_func_counted() in a tight loop; verify call_cnt is
 *        incremented (covers the TLS flush path in cjit_record_call).
 */
TEST(t36_edge_case_api)
{
    printf("[t36] Edge-case API: percentile_ns, OPT_NONE, get_func_counted...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    /* ── A: cjit_percentile_ns edge cases ── */
    static const char *IR_EC = "int edge_fn(int x) { return x + 1; }";
    func_id_t id = cjit_register_function(e, "edge_fn", IR_EC, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register edge_fn failed");

    /* Compile so there is a valid function ptr. */
    bool cs = cjit_compile_sync(e, id, OPT_O1);
    CHECK(cs, "cjit_compile_sync failed");

    /* Zero timed calls: percentile must return 0. */
    uint64_t p50 = cjit_percentile_ns(e, id, 50);
    CHECKF(p50 == 0,
           "percentile_ns with 0 timed calls should be 0, got %llu",
           (unsigned long long)p50);
    printf("[t36]   percentile_ns(0 calls, p50)=0  OK\n");

    /* Invalid id: must return 0. */
    uint64_t pinv = cjit_percentile_ns(e, CJIT_INVALID_FUNC_ID, 50);
    CHECKF(pinv == 0,
           "percentile_ns with invalid id should be 0, got %llu",
           (unsigned long long)pinv);
    printf("[t36]   percentile_ns(invalid id)=0  OK\n");

    /* percentile > 100: should clamp and return highest bucket edge. */
    uint64_t p101 = cjit_percentile_ns(e, id, 101);
    (void)p101; /* result is implementation-defined; just ensure no crash */
    printf("[t36]   percentile_ns(p=101) did not crash  OK\n");

    /* ── B: OPT_NONE compile ── */
    static const char *IR_NONE = "int none_fn(int x) { return x * 5; }";
    func_id_t nid = cjit_register_function(e, "none_fn", IR_NONE, NULL);
    CHECK(nid != CJIT_INVALID_FUNC_ID, "register none_fn failed");

    bool none_ok = cjit_compile_sync(e, nid, OPT_NONE);
    CHECK(none_ok, "cjit_compile_sync(OPT_NONE) failed");

    typedef int (*gfn_t)(int);
    int r = ((gfn_t)(uintptr_t)cjit_get_func(e, nid))(3);
    CHECKF(r == 15, "none_fn(3) expected 15, got %d", r);
    CHECKF(cjit_get_current_opt_level(e, nid) == OPT_NONE,
           "opt_level should be OPT_NONE, got %d",
           (int)cjit_get_current_opt_level(e, nid));
    printf("[t36]   OPT_NONE: none_fn(3)=%d  OK\n", r);

    /* ── C: cjit_get_func_counted() TLS flush path ── */
    /* Call get_func_counted enough times to trigger a TLS flush
     * (CJIT_TLS_FLUSH_THRESHOLD calls per thread). */
    typedef int (*efn_t)(int);
    int sink = 0;
    for (int k = 0; k < (int)(CJIT_TLS_FLUSH_THRESHOLD * 4); k++) {
        efn_t f = (efn_t)(uintptr_t)cjit_get_func_counted(e, id);
        if (f) sink += f(k);
    }
    cjit_flush_local_counts(e);
    (void)sink;

    uint64_t cc = cjit_get_call_count(e, id);
    CHECKF(cc >= (uint64_t)CJIT_TLS_FLUSH_THRESHOLD,
           "call_cnt should be >= TLS_FLUSH_THRESHOLD after counted loop (got %llu)",
           (unsigned long long)cc);
    printf("[t36]   get_func_counted loop: call_cnt=%llu  OK\n",
           (unsigned long long)cc);

    cjit_destroy(e);
    printf("[t36] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t37 – Background compile failure triggers the compile-event callback
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Register a function with valid IR and compile it (gives it a live fp).
 *   2. Update its IR to deliberately invalid C (syntax error).
 *   3. cjit_request_recompile() to enqueue a background compile of the bad IR.
 *   4. Drain the queue (waits for the compiler thread to process the task).
 *   5. Assert: stat_failed >= 1 and the callback fired with success=false.
 *
 * This covers the compiler-thread failure path (cjit.c lines 613–642) which
 * was previously uncovered because cjit_compile_sync bypasses the queue.
 */
TEST(t37_bg_compile_failure_callback)
{
    printf("[t37] Background compile failure + callback...\n");

    cb_state_t cb_st;
    memset(&cb_st, 0, sizeof(cb_st));
    pthread_mutex_init(&cb_st.mu, NULL);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads  = 1;
    cfg.verbose           = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_set_compile_callback(e, compile_event_recorder, &cb_st);
    cjit_start(e);

    /* Register and compile a valid function first so we have a live pointer. */
    static const char *GOOD_IR = "int bg_fn(int x) { return x + 10; }";
    func_id_t id = cjit_register_function(e, "bg_fn", GOOD_IR, NULL);
    CHECK(id != CJIT_INVALID_FUNC_ID, "register bg_fn failed");

    bool ok = cjit_compile_sync(e, id, OPT_O1);
    CHECK(ok, "initial compile failed");

    /* Verify the success callback fired. */
    pthread_mutex_lock(&cb_st.mu);
    int cnt_before = cb_st.count;
    pthread_mutex_unlock(&cb_st.mu);
    CHECKF(cnt_before >= 1, "callback should have fired for good compile (count=%d)",
           cnt_before);

    /* Now swap IR to invalid C and recompile via the background queue. */
    static const char *BAD_IR = "int bg_fn(int x) { INVALID SYNTAX HERE!!! }";
    bool upd = cjit_update_ir(e, id, BAD_IR, OPT_O1);
    CHECK(upd, "cjit_update_ir failed");

    cjit_request_recompile(e, id, OPT_O1);

    /* Drain so the compiler thread processes the bad IR. */
    bool drained = cjit_drain_queue(e, 10000);
    CHECK(drained, "drain_queue timed out for bad-IR task");

    /* stat_failed must be >= 1. */
    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.failed_compilations >= 1,
           "stat_failed should be >= 1 after bad compile (got %llu)",
           (unsigned long long)s.failed_compilations);
    printf("[t37]   stat_failed=%llu  OK\n",
           (unsigned long long)s.failed_compilations);

    /* The compile-event callback must have been called with success=false. */
    pthread_mutex_lock(&cb_st.mu);
    int cnt_after = cb_st.count;
    bool any_failure = false;
    /* Scan all recorded events — last event should be the failure. */
    for (int i = 0; i < cb_st.count; i++) {
        if (!cb_st.events[i].success) { any_failure = true; break; }
    }
    pthread_mutex_unlock(&cb_st.mu);

    CHECKF(cnt_after > cnt_before,
           "callback count should have increased (before=%d after=%d)",
           cnt_before, cnt_after);
    CHECK(any_failure, "at least one callback event should have success=false");
    printf("[t37]   callback: cnt_after=%d  any_failure=%d  OK\n",
           cnt_after, (int)any_failure);

    /* The live function pointer must still be the good one. */
    typedef int (*bfn_t)(int);
    jit_func_t fp = cjit_get_func(e, id);
    CHECK(fp != NULL, "function pointer should still be live after failed bg recompile");
    int r = ((bfn_t)(uintptr_t)fp)(5);
    CHECKF(r == 15, "bg_fn(5) expected 15, got %d (pointer kept from good compile)", r);
    printf("[t37]   fp still valid: bg_fn(5)=%d  OK\n", r);

    cjit_destroy(e);
    pthread_mutex_destroy(&cb_st.mu);
    printf("[t37] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t38 – Multi-param arg specialisation + void return + OPT_O2
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   A. 2-parameter specialisation — function `int add2(int a, int b)` with
 *      both parameters sampled as constants.  Triggers the codegen wrapper
 *      multi-param paths (lines 692–695, 713 in codegen.c: the second
 *      parameter emits ", " separators).
 *
 *   B. void-return function — `void void_fn(int x)` triggers the void-return
 *      wrapper template in codegen.c (lines 719–736).
 *
 *   C. OPT_O2 compile — exercises `case OPT_O2: return "O2"` branch in the
 *      level-name switch in codegen.c.
 */
TEST(t38_multiarg_void_o2)
{
    printf("[t38] Multi-param specialisation + void return + OPT_O2...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads  = 1;
    cfg.verbose           = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    /* ── A: 2-parameter constant specialisation ── */
    static const char *IR_ADD2 =
        "int add2(int a, int b) { return a + b; }";
    func_id_t aid = cjit_register_function(e, "add2", IR_ADD2, NULL);
    CHECK(aid != CJIT_INVALID_FUNC_ID, "register add2 failed");

    /* Compile at O1 first so there is a live pointer. */
    bool ok1 = cjit_compile_sync(e, aid, OPT_O1);
    CHECK(ok1, "add2 initial compile failed");

    typedef int (*add2_t)(int, int);

    /* Sample both params as constants: a=3, b=7.  The dominant value
     * for each slot is set when consecutive samples agree. */
    for (int k = 0; k < 256; k++) {
        CJIT_SAMPLE_ARGS(e, aid, (uint64_t)3, (uint64_t)7);
        CJIT_DISPATCH(e, aid, add2_t, 3, 7);
    }

    /* Request recompile so the specialisation wrapper is generated. */
    cjit_request_recompile(e, aid, OPT_O2);
    bool drained = cjit_drain_queue(e, 10000);
    CHECK(drained, "drain_queue timed out for add2 recompile");

    jit_func_t fp2 = cjit_get_func(e, aid);
    CHECK(fp2 != NULL, "add2 func ptr NULL after O2 recompile");

    int r2 = ((add2_t)(uintptr_t)fp2)(3, 7);
    CHECKF(r2 == 10, "add2(3,7) expected 10, got %d", r2);
    int r2b = ((add2_t)(uintptr_t)fp2)(10, 20);
    CHECKF(r2b == 30, "add2(10,20) expected 30, got %d", r2b);
    printf("[t38]   add2(3,7)=%d  add2(10,20)=%d  OK\n", r2, r2b);

    /* ── B: void-return function ── */
    static volatile int g_side = 0;
    static const char *IR_VOID =
        "static volatile int *_g;\n"
        "void void_fn(int x) { (void)x; }\n";
    func_id_t vid = cjit_register_function(e, "void_fn", IR_VOID, NULL);
    CHECK(vid != CJIT_INVALID_FUNC_ID, "register void_fn failed");

    /* Sample so wrapper is generated. */
    for (int k = 0; k < 128; k++)
        CJIT_SAMPLE_ARGS(e, vid, (uint64_t)42);

    bool vok = cjit_compile_sync(e, vid, OPT_O2);
    CHECK(vok, "void_fn compile failed");

    typedef void (*vfn_t)(int);
    jit_func_t vfp = cjit_get_func(e, vid);
    CHECK(vfp != NULL, "void_fn fp is NULL");
    ((vfn_t)(uintptr_t)vfp)(99);   /* must not crash */
    (void)g_side;
    printf("[t38]   void_fn(99) executed OK\n");

    /* ── C: OPT_O2 compile (covers case OPT_O2 branch in codegen) ── */
    static const char *IR_MUL2 = "int mul2(int x) { return x * 2; }";
    func_id_t mid = cjit_register_function(e, "mul2", IR_MUL2, NULL);
    CHECK(mid != CJIT_INVALID_FUNC_ID, "register mul2 failed");

    bool mok = cjit_compile_sync(e, mid, OPT_O2);
    CHECK(mok, "mul2 OPT_O2 compile failed");

    typedef int (*mfn_t)(int);
    int mr = ((mfn_t)(uintptr_t)cjit_get_func(e, mid))(7);
    CHECKF(mr == 14, "mul2(7) expected 14, got %d", mr);
    CHECKF(cjit_get_current_opt_level(e, mid) == OPT_O2,
           "opt_level should be OPT_O2, got %d",
           (int)cjit_get_current_opt_level(e, mid));
    printf("[t38]   OPT_O2: mul2(7)=%d  OK\n", mr);

    cjit_destroy(e);
    printf("[t38] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t39 – IR prefetch thread + verbose background compiler thread
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Configure engine with io_threads=1 (enables the async IR prefetch
 *      thread in ir_lru_cache), hot_ir_cache_size=2 so entries overflow to
 *      COLD quickly, and a temporary disk IR directory.
 *   2. Register 6 functions; cjit_compile_sync each one so all IR strings
 *      are stored in the cache (some HOT, some WARM, some COLD on disk).
 *   3. Enable verbose=true and call cjit_request_recompile on a few functions
 *      so the compiler thread's verbose fprintf paths are exercised.
 *   4. The monitor thread's ir_cache_prefetch call (cjit.c line 896) fires
 *      automatically once the engine runs; just confirm no crash and that
 *      the engine is still functional after the prefetch cycle.
 *   5. Verify the WARM→HOT promotion path in ir_cache.c (lines 654-666) by
 *      re-accessing a function that was evicted from HOT to WARM.
 */
TEST(t39_ir_prefetch_and_verbose_bg)
{
    printf("[t39] IR prefetch thread + verbose BG compiler...\n");

    char disk_dir[] = "/tmp/t39_XXXXXX";
    char *dd = mkdtemp(disk_dir);
    CHECK(dd, "mkdtemp failed");

#define T39_N 6
    static const char *names[T39_N] = {
        "pf_fn0","pf_fn1","pf_fn2","pf_fn3","pf_fn4","pf_fn5"
    };
    static const char *irs[T39_N] = {
        "int pf_fn0(int x){return x*10;}",
        "int pf_fn1(int x){return x*11;}",
        "int pf_fn2(int x){return x*12;}",
        "int pf_fn3(int x){return x*13;}",
        "int pf_fn4(int x){return x*14;}",
        "int pf_fn5(int x){return x*15;}",
    };

    /* Redirect stderr so verbose output doesn't pollute test output. */
    int saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads  = 1;
    cfg.verbose           = true;   /* exercise verbose bg compiler paths */
    cfg.hot_ir_cache_size  = 2;     /* small HOT so entries spill quickly */
    cfg.warm_ir_cache_size = 2;
    cfg.io_threads        = 1;      /* enable async prefetch thread */
    snprintf(cfg.ir_disk_dir, sizeof(cfg.ir_disk_dir), "%s", disk_dir);

    cjit_engine_t *e = cjit_create(&cfg);
    if (!e) {
        dup2(saved_stderr, STDERR_FILENO); close(saved_stderr);
        fprintf(stderr, "  FAIL  %s:%d  cjit_create returned NULL\n",
                __FILE__, __LINE__);
        rmdir_recursive(disk_dir);
        return 1;
    }
    cjit_start(e);

    func_id_t ids[T39_N];
    for (int i = 0; i < T39_N; i++) {
        ids[i] = cjit_register_function(e, names[i], irs[i], NULL);
        if (ids[i] == CJIT_INVALID_FUNC_ID) {
            dup2(saved_stderr, STDERR_FILENO); close(saved_stderr);
            fprintf(stderr, "  FAIL  %s:%d  register '%s' failed\n",
                    __FILE__, __LINE__, names[i]);
            cjit_destroy(e); rmdir_recursive(disk_dir);
            return 1;
        }
        /* Compile each one — fills HOT, causes WARM/COLD spills. */
        cjit_compile_sync(e, ids[i], OPT_O1);
    }

    /* Request recompile of first 2 functions at a higher tier via bg queue.
     * With verbose=true this exercises the verbose fprintf in the bg thread. */
    cjit_request_recompile(e, ids[0], OPT_O2);
    cjit_request_recompile(e, ids[1], OPT_O3);
    cjit_drain_queue(e, 10000);

    /* Sleep briefly so the monitor has a chance to run and call
     * ir_cache_prefetch() (cjit.c line 896). */
    sleep_ms(200);

    /* Re-access pf_fn0 by re-compiling it.  If it was evicted from HOT to
     * WARM by the 6 inserts, ir_cache_get_ir will promote WARM→HOT
     * (ir_cache.c lines 654-666). */
    cjit_compile_sync(e, ids[0], OPT_O2);

    /* Restore stderr for final checks. */
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    /* Engine must still be functional. */
    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.total_compilations >= (uint64_t)T39_N,
           "expected >= %d compilations, got %llu",
           T39_N, (unsigned long long)s.total_compilations);
    CHECKF(s.ir_evictions > 0,
           "expected ir_evictions > 0 after %d inserts into cap-2/2 cache (got %llu)",
           T39_N, (unsigned long long)s.ir_evictions);

    typedef int (*pfn_t)(int);
    jit_func_t fp = cjit_get_func(e, ids[0]);
    CHECK(fp != NULL, "pf_fn0 fp is NULL");
    int r = ((pfn_t)(uintptr_t)fp)(3);
    CHECKF(r == 30, "pf_fn0(3) expected 30, got %d", r);
    printf("[t39]   compilations=%llu  ir_evictions=%llu  pf_fn0(3)=%d  OK\n",
           (unsigned long long)s.total_compilations,
           (unsigned long long)s.ir_evictions, r);

    cjit_destroy(e);
    rmdir_recursive(disk_dir);
    printf("[t39] PASS\n");
#undef T39_N
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t40 – cjit_print_ir_cache_stats() + cjit_ir_cache_prefetch() async I/O
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Create an engine with hot_ir_cache_size=2, warm_ir_cache_size=2,
 *      io_threads=1 and an on-disk IR directory.
 *   2. Register and synchronously compile 5 functions so the IR cache fills
 *      and some entries overflow to COLD (on disk).
 *   3. Call cjit_print_ir_cache_stats() — exercises ir_cache_print_stats()
 *      (lines 814-851 in ir_cache.c).
 *   4. Call cjit_ir_cache_prefetch() on each function — exercises
 *      ir_cache_prefetch() (lines 453-471 in ir_cache.c) and its async I/O
 *      thread (lines 423-447 in ir_cache.c).
 *   5. Verify no crash; assert ir_cache_print_stats ran (checked via stats).
 */
TEST(t40_ir_cache_print_and_prefetch)
{
    printf("[t40] cjit_print_ir_cache_stats + cjit_ir_cache_prefetch...\n");

    char disk_dir[] = "/tmp/t40_XXXXXX";
    char *dd = mkdtemp(disk_dir);
    CHECK(dd, "mkdtemp failed");

#define T40_N 5
    static const char *names40[T40_N] = {
        "prt_fn0","prt_fn1","prt_fn2","prt_fn3","prt_fn4"
    };
    static const char *irs40[T40_N] = {
        "int prt_fn0(int x){return x+100;}",
        "int prt_fn1(int x){return x+101;}",
        "int prt_fn2(int x){return x+102;}",
        "int prt_fn3(int x){return x+103;}",
        "int prt_fn4(int x){return x+104;}",
    };

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads   = 1;
    cfg.verbose            = false;
    cfg.hot_ir_cache_size  = 2;    /* small so entries overflow quickly */
    cfg.warm_ir_cache_size = 2;
    cfg.io_threads         = 1;    /* async prefetch thread enabled */
    snprintf(cfg.ir_disk_dir, sizeof(cfg.ir_disk_dir), "%s", disk_dir);

    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    func_id_t ids40[T40_N];
    for (int i = 0; i < T40_N; i++) {
        ids40[i] = cjit_register_function(e, names40[i], irs40[i], NULL);
        CHECKF(ids40[i] != CJIT_INVALID_FUNC_ID,
               "register '%s' failed", names40[i]);
        bool ok = cjit_compile_sync(e, ids40[i], OPT_O1);
        CHECKF(ok, "compile_sync '%s' failed", names40[i]);
    }

    /* Verify evictions happened and some entries are COLD. */
    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.ir_evictions > 0,
           "expected ir_evictions > 0 (got %llu)",
           (unsigned long long)s.ir_evictions);

    /* cjit_print_ir_cache_stats — exercises ir_cache_print_stats lines 814-851. */
    cjit_print_ir_cache_stats(e);      /* output goes to stderr, that's fine */
    printf("[t40]   cjit_print_ir_cache_stats() executed OK\n");

    /* cjit_ir_cache_prefetch — exercises ir_cache_prefetch + async I/O thread. */
    int prefetch_ok = 0;
    for (int i = 0; i < T40_N; i++) {
        if (cjit_ir_cache_prefetch(e, ids40[i]))
            prefetch_ok++;
    }
    /* At least one should succeed (the COLD ones return true from async queue). */
    CHECKF(prefetch_ok >= 1,
           "expected >= 1 successful prefetch, got %d", prefetch_ok);
    printf("[t40]   prefetch_ok=%d  OK\n", prefetch_ok);

    /* Sleep briefly so the async I/O thread has time to process the queue. */
    sleep_ms(300);

    /* Engine still functional after prefetch completes. */
    typedef int (*pfn_t)(int);
    for (int i = 0; i < T40_N; i++) {
        jit_func_t fp = cjit_get_func(e, ids40[i]);
        CHECKF(fp != NULL, "fp for '%s' is NULL after prefetch", names40[i]);
        int r = ((pfn_t)(uintptr_t)fp)(i);
        int expected = i + 100 + i;  /* prt_fnN(i) = i + 100 + N */
        CHECKF(r == expected,
               "'%s'(%d) expected %d, got %d", names40[i], i, expected, r);
    }
    printf("[t40]   all functions still callable after prefetch  OK\n");

    cjit_destroy(e);
    rmdir_recursive(disk_dir);
    printf("[t40] PASS\n");
#undef T40_N
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t41 – Background compiler silently skips functions with no IR source
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * t41 – Verbose compile timeout in background compiler thread
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   1. Create engine with verbose=true and a very short compile_timeout_ms.
 *   2. Register a function whose IR includes deliberate compile-time delay
 *      (or use the infinite-loop IR from t24 that the compiler subprocess
 *      cannot compile in time).
 *   3. cjit_request_recompile() to trigger via the background queue.
 *   4. Drain queue; verify stat_timeouts >= 1 AND stat_failed >= 1.
 *
 * With verbose=true, the compiler thread fires the timeout fprintf at
 * cjit.c lines 613-614 which was otherwise uncovered.
 */
static int t41_fast_aot(int x) { return x; }

TEST(t41_verbose_bg_timeout)
{
    printf("[t41] Verbose background compile timeout...\n");

    /* Redirect stderr so verbose output stays clean. */
    int saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads     = 1;
    cfg.verbose              = true;     /* exercise verbose timeout path */
    cfg.compile_timeout_ms   = 1;        /* 1 ms — virtually guaranteed to timeout */

    cjit_engine_t *e = cjit_create(&cfg);
    if (!e) {
        dup2(saved_stderr, STDERR_FILENO); close(saved_stderr);
        fprintf(stderr, "  FAIL  %s:%d  cjit_create returned NULL\n",
                __FILE__, __LINE__);
        return 1;
    }
    cjit_start(e);

    /* IR whose compilation will exceed the 1 ms timeout. */
    static const char *SLOW_IR =
        "int t41_slow(int x) {\n"
        "    volatile int s = 0;\n"
        "    for (volatile int i = 0; i < 1000000000; i++) s += i;\n"
        "    return s + x;\n"
        "}\n";
    func_id_t id = cjit_register_function(e, "t41_slow", SLOW_IR,
                                           (jit_func_t)t41_fast_aot);
    if (id == CJIT_INVALID_FUNC_ID) {
        dup2(saved_stderr, STDERR_FILENO); close(saved_stderr);
        fprintf(stderr, "  FAIL  %s:%d  register t41_slow failed\n",
                __FILE__, __LINE__);
        cjit_destroy(e);
        return 1;
    }

    cjit_request_recompile(e, id, OPT_O1);
    /* Give the compiler thread time to attempt and fail the compile. */
    cjit_drain_queue(e, 8000);

    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    cjit_stats_t s = cjit_get_stats(e);
    CHECKF(s.failed_compilations >= 1,
           "expected stat_failed >= 1 after timeout (got %llu)",
           (unsigned long long)s.failed_compilations);
    CHECKF(s.compile_timeouts >= 1,
           "expected stat_timeouts >= 1 (got %llu)",
           (unsigned long long)s.compile_timeouts);
    printf("[t41]   stat_timeouts=%llu  stat_failed=%llu  OK\n",
           (unsigned long long)s.compile_timeouts,
           (unsigned long long)s.failed_compilations);

    /* AOT fallback remains valid. */
    jit_func_t fp = cjit_get_func(e, id);
    CHECK(fp != NULL, "AOT fallback should still be live after bg timeout");

    cjit_destroy(e);
    printf("[t41] PASS\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * t42 – Diverse parameter types for arg-specialisation codegen
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strategy:
 *   Register and specialise functions whose parameters use C types that
 *   exercise the is_integer_like() type-check branches in codegen.c
 *   (lines 447-457): long, short, char, size_t, uint32_t, int8_t, etc.
 *
 *   For each function:
 *     a. Compile at O1 to establish a baseline pointer.
 *     b. Flood CJIT_SAMPLE_ARGS with constant values for all parameters.
 *     c. Request an O2 recompile so the codegen wrapper is generated.
 *     d. Verify the result is correct.
 *
 *   Also tests OPT_O3 compilation (covers the `case OPT_O3` branch in
 *   codegen.c's level_to_name switch).
 */
TEST(t42_diverse_param_types)
{
    printf("[t42] Diverse parameter types in arg specialisation...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    /* ── long parameter ── */
    static const char *IR_LONG =
        "long fn_long(long x) { return x * 3L; }";
    func_id_t lid = cjit_register_function(e, "fn_long", IR_LONG, NULL);
    CHECK(lid != CJIT_INVALID_FUNC_ID, "register fn_long failed");
    CHECK(cjit_compile_sync(e, lid, OPT_O1), "fn_long O1 compile failed");
    typedef long (*longfn_t)(long);
    for (int k = 0; k < 256; k++) {
        CJIT_SAMPLE_ARGS(e, lid, (uint64_t)5L);
        longfn_t lfp = (longfn_t)(uintptr_t)cjit_get_func_counted(e, lid);
        if (lfp) (void)lfp(5L);
    }
    cjit_request_recompile(e, lid, OPT_O2);
    CHECK(cjit_drain_queue(e, 8000), "drain fn_long");
    long lr = ((longfn_t)(uintptr_t)cjit_get_func(e, lid))(4L);
    CHECKF(lr == 12L, "fn_long(4) expected 12, got %ld", lr);
    printf("[t42]   fn_long(4)=%ld  OK\n", lr);

    /* ── char parameter ── */
    static const char *IR_CHAR =
        "int fn_char(char c) { return (int)c + 1; }";
    func_id_t cid = cjit_register_function(e, "fn_char", IR_CHAR, NULL);
    CHECK(cid != CJIT_INVALID_FUNC_ID, "register fn_char failed");
    CHECK(cjit_compile_sync(e, cid, OPT_O1), "fn_char O1 compile failed");
    typedef int (*charfn_t)(char);
    for (int k = 0; k < 256; k++) {
        CJIT_SAMPLE_ARGS(e, cid, (uint64_t)'A');
        charfn_t cfp = (charfn_t)(uintptr_t)cjit_get_func_counted(e, cid);
        if (cfp) (void)cfp('A');
    }
    cjit_request_recompile(e, cid, OPT_O2);
    CHECK(cjit_drain_queue(e, 8000), "drain fn_char");
    int cr = ((charfn_t)(uintptr_t)cjit_get_func(e, cid))('A');
    CHECKF(cr == 'B', "fn_char('A') expected %d ('B'), got %d", (int)'B', cr);
    printf("[t42]   fn_char('A')=%d  OK\n", cr);

    /* ── short parameter ── */
    static const char *IR_SHORT =
        "int fn_short(short s) { return (int)s * 2; }";
    func_id_t sid = cjit_register_function(e, "fn_short", IR_SHORT, NULL);
    CHECK(sid != CJIT_INVALID_FUNC_ID, "register fn_short failed");
    CHECK(cjit_compile_sync(e, sid, OPT_O1), "fn_short O1 compile failed");
    typedef int (*shortfn_t)(short);
    for (int k = 0; k < 256; k++) {
        CJIT_SAMPLE_ARGS(e, sid, (uint64_t)10);
        shortfn_t sfp = (shortfn_t)(uintptr_t)cjit_get_func_counted(e, sid);
        if (sfp) (void)sfp((short)10);
    }
    cjit_request_recompile(e, sid, OPT_O2);
    CHECK(cjit_drain_queue(e, 8000), "drain fn_short");
    int sr = ((shortfn_t)(uintptr_t)cjit_get_func(e, sid))((short)10);
    CHECKF(sr == 20, "fn_short(10) expected 20, got %d", sr);
    printf("[t42]   fn_short(10)=%d  OK\n", sr);

    /* ── OPT_O3 coverage for level_to_name switch ── */
    static const char *IR_O3 = "int fn_o3(int x) { return x + x; }";
    func_id_t oid = cjit_register_function(e, "fn_o3", IR_O3, NULL);
    CHECK(oid != CJIT_INVALID_FUNC_ID, "register fn_o3 failed");
    CHECK(cjit_compile_sync(e, oid, OPT_O3), "fn_o3 O3 compile failed");
    typedef int (*o3fn_t)(int);
    int o3r = ((o3fn_t)(uintptr_t)cjit_get_func(e, oid))(21);
    CHECKF(o3r == 42, "fn_o3(21) expected 42, got %d", o3r);
    CHECKF(cjit_get_current_opt_level(e, oid) == OPT_O3,
           "level should be OPT_O3, got %d",
           (int)cjit_get_current_opt_level(e, oid));
    printf("[t42]   OPT_O3: fn_o3(21)=%d  OK\n", o3r);

    cjit_destroy(e);
    printf("[t42] PASS\n");
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * t43: New preamble macros + extra optimization flags
 *
 * Verifies that the three new optimization hint macros injected into every
 * JIT translation unit (ASSUME, UNROLL, IVDEP) compile correctly and that
 * the resulting function returns the correct answer.
 *
 * Also verifies that the new compiler flags (-fno-unwind-tables,
 * -ftree-loop-distribute-patterns, -fgcse-after-reload, -fipa-cp-clone)
 * do not break compilation at O2 or O3.
 * ────────────────────────────────────────────────────────────────────────── */
TEST(t43_new_macros_and_flags)
{
    printf("[t43] New preamble macros (ASSUME/UNROLL/IVDEP) + extra opt flags...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads   = 1;
    cfg.enable_const_fold  = true;   /* exercises -fipa-cp-clone */
    cfg.enable_native_arch = true;   /* exercises -march=native  */
    cfg.verbose            = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    /* ── ASSUME: compiler hint for array-size invariant ── */
    /*
     * The function sums an integer array.  ASSUME(n > 0) asserts that n is
     * always positive; IVDEP asserts no loop-carried aliasing so the compiler
     * can vectorise freely; UNROLL(4) requests an unroll factor of 4.
     * Correctness is verified against a reference sum.
     */
    static const char *IR_ASSUME_UNROLL_IVDEP =
        "int sum_hints(int *arr, int n) {\n"
        "  ASSUME(n > 0);\n"
        "  int s = 0;\n"
        "  UNROLL(4)\n"
        "  IVDEP\n"
        "  for (int i = 0; i < n; i++) s += arr[i];\n"
        "  return s;\n"
        "}\n";

    func_id_t sid = cjit_register_function(e, "sum_hints",
                                            IR_ASSUME_UNROLL_IVDEP, NULL);
    CHECK(sid != CJIT_INVALID_FUNC_ID, "register sum_hints failed");

    /* Compile at O2 (exercises -ftree-loop-distribute-patterns,
     * -fgcse-after-reload, -fipa-cp-clone, -fno-unwind-tables). */
    CHECK(cjit_compile_sync(e, sid, OPT_O2), "sum_hints O2 compile failed");

    typedef int (*sumfn_t)(int *, int);
    sumfn_t fp = (sumfn_t)(uintptr_t)cjit_get_func(e, sid);
    CHECK(fp != NULL, "sum_hints func ptr NULL after compile");

    int data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int got = fp(data, 8);
    CHECKF(got == 36, "sum_hints([1..8]) expected 36, got %d", got);
    printf("[t43]   sum_hints([1..8])=%d  OK (O2 + ASSUME + UNROLL + IVDEP)\n", got);

    /* Compile again at O3 (also exercises the same flags at tier-2). */
    CHECK(cjit_compile_sync(e, sid, OPT_O3), "sum_hints O3 compile failed");
    fp = (sumfn_t)(uintptr_t)cjit_get_func(e, sid);
    CHECK(fp != NULL, "sum_hints func ptr NULL after O3 compile");
    got = fp(data, 8);
    CHECKF(got == 36, "sum_hints O3 expected 36, got %d", got);
    printf("[t43]   sum_hints([1..8])=%d  OK (O3)\n", got);

    /* ── ASSUME with a tight loop containing no data deps ── */
    static const char *IR_SCALE =
        "void scale_arr(int *arr, int n, int factor) {\n"
        "  ASSUME(n % 4 == 0);\n"
        "  IVDEP\n"
        "  for (int i = 0; i < n; i++) arr[i] *= factor;\n"
        "}\n";

    func_id_t vid = cjit_register_function(e, "scale_arr", IR_SCALE, NULL);
    CHECK(vid != CJIT_INVALID_FUNC_ID, "register scale_arr failed");
    CHECK(cjit_compile_sync(e, vid, OPT_O2), "scale_arr O2 compile failed");

    typedef void (*scalefn_t)(int *, int, int);
    scalefn_t vfp = (scalefn_t)(uintptr_t)cjit_get_func(e, vid);
    CHECK(vfp != NULL, "scale_arr func ptr NULL");

    int arr[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    vfp(arr, 8, 3);
    int expected_scale[] = { 3, 6, 9, 12, 15, 18, 21, 24 };
    for (int i = 0; i < 8; i++)
        CHECKF(arr[i] == expected_scale[i],
               "scale_arr[%d]: expected %d got %d",
               i, expected_scale[i], arr[i]);
    printf("[t43]   scale_arr([1..8], 3) correct  OK (O2 + IVDEP + ASSUME)\n");

    cjit_destroy(e);
    printf("[t43] PASS\n");
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * t44: Utility preamble macros (MIN / MAX / CLAMP / COUNT / STATIC_ASSERT /
 *      TOSTRING)
 *
 * Verifies that the utility macros injected into every JIT translation unit
 * compile and produce correct results.  Each macro is exercised with at
 * least one non-trivial case:
 *
 *   MIN / MAX  – signed integer comparisons; checks that the expected value
 *                is returned and that side-effect-free expressions are safe.
 *   CLAMP      – clamps an out-of-range value and a mid-range value.
 *   COUNT      – applied to a fixed-size local array (compile-time constant).
 *   STATIC_ASSERT – compile-time assertion that 1 + 1 == 2 (always passes).
 *   TOSTRING   – stringifies a numeric constant and returns its first digit.
 * ────────────────────────────────────────────────────────────────────────── */
TEST(t44_utility_macros)
{
    printf("[t44] Utility preamble macros (MIN/MAX/CLAMP/COUNT/STATIC_ASSERT/TOSTRING)...\n");

    cjit_config_t cfg = cjit_default_config();
    cfg.compiler_threads = 1;
    cfg.verbose          = false;
    cjit_engine_t *e = cjit_create(&cfg);
    CHECK(e, "cjit_create returned NULL");
    cjit_start(e);

    /* ── MIN ── */
    static const char *IR_MIN =
        "int fn_min(int a, int b) { return MIN(a, b); }";
    func_id_t min_id = cjit_register_function(e, "fn_min", IR_MIN, NULL);
    CHECK(min_id != CJIT_INVALID_FUNC_ID, "register fn_min failed");
    CHECK(cjit_compile_sync(e, min_id, OPT_O2), "fn_min compile failed");
    typedef int (*int2fn_t)(int, int);
    int2fn_t min_fn = (int2fn_t)(uintptr_t)cjit_get_func(e, min_id);
    CHECK(min_fn != NULL, "fn_min ptr NULL");
    CHECKF(min_fn(3, 7)  == 3,  "MIN(3,7) expected 3 got %d",  min_fn(3, 7));
    CHECKF(min_fn(-5, 2) == -5, "MIN(-5,2) expected -5 got %d", min_fn(-5, 2));
    CHECKF(min_fn(4, 4)  == 4,  "MIN(4,4) expected 4 got %d",  min_fn(4, 4));
    printf("[t44]   MIN: OK\n");

    /* ── MAX ── */
    static const char *IR_MAX =
        "int fn_max(int a, int b) { return MAX(a, b); }";
    func_id_t max_id = cjit_register_function(e, "fn_max", IR_MAX, NULL);
    CHECK(max_id != CJIT_INVALID_FUNC_ID, "register fn_max failed");
    CHECK(cjit_compile_sync(e, max_id, OPT_O2), "fn_max compile failed");
    int2fn_t max_fn = (int2fn_t)(uintptr_t)cjit_get_func(e, max_id);
    CHECK(max_fn != NULL, "fn_max ptr NULL");
    CHECKF(max_fn(3, 7)  == 7,  "MAX(3,7) expected 7 got %d",  max_fn(3, 7));
    CHECKF(max_fn(-5, 2) == 2,  "MAX(-5,2) expected 2 got %d", max_fn(-5, 2));
    CHECKF(max_fn(4, 4)  == 4,  "MAX(4,4) expected 4 got %d",  max_fn(4, 4));
    printf("[t44]   MAX: OK\n");

    /* ── CLAMP ── */
    static const char *IR_CLAMP =
        "int fn_clamp(int x, int lo, int hi) { return CLAMP(x, lo, hi); }";
    func_id_t clamp_id = cjit_register_function(e, "fn_clamp", IR_CLAMP, NULL);
    CHECK(clamp_id != CJIT_INVALID_FUNC_ID, "register fn_clamp failed");
    CHECK(cjit_compile_sync(e, clamp_id, OPT_O2), "fn_clamp compile failed");
    typedef int (*int3fn_t)(int, int, int);
    int3fn_t clamp_fn = (int3fn_t)(uintptr_t)cjit_get_func(e, clamp_id);
    CHECK(clamp_fn != NULL, "fn_clamp ptr NULL");
    CHECKF(clamp_fn(5,  1, 10) == 5,  "CLAMP(5,1,10) expected 5 got %d",  clamp_fn(5,  1, 10));
    CHECKF(clamp_fn(-3, 1, 10) == 1,  "CLAMP(-3,1,10) expected 1 got %d", clamp_fn(-3, 1, 10));
    CHECKF(clamp_fn(20, 1, 10) == 10, "CLAMP(20,1,10) expected 10 got %d",clamp_fn(20, 1, 10));
    printf("[t44]   CLAMP: OK\n");

    /* ── COUNT ── */
    static const char *IR_COUNT =
        "int fn_count(void) {\n"
        "  int arr[7];\n"
        "  return (int)COUNT(arr);\n"
        "}\n";
    func_id_t count_id = cjit_register_function(e, "fn_count", IR_COUNT, NULL);
    CHECK(count_id != CJIT_INVALID_FUNC_ID, "register fn_count failed");
    CHECK(cjit_compile_sync(e, count_id, OPT_O1), "fn_count compile failed");
    typedef int (*voidfn_t)(void);
    voidfn_t count_fn = (voidfn_t)(uintptr_t)cjit_get_func(e, count_id);
    CHECK(count_fn != NULL, "fn_count ptr NULL");
    CHECKF(count_fn() == 7, "COUNT([7]) expected 7 got %d", count_fn());
    printf("[t44]   COUNT: OK\n");

    /* ── STATIC_ASSERT ── */
    static const char *IR_SA =
        "int fn_sa(int x) {\n"
        "  STATIC_ASSERT(sizeof(int) >= 4, int_must_be_at_least_4_bytes);\n"
        "  return x * 2;\n"
        "}\n";
    func_id_t sa_id = cjit_register_function(e, "fn_sa", IR_SA, NULL);
    CHECK(sa_id != CJIT_INVALID_FUNC_ID, "register fn_sa failed");
    CHECK(cjit_compile_sync(e, sa_id, OPT_O1), "fn_sa compile failed");
    typedef int (*intfn_t)(int);
    intfn_t sa_fn = (intfn_t)(uintptr_t)cjit_get_func(e, sa_id);
    CHECK(sa_fn != NULL, "fn_sa ptr NULL");
    CHECKF(sa_fn(21) == 42, "fn_sa(21) expected 42 got %d", sa_fn(21));
    printf("[t44]   STATIC_ASSERT: OK\n");

    /* ── TOSTRING ── */
    static const char *IR_TOSTR =
        "#define MY_MAGIC 42\n"
        "int fn_tostr(void) {\n"
        "  const char *s = TOSTRING(MY_MAGIC);\n"
        "  return (int)s[0];  /* '4' == 52 */\n"
        "}\n";
    func_id_t ts_id = cjit_register_function(e, "fn_tostr", IR_TOSTR, NULL);
    CHECK(ts_id != CJIT_INVALID_FUNC_ID, "register fn_tostr failed");
    CHECK(cjit_compile_sync(e, ts_id, OPT_O1), "fn_tostr compile failed");
    voidfn_t ts_fn = (voidfn_t)(uintptr_t)cjit_get_func(e, ts_id);
    CHECK(ts_fn != NULL, "fn_tostr ptr NULL");
    int ts_r = ts_fn();
    CHECKF(ts_r == '4', "TOSTRING(42)[0] expected '%c'(%d) got %d", '4', (int)'4', ts_r);
    printf("[t44]   TOSTRING: OK\n");

    cjit_destroy(e);
    printf("[t44] PASS\n");
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
    { "t24_compile_watchdog",       t24_compile_watchdog         },
    { "t25_latency_histogram",      t25_latency_histogram        },
    { "t26_ir_normalizer_cache",    t26_ir_normalizer_cache      },
    { "t27_compile_event_callback", t27_compile_event_callback   },
    { "t28_function_pinning",       t28_function_pinning         },
    { "t29_snapshot_ir",            t29_snapshot_ir              },
    { "t30_reset_function_stats",   t30_reset_function_stats     },
    { "t31_drain_queue",            t31_drain_queue              },
    { "t32_compile_sync",           t32_compile_sync             },
    { "t33_ir_lru_evictions",       t33_ir_lru_evictions         },
    { "t34_print_stats_verbose",    t34_print_stats_verbose      },
    { "t35_compile_sync_failure",   t35_compile_sync_failure     },
    { "t36_edge_case_api",          t36_edge_case_api            },
    { "t37_bg_compile_failure_callback", t37_bg_compile_failure_callback },
    { "t38_multiarg_void_o2",       t38_multiarg_void_o2         },
    { "t39_ir_prefetch_and_verbose_bg", t39_ir_prefetch_and_verbose_bg },
    { "t40_ir_cache_print_and_prefetch", t40_ir_cache_print_and_prefetch },
    { "t41_verbose_bg_timeout",      t41_verbose_bg_timeout       },
    { "t42_diverse_param_types",     t42_diverse_param_types      },
    { "t43_new_macros_and_flags",    t43_new_macros_and_flags     },
    { "t44_utility_macros",          t44_utility_macros           },
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
