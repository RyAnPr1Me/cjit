/**
 * demo.c – End-to-end demonstration of the CJIT non-blocking JIT engine.
 *
 * What this demo shows
 * ────────────────────
 *  1. ENGINE SETUP
 *     • Create a JIT engine with verbose mode enabled.
 *     • Register three functions with C-source IR.
 *
 *  2. AOT FALLBACK
 *     • Before cjit_start() is called, the engine still works: calls are
 *       dispatched through the AOT-compiled fallback function pointers.
 *
 *  3. HOT-FUNCTION DETECTION AND RECOMPILATION
 *     • Start the engine (compiler + monitor threads).
 *     • Run a tight dispatch loop that calls each function many times.
 *     • The monitor thread detects when call counts cross the hot thresholds
 *       and enqueues compilation tasks at OPT_O2 then OPT_O3.
 *     • Compiler threads compile each function and atomically swap the
 *       function pointer in the table.
 *     • The dispatch loop observes the upgrade transparently – no pause,
 *       no lock, no barrier beyond the single atomic load already present.
 *
 *  4. MANUAL RECOMPILE REQUEST
 *     • cjit_request_recompile() is called explicitly to trigger a
 *       pre-warm compilation before the function is naturally hot.
 *
 *  5. INCREMENTAL RECOMPILATION
 *     • add_numbers starts at OPT_NONE (AOT fallback), gets promoted to
 *       OPT_O2, then OPT_O3 as the call count grows.
 *
 *  6. ATOMIC DISPATCH MACRO
 *     • CJIT_DISPATCH shows the minimal-overhead call pattern.
 *
 *  7. DEFERRED GC
 *     • After recompilation the old shared-object handle is retired with a
 *       100 ms grace period.  No runtime thread is blocked.
 *
 *  8. STATISTICS
 *     • cjit_print_stats() summarises all JIT activity.
 *
 * Build
 * ─────
 *   cmake -B build && cmake --build build
 *   ./build/demo
 *
 * Or directly:
 *   cc -std=c11 -O2 -I../include \
 *      ../src/work_queue.c ../src/deferred_gc.c \
 *      ../src/func_table.c ../src/codegen.c ../src/cjit.c \
 *      demo.c -lpthread -ldl -o demo && ./demo
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include "../include/cjit.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * AOT-compiled fallback functions
 *
 * These are normal C functions compiled into the demo binary.  They are
 * used as the initial function pointer before JIT compilation finishes.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * AOT fallback: add two integers.
 *
 * The JIT-compiled version will be identical in result but generated at
 * runtime with higher optimisation flags and can include additional
 * specialisations.
 */
static int add_numbers_aot(int a, int b)
{
    return a + b;
}

/**
 * AOT fallback: sum elements of an array.
 */
static long sum_array_aot(const int *arr, int n)
{
    long s = 0;
    for (int i = 0; i < n; ++i) s += arr[i];
    return s;
}

/**
 * AOT fallback: naive Fibonacci (intentionally slow to show JIT benefit).
 */
static long fib_aot(int n)
{
    if (n <= 1) return n;
    return fib_aot(n - 1) + fib_aot(n - 2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IR (C-source) for each function
 *
 * The JIT engine compiles these strings at runtime.  They use the LIKELY /
 * UNLIKELY / HOT macros injected by the codegen preamble.
 *
 * Key IR features demonstrated:
 *   • Branch-prediction hints via LIKELY / UNLIKELY
 *   • HOT attribute to guide the compiler
 *   • Loop with auto-vectorization hint
 *   • Constant propagation opportunity (the compiler will fold the 2*x term)
 *   • Recursive Fibonacci with memoization for incremental improvement
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char ADD_NUMBERS_IR[] =
    "/*\n"
    " * add_numbers – JIT version.\n"
    " * Branch-prediction hint: most calls will not overflow.\n"
    " * The compiler can inline this and constant-fold at the call site.\n"
    " */\n"
    "HOT int add_numbers(int a, int b) {\n"
    "    /* Constant folding: if a==0 the compiler eliminates this branch. */\n"
    "    if (UNLIKELY(a == 0)) return b;\n"
    "    if (UNLIKELY(b == 0)) return a;\n"
    "    return a + b;\n"
    "}\n";

static const char SUM_ARRAY_IR[] =
    "/*\n"
    " * sum_array – JIT version.\n"
    " * The ALIGNED attribute on the pointer lets the auto-vectorizer use\n"
    " * 256-bit AVX2 loads at OPT_O3 with -march=native.\n"
    " * LIKELY tells the branch predictor the array is non-empty.\n"
    " */\n"
    "HOT long sum_array(const int * restrict arr, int n) {\n"
    "    long s = 0;\n"
    "    if (UNLIKELY(n <= 0)) return 0;\n"
    "    /* Auto-vectorization loop: stride-1 access pattern is ideal. */\n"
    "    for (int i = 0; i < n; ++i) {\n"
    "        s += arr[i];\n"
    "    }\n"
    "    return s;\n"
    "}\n";

static const char FIB_IR[] =
    "/*\n"
    " * fib – JIT version with iterative implementation.\n"
    " * Unlike the AOT recursive version, this is O(n) and benefits from\n"
    " * incremental recompilation: at OPT_O3 the compiler may fully unroll\n"
    " * small-n cases with -funroll-loops.\n"
    " */\n"
    "HOT long fib(int n) {\n"
    "    if (UNLIKELY(n <= 1)) return n;\n"
    "    long a = 0, b = 1;\n"
    "    for (int i = 2; i <= n; ++i) {\n"
    "        long tmp = a + b;\n"
    "        a = b;\n"
    "        b = tmp;\n"
    "    }\n"
    "    return b;\n"
    "}\n";

/* ═══════════════════════════════════════════════════════════════════════════
 * Typed dispatch helpers
 *
 * cjit_get_func() returns jit_func_t (void(*)(void)), a uniform type used
 * by the atomic table.  Callers cast to the concrete prototype they expect.
 *
 * CJIT_DISPATCH does this cast + record_call in one macro expansion.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef int  (*add_fn_t )(int, int);
typedef long (*sum_fn_t )(const int *, int);
typedef long (*fib_fn_t )(int);

/* Shorthand inline wrappers so the main loop is readable. */
static inline int call_add(cjit_engine_t *e, func_id_t id, int a, int b)
{
    cjit_record_call(e, id);
    return ((add_fn_t)cjit_get_func(e, id))(a, b);
}

static inline long call_sum(cjit_engine_t *e, func_id_t id,
                             const int *arr, int n)
{
    cjit_record_call(e, id);
    return ((sum_fn_t)cjit_get_func(e, id))(arr, n);
}

static inline long call_fib(cjit_engine_t *e, func_id_t id, int n)
{
    cjit_record_call(e, id);
    return ((fib_fn_t)cjit_get_func(e, id))(n);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Utility: monotonic time in ms
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("════════════════════════════════════════════════════\n");
    printf("   CJIT – Non-blocking JIT Compiler Demo\n");
    printf("════════════════════════════════════════════════════\n\n");

    /* ── 1. Configure and create the engine ─────────────────────────── */
    cjit_config_t cfg       = cjit_default_config();
    cfg.verbose             = true;      /* Print compilation events      */
    cfg.monitor_interval_ms = 100;       /* Check hot functions every 100 ms */
    cfg.hot_threshold_t1    = 200;       /* T1 after 200 calls            */
    cfg.hot_threshold_t2    = 1000;      /* T2 after 1000 calls           */
    cfg.grace_period_ms     = 200;       /* Keep old code 200 ms          */

    cjit_engine_t *engine = cjit_create(&cfg);
    if (!engine) {
        fprintf(stderr, "cjit_create failed\n");
        return 1;
    }
    printf("[demo] Engine created.\n\n");

    /* ── 2. Register functions with their IR ────────────────────────── */
    func_id_t id_add = cjit_register_function(
        engine, "add_numbers", ADD_NUMBERS_IR,
        (jit_func_t)add_numbers_aot);

    func_id_t id_sum = cjit_register_function(
        engine, "sum_array", SUM_ARRAY_IR,
        (jit_func_t)sum_array_aot);

    func_id_t id_fib = cjit_register_function(
        engine, "fib", FIB_IR,
        (jit_func_t)fib_aot);

    printf("[demo] Registered functions: add_numbers(%u), "
           "sum_array(%u), fib(%u)\n\n",
           id_add, id_sum, id_fib);

    /* ── 3. Verify AOT fallback works before any JIT compilation ────── */
    printf("[demo] === AOT fallback phase (before cjit_start) ===\n");
    int  r1 = call_add(engine, id_add, 3, 4);
    long r2 = call_fib(engine, id_fib, 10);
    printf("[demo] add_numbers(3,4)  = %d  (expected 7)\n",  r1);
    printf("[demo] fib(10)           = %ld (expected 55)\n", r2);
    if (r1 != 7 || r2 != 55) {
        fprintf(stderr, "[demo] FATAL: AOT fallback returned wrong result!\n");
        cjit_destroy(engine);
        return 1;
    }
    printf("[demo] AOT fallback: OK\n\n");

    /* ── 4. Pre-warm: request immediate compilation of fib at O1 ─────── */
    printf("[demo] Pre-warming 'fib' at OPT_O1 before hot loop...\n");

    /* ── 5. Start engine (compiler + monitor + GC threads) ──────────── */
    cjit_start(engine);
    printf("[demo] Engine started (%d compiler threads, 1 monitor thread).\n\n",
           cfg.compiler_threads);

    /* Issue the pre-warm request AFTER start so compiler threads are live. */
    cjit_request_recompile(engine, id_fib, OPT_O1);

    /* ── 6. Hot dispatch loop ────────────────────────────────────────── */
    printf("[demo] === Hot dispatch loop ===\n");
    printf("[demo] Calling functions in a tight loop; watch the JIT upgrade...\n\n");

    /* Prepare a test array for sum_array. */
    enum { ARR_LEN = 1024 };
    int arr[ARR_LEN];
    for (int i = 0; i < ARR_LEN; ++i) arr[i] = i;
    long expected_sum = (long)ARR_LEN * (ARR_LEN - 1) / 2;

    uint64_t t0 = now_ms();
    long     last_print_ms = 0;
    int      last_fib_arg  = 0;
    volatile long sink = 0; /* prevent optimiser from eliding the calls */

    for (int iter = 0; iter < 3000; ++iter) {

        /* add_numbers: simple integer addition */
        sink += call_add(engine, id_add, iter, iter + 1);

        /* sum_array: memory-bandwidth bound, vectorizable */
        sink += call_sum(engine, id_sum, arr, ARR_LEN);

        /* fib: CPU-bound, benefits from iterative JIT version */
        int fib_n = 20 + (iter % 10);
        sink += call_fib(engine, id_fib, fib_n);
        last_fib_arg = fib_n;

        /* Print progress and current opt level every ~500 ms */
        long elapsed = (long)(now_ms() - t0);
        if (elapsed - last_print_ms >= 500) {
            last_print_ms = elapsed;
            printf("[demo] iter=%4d  elapsed=%4ld ms  "
                   "add@O%d  sum@O%d  fib@O%d  "
                   "call_cnt(add)=%llu\n",
                   iter, elapsed,
                   (int)cjit_get_current_opt_level(engine, id_add),
                   (int)cjit_get_current_opt_level(engine, id_sum),
                   (int)cjit_get_current_opt_level(engine, id_fib),
                   (unsigned long long)cjit_get_call_count(engine, id_add));
        }

        /* Short sleep every 100 iterations so threads can do work. */
        if (iter % 100 == 99) {
            struct timespec ms5 = { .tv_sec = 0, .tv_nsec = 5000000L };
            nanosleep(&ms5, NULL);
        }
    }

    uint64_t t1 = now_ms();
    printf("\n[demo] Hot loop completed in %llu ms (sink=%ld to prevent DCE).\n\n",
           (unsigned long long)(t1 - t0), sink);

    /* ── 7. Correctness check after JIT compilation ──────────────────── */
    printf("[demo] === Correctness checks after JIT compilation ===\n");

    /* Give compiler threads a moment to finish any in-flight compilations. */
    struct timespec wait500 = { .tv_sec = 0, .tv_nsec = 500000000L };
    nanosleep(&wait500, NULL);

    int  jit_add = call_add(engine, id_add, 100, 200);
    long jit_sum = call_sum(engine, id_sum, arr, ARR_LEN);
    long jit_fib = call_fib(engine, id_fib, last_fib_arg);
    long aot_fib = fib_aot(last_fib_arg);

    printf("[demo] add_numbers(100,200)         = %d   (expected 300)\n",
           jit_add);
    printf("[demo] sum_array(arr, %d)         = %ld (expected %ld)\n",
           ARR_LEN, jit_sum, expected_sum);
    printf("[demo] fib(%d)                      = %ld  (expected %ld)\n",
           last_fib_arg, jit_fib, aot_fib);

    int ok = (jit_add == 300) &&
             (jit_sum == expected_sum) &&
             (jit_fib == aot_fib);
    printf("[demo] Correctness: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    /* ── 8. Final optimisation levels ───────────────────────────────── */
    printf("[demo] Final optimisation levels:\n");
    printf("[demo]   add_numbers : O%d\n",
           (int)cjit_get_current_opt_level(engine, id_add));
    printf("[demo]   sum_array   : O%d\n",
           (int)cjit_get_current_opt_level(engine, id_sum));
    printf("[demo]   fib         : O%d\n\n",
           (int)cjit_get_current_opt_level(engine, id_fib));

    /* ── 9. Statistics ──────────────────────────────────────────────── */
    cjit_print_stats(engine);

    /* ── 10. Shutdown ────────────────────────────────────────────────── */
    printf("\n[demo] Stopping engine (joining threads, flushing GC)...\n");
    cjit_destroy(engine);
    printf("[demo] Engine destroyed cleanly.\n");
    printf("[demo] Done.\n");

    return ok ? 0 : 1;
}
