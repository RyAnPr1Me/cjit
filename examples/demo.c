/**
 * demo.c – End-to-end demonstration of the CJIT non-blocking JIT engine,
 *           multi-generation LRU IR cache, and memory-pressure awareness.
 *
 * What this demo shows
 * ────────────────────
 *  1. ENGINE SETUP
 *     • Create a JIT engine with a tiny HOT/WARM cache (5/10 slots) so that
 *       LRU eviction and disk spill are visible even with few functions.
 *
 *  2. LRU CACHE UNDER PRESSURE
 *     • Register 20 functions.  With hot_ir_cache_size=5 and
 *       warm_ir_cache_size=10, the remaining 5 functions are immediately
 *       spilled to disk (COLD generation) at registration time.
 *
 *  3. COLD LOAD ON COMPILATION
 *     • When a COLD function is compiled for the first time, the compiler
 *       thread calls ir_cache_get_ir(), which loads its IR from disk and
 *       promotes it to WARM.  A [ir_cache/pressure] log line shows any
 *       live memory-pressure level.
 *
 *  4. HOT-FUNCTION DETECTION AND RECOMPILATION
 *     • Start the engine (compiler + monitor + GC + pressure threads).
 *     • Run a tight dispatch loop; the monitor promotes hot functions to
 *       OPT_O2 then OPT_O3 via atomic pointer swaps.
 *
 *  5. CORRECTNESS CHECK
 *     • Verify all 3 "real" functions produce correct results after JIT.
 *
 *  6. STATISTICS
 *     • Print the full stats table showing IR HOT/WARM/COLD counts, disk
 *       reads/writes, LRU evictions, pressure-driven evictions, and the
 *       current memory pressure level read from /proc/meminfo.
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
 * AOT fallbacks
 * ═══════════════════════════════════════════════════════════════════════════ */

static int  add_numbers_aot(int a, int b)            { return a + b; }
static long sum_array_aot(const int *arr, int n)
{
    long s = 0;
    for (int i = 0; i < n; ++i) s += arr[i];
    return s;
}
static long fib_aot(int n)
{
    if (n <= 1) return n;
    return fib_aot(n - 1) + fib_aot(n - 2);
}
/* Generic "cold" fallback for padding functions */
static int cold_fn_aot(int x) { return x * 2; }

/* ═══════════════════════════════════════════════════════════════════════════
 * IR strings for the 3 primary functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char ADD_NUMBERS_IR[] =
    "HOT int add_numbers(int a, int b) {\n"
    "    if (UNLIKELY(a == 0)) return b;\n"
    "    if (UNLIKELY(b == 0)) return a;\n"
    "    return a + b;\n"
    "}\n";

static const char SUM_ARRAY_IR[] =
    "HOT long sum_array(const int * restrict arr, int n) {\n"
    "    long s = 0;\n"
    "    if (UNLIKELY(n <= 0)) return 0;\n"
    "    for (int i = 0; i < n; ++i) s += arr[i];\n"
    "    return s;\n"
    "}\n";

static const char FIB_IR[] =
    "HOT long fib(int n) {\n"
    "    if (UNLIKELY(n <= 1)) return n;\n"
    "    long a = 0, b = 1;\n"
    "    for (int i = 2; i <= n; ++i) {\n"
    "        long tmp = a + b; a = b; b = tmp;\n"
    "    }\n"
    "    return b;\n"
    "}\n";

/* Template IR for the 17 padding ("cold") functions. */
static char cold_ir_buf[17][256];

static void build_cold_ir(void)
{
    for (int i = 0; i < 17; ++i) {
        snprintf(cold_ir_buf[i], sizeof(cold_ir_buf[i]),
                 "HOT int cold_fn_%02d(int x) {\n"
                 "    return x * %d;\n"
                 "}\n", i, i + 2);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Call helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef int  (*add_fn_t)(int, int);
typedef long (*sum_fn_t)(const int *, int);
typedef long (*fib_fn_t)(int);

static inline int  call_add(cjit_engine_t *e, func_id_t id, int a, int b)
{
    cjit_record_call(e, id);
    return ((add_fn_t)cjit_get_func(e, id))(a, b);
}
static inline long call_sum(cjit_engine_t *e, func_id_t id, const int *arr, int n)
{
    cjit_record_call(e, id);
    return ((sum_fn_t)cjit_get_func(e, id))(arr, n);
}
static inline long call_fib(cjit_engine_t *e, func_id_t id, int n)
{
    cjit_record_call(e, id);
    return ((fib_fn_t)cjit_get_func(e, id))(n);
}

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
    build_cold_ir();

    printf("═══════════════════════════════════════════════════════\n");
    printf("  CJIT Demo: Multi-Gen LRU + Memory-Pressure Awareness\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    /* ── 1. Configure the engine ──────────────────────────────────────── */
    cjit_config_t cfg          = cjit_default_config();
    cfg.verbose                = true;
    cfg.monitor_interval_ms    = 100;
    cfg.hot_threshold_t1       = 200;
    cfg.hot_threshold_t2       = 800;
    cfg.grace_period_ms        = 200;

    /*
     * Deliberately small LRU capacities so we can observe evictions with
     * only 20 registered functions:
     *   HOT  = 5 slots  → functions 6-20 start in WARM or COLD
     *   WARM = 10 slots → functions 16-20 are immediately COLD (disk-only)
     *
     * In a real deployment these would be O(thousands).
     */
    cfg.hot_ir_cache_size      = 5;
    cfg.warm_ir_cache_size     = 10;

    /* Memory-pressure thresholds (real /proc/meminfo values on Linux).
     * Low thresholds so we can observe the pressure level even in CI. */
    cfg.mem_pressure_check_ms     = 200;
    cfg.mem_pressure_low_pct      = 50;  /* MEDIUM when < 50% available */
    cfg.mem_pressure_high_pct     = 20;  /* HIGH   when < 20% available */
    cfg.mem_pressure_critical_pct = 10;  /* CRITICAL when < 10% available */

    cjit_engine_t *engine = cjit_create(&cfg);
    if (!engine) { fprintf(stderr, "cjit_create failed\n"); return 1; }
    printf("[demo] Engine created.\n"
           "[demo]   HOT capacity  = %u\n"
           "[demo]   WARM capacity = %u\n\n",
           cfg.hot_ir_cache_size, cfg.warm_ir_cache_size);

    /* ── 2. Register 3 real + 17 cold padding functions ──────────────── */
    func_id_t id_add = cjit_register_function(
        engine, "add_numbers", ADD_NUMBERS_IR, (jit_func_t)add_numbers_aot);
    func_id_t id_sum = cjit_register_function(
        engine, "sum_array",   SUM_ARRAY_IR,   (jit_func_t)sum_array_aot);
    func_id_t id_fib = cjit_register_function(
        engine, "fib",         FIB_IR,         (jit_func_t)fib_aot);

    char cold_name[32];
    for (int i = 0; i < 17; ++i) {
        snprintf(cold_name, sizeof(cold_name), "cold_fn_%02d", i);
        cjit_register_function(engine, cold_name,
                               cold_ir_buf[i], (jit_func_t)cold_fn_aot);
    }

    printf("[demo] Registered 20 functions.\n");

    /* Snapshot the IR cache state after registration. */
    {
        cjit_stats_t s = cjit_get_stats(engine);
        printf("[demo] IR cache after registration:\n"
               "[demo]   HOT=%u  WARM=%u  COLD=%u (on-disk)\n\n",
               s.ir_hot_count, s.ir_warm_count, s.ir_cold_count);

        if (s.ir_cold_count == 0) {
            printf("[demo] NOTE: all functions fit in memory; increase the\n"
                   "[demo]       number of registrations to observe eviction.\n\n");
        }
    }

    /* ── 3. AOT correctness before JIT ───────────────────────────────── */
    printf("[demo] === AOT fallback phase ===\n");
    int  r1 = call_add(engine, id_add, 3, 4);
    long r2 = call_fib(engine, id_fib, 10);
    printf("[demo]   add_numbers(3,4)=%d (expect 7)   fib(10)=%ld (expect 55)\n",
           r1, r2);
    if (r1 != 7 || r2 != 55) {
        fprintf(stderr, "[demo] FATAL: AOT fallback wrong!\n");
        cjit_destroy(engine); return 1;
    }
    printf("[demo] AOT fallback: OK\n\n");

    /* ── 4. Start engine ──────────────────────────────────────────────── */
    cjit_start(engine);
    printf("[demo] Engine started (%d compiler threads + 1 monitor + "
           "1 pressure thread).\n\n", cfg.compiler_threads);

    /* Pre-warm fib at O1. */
    cjit_request_recompile(engine, id_fib, OPT_O1);

    /* ── 5. Hot dispatch loop ─────────────────────────────────────────── */
    printf("[demo] === Hot dispatch loop ===\n");

    enum { ARR_LEN = 512 };
    int arr[ARR_LEN];
    for (int i = 0; i < ARR_LEN; ++i) arr[i] = i;
    const long expected_sum = (long)ARR_LEN * (ARR_LEN - 1) / 2;

    uint64_t t0           = now_ms();
    long     last_print   = 0;
    volatile long sink    = 0;

    for (int iter = 0; iter < 3000; ++iter) {
        sink += call_add(engine, id_add, iter, iter + 1);
        sink += call_sum(engine, id_sum, arr, ARR_LEN);
        sink += call_fib(engine, id_fib, 20 + (iter % 8));

        long elapsed = (long)(now_ms() - t0);
        if (elapsed - last_print >= 500) {
            last_print = elapsed;
            cjit_stats_t s = cjit_get_stats(engine);
            static const char *const pnames[] =
                { "NORMAL", "MEDIUM", "HIGH", "CRITICAL" };
            printf("[demo] iter=%4d  %4ld ms  "
                   "add@O%d sum@O%d fib@O%d  "
                   "IR(H=%u W=%u C=%u)  mem=%s(%lluMB/%lluMB)\n",
                   iter, elapsed,
                   (int)cjit_get_current_opt_level(engine, id_add),
                   (int)cjit_get_current_opt_level(engine, id_sum),
                   (int)cjit_get_current_opt_level(engine, id_fib),
                   s.ir_hot_count, s.ir_warm_count, s.ir_cold_count,
                   pnames[s.mem_pressure],
                   (unsigned long long)s.mem_available_mb,
                   (unsigned long long)s.mem_total_mb);
        }

        if (iter % 100 == 99) {
            struct timespec ms5 = { .tv_nsec = 5000000L };
            nanosleep(&ms5, NULL);
        }
    }

    printf("\n[demo] Hot loop done in %llu ms (sink=%ld).\n\n",
           (unsigned long long)(now_ms() - t0), sink);

    /* ── 6. Wait for in-flight compilations to settle ────────────────── */
    struct timespec wait = { .tv_nsec = 600000000L };
    nanosleep(&wait, NULL);

    /* ── 7. Correctness after JIT ────────────────────────────────────── */
    printf("[demo] === Post-JIT correctness ===\n");
    int  jit_add = call_add(engine, id_add, 100, 200);
    long jit_sum = call_sum(engine, id_sum, arr, ARR_LEN);
    long jit_fib = call_fib(engine, id_fib, 25);
    long aot_fib = fib_aot(25);

    printf("[demo]   add_numbers(100,200)   = %d  (expect 300)\n",   jit_add);
    printf("[demo]   sum_array(%d)         = %ld (expect %ld)\n",
           ARR_LEN, jit_sum, expected_sum);
    printf("[demo]   fib(25)               = %ld (expect %ld)\n",    jit_fib, aot_fib);

    int ok = (jit_add == 300) && (jit_sum == expected_sum) && (jit_fib == aot_fib);
    printf("[demo] Correctness: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    /* ── 8. Final opt levels ──────────────────────────────────────────── */
    printf("[demo] Final opt levels: add@O%d  sum@O%d  fib@O%d\n\n",
           (int)cjit_get_current_opt_level(engine, id_add),
           (int)cjit_get_current_opt_level(engine, id_sum),
           (int)cjit_get_current_opt_level(engine, id_fib));

    /* ── 9. Full statistics ───────────────────────────────────────────── */
    cjit_print_stats(engine);

    /* ── 10. Shutdown ────────────────────────────────────────────────── */
    printf("\n[demo] Shutting down...\n");
    cjit_destroy(engine);
    printf("[demo] Done.\n");
    return ok ? 0 : 1;
}
