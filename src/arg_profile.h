/**
 * arg_profile.h – Per-function argument-value profiling for the JIT engine.
 *
 * Overview
 * ────────
 * Each registered JIT function optionally tracks which argument VALUES are
 * most frequently passed at call sites.  When the monitor decides to
 * recompile a function at a higher optimization tier, codegen checks for a
 * "confident dominant value" on any integer argument and, if found, wraps the
 * user's function in a thin specialization shell:
 *
 *   1. The user's original function is renamed to an internal symbol via a
 *      preprocessor -D flag (e.g. -Dmy_func=_cjit_i_my_func).
 *
 *   2. A new exported `my_func` wrapper is prepended to the source:
 *
 *        // [JIT-generated specialization for my_func: arg0 always 0]
 *        static int _cjit_i_my_func(int x, int y);  // forward decl
 *        int my_func(int x, int y) {
 *            if (__builtin_expect(x == 0, 1))
 *                return _cjit_i_my_func(0, y);   // constant-folded path
 *            return _cjit_i_my_func(x, y);        // generic fallback
 *        }
 *
 *   3. With -finline-functions (enabled at O2+), the compiler inlines
 *      _cjit_i_my_func(0, y) into the hot path and constant-folds through
 *      it, eliminating dead branches, simplifying loops, etc.
 *
 * Sampling
 * ────────
 * Collection overhead is minimised by piggybacking on the existing
 * CJIT_TLS_FLUSH_THRESHOLD flush cycle: the CJIT_SAMPLE_ARGS macro checks
 * whether cjit_tls_counts[id] is at the flush boundary (the one-in-N sample
 * point) BEFORE cjit_get_func_counted() resets it.  This means sampling
 * happens approximately once per CJIT_TLS_FLUSH_THRESHOLD calls per thread,
 * with no additional counter or TLS variable.  On the common (non-sample)
 * path the overhead is:
 *   • One TLS load (already in L1 cache — shared with cjit_get_func_counted)
 *   • One compare + branch (predicted not-taken: < 1 cycle amortised)
 *
 * Dominant-value tracking
 * ───────────────────────
 * Each argument slot uses a Boyer-Moore majority-vote style algorithm:
 *   • On match (v == dominant_val):  dominant_cnt++
 *   • On mismatch:                   dominant_cnt--; replace if cnt reaches 0
 *   • total_samples always increments
 *
 * This gives an O(1) per-sample update with no heap allocation.  Confidence
 * is computed at specialisation time as:
 *   confidence = dominant_cnt * 100 / total_samples
 * and must exceed CJIT_PROFILE_MIN_CONFIDENCE_PCT with at least
 * CJIT_PROFILE_MIN_SAMPLES total samples before specialisation is attempted.
 *
 * Safety
 * ──────
 * All fields in cjit_arg_profile_t are plain (non-atomic).  The profile is
 * written by calling threads (via cjit_record_arg_samples) and read by the
 * compiler thread (during codegen_compile).  Since the data is statistical
 * and the worst case is a slightly stale snapshot — never a memory-safety
 * violation — no locking is required.  64-bit dominant_val writes are
 * naturally atomic on x86-64 and AArch64 when 8-byte aligned.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

/* ─────────────────────────── constants ────────────────────────────────────── */

/**
 * Maximum number of argument slots profiled per function.
 *
 * Slots beyond this count are silently ignored.  8 is sufficient for virtually
 * all real JIT workloads; reducing it saves memory on the cold cache line.
 */
#define CJIT_MAX_PROFILED_ARGS  8u

/**
 * Minimum number of samples (total_samples) before specialisation is
 * considered.  Guards against specialising on a single observation.
 *
 * At CJIT_TLS_FLUSH_THRESHOLD = 32, each thread contributes ~1 sample per
 * 32 calls; after 256 calls from a single thread total_samples >= 8.
 */
#define CJIT_PROFILE_MIN_SAMPLES  8u

/**
 * Minimum confidence (dominant_cnt * 100 / total_samples) required for
 * specialisation.  75 % means the dominant value was observed in at least
 * 3 of every 4 sample points.
 *
 * Expressed as a percentage in [1, 100].
 */
#define CJIT_PROFILE_MIN_CONFIDENCE_PCT  75u

/**
 * Safety limit: if the function symbol name appears more than this many times
 * in the IR source text (e.g. due to recursive calls or type aliases), skip
 * specialisation to avoid unintended -D macro substitution.
 */
#define CJIT_SPEC_MAX_NAME_OCCURRENCES   8u

/* ─────────────────────────── types ────────────────────────────────────────── */

/**
 * One tracked argument slot.
 *
 * Size: 16 bytes (fits three slots per cache line with 4-byte padding).
 */
typedef struct {
    uint64_t dominant_val;  /**< Current dominant (majority-candidate) value.  */
    uint32_t dominant_cnt;  /**< Boyer-Moore vote count for dominant_val.       */
    uint32_t total_samples; /**< Total sample points observed on this slot.     */
} cjit_arg_slot_t;

/**
 * Full per-function argument profile.
 *
 * Stored on the cold cache line of func_table_entry_t; never touched on the
 * hot dispatch path unless CJIT_SAMPLE_ARGS is used.
 *
 * Size: CJIT_MAX_PROFILED_ARGS × 16 + 8 = 136 bytes (two cache lines).
 */
typedef struct {
    cjit_arg_slot_t slots[CJIT_MAX_PROFILED_ARGS]; /**< Per-arg slots (16 B ea) */
    uint8_t         n_profiled; /**< Number of argument slots being tracked.    */
    uint8_t         _pad[7];
} cjit_arg_profile_t;

/* ─────────────────────────── inline helpers ────────────────────────────────── */

/**
 * Update one argument slot with a new observed value v.
 *
 * Boyer-Moore majority-vote update:
 *   • First sample: adopt v as the dominant value.
 *   • Match:        increment dominant_cnt (saturating).
 *   • Mismatch:     decrement dominant_cnt; if it hits 0, replace with v.
 *   • total_samples always incremented (saturating).
 */
static inline void cjit_arg_slot_update(cjit_arg_slot_t *slot, uint64_t v)
{
    if (slot->total_samples == 0) {
        slot->dominant_val = v;
        slot->dominant_cnt = 1;
    } else if (v == slot->dominant_val) {
        if (slot->dominant_cnt < UINT32_MAX) slot->dominant_cnt++;
    } else {
        if (slot->dominant_cnt > 0) {
            slot->dominant_cnt--;
        } else {
            slot->dominant_val = v;
            slot->dominant_cnt = 1;
        }
    }
    if (slot->total_samples < UINT32_MAX) slot->total_samples++;
}

/**
 * Return true if slot has a confident dominant value that is safe to
 * specialize on.
 *
 * "Confident" = total_samples >= CJIT_PROFILE_MIN_SAMPLES AND
 *               dominant_cnt * 100 / total_samples >= CJIT_PROFILE_MIN_CONFIDENCE_PCT.
 */
static inline bool cjit_arg_slot_confident(const cjit_arg_slot_t *slot)
{
    if (slot->total_samples < CJIT_PROFILE_MIN_SAMPLES) return false;
    /* Avoid division: rearrange to dominant_cnt * 100 >= pct * total_samples */
    return (uint64_t)slot->dominant_cnt * 100u
               >= (uint64_t)CJIT_PROFILE_MIN_CONFIDENCE_PCT * slot->total_samples;
}
