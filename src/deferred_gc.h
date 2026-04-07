/**
 * deferred_gc.h – Grace-period deferred dlclose for RCU-safe reclamation.
 *
 * Problem
 * ───────
 * When a JIT compiler thread replaces an old function pointer with a freshly
 * compiled one, it must not immediately dlclose() the old shared-object
 * handle.  A runtime thread that loaded the old pointer *just before* the
 * swap may still be inside the old function.  Calling dlclose() while
 * another thread executes code from that library causes undefined behaviour
 * (SIGSEGV or silent data corruption).
 *
 * Solution: grace-period deferred reclamation
 * ────────────────────────────────────────────
 * 1. When a handle is replaced, the compiler thread "retires" it by pushing
 *    it onto a lock-free LIFO stack together with the current timestamp.
 *    This operation is O(1) and entirely wait-free (a single CAS loop).
 *
 * 2. A background GC thread wakes adaptively:
 *      • If the retire stack has pending (not-yet-freeable) entries it sleeps
 *        exactly until the youngest entry becomes freeable.
 *      • If the stack was empty last sweep it backs off exponentially
 *        (doubling up to grace_period_ms/2) to avoid burning CPU when idle.
 *    On each wake it harvests the stack atomically (a single CAS of the head
 *    to NULL) and for each entry:
 *      a. If (now - retire_time) ≥ grace_period_ms: call dlclose().
 *      b. Otherwise: push the entry back and record its remaining age.
 *
 * 3. The grace period is chosen to exceed the maximum time any runtime
 *    thread can remain inside a JIT-compiled function call.  100 ms is the
 *    default; latency-sensitive callers may tune this down.
 *
 * Why this gives safety without runtime overhead
 * ──────────────────────────────────────────────
 * Runtime threads perform only: atomic_load + indirect call.  They publish
 * nothing, update no epoch counters, and acquire no hazard pointers.
 *
 * Safety relies on the assumption that no single function call executes
 * for longer than grace_period_ms.  This is reasonable for any non-blocking
 * function; for long-running functions callers should either (a) increase the
 * grace period, or (b) use hazard pointers instead.
 *
 * The retire stack is a classic Michael-Scott lock-free LIFO:
 *
 *   Push:  new->next = head; CAS(head, old, new)   (retry on failure)
 *   Drain: local = CAS(head, head, NULL)             (single CAS to steal all)
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>

/* ─────────────────────────── types ─────────────────────────────────────────── */

/** One entry in the lock-free retire stack. */
typedef struct retire_entry {
    void                *handle;        /**< dlopen handle to close eventually. */
    uint64_t             retire_ms;     /**< Timestamp (ms since epoch) of retirement. */
    struct retire_entry *next;          /**< Intrusive linked-list pointer.     */
} retire_entry_t;

/**
 * Number of pre-allocated retire_entry_t nodes in the embedded pool.
 *
 * At any moment the retire stack holds at most (compiler_threads × 2) +
 * in-flight-GC entries.  256 is generous even for the maximum 16 compiler
 * threads; the pool is intentionally oversized so malloc fallback is
 * essentially never triggered in practice.
 */
#define DGC_POOL_SIZE 256u

/**
 * Deferred-GC context.
 *
 * One instance is embedded inside cjit_engine_t.  The GC thread is started
 * by dgc_start() and stopped by dgc_stop().
 */
typedef struct {
    /** Lock-free retire stack (NULL = empty). */
    _Atomic(retire_entry_t *) head;

    /**
     * Lock-free freelist of pre-allocated retire_entry_t nodes.
     *
     * dgc_retire() pops a node from this list instead of calling malloc().
     * dgc_sweep() returns freed nodes to the list instead of calling free().
     * On pool exhaustion dgc_retire() falls back to malloc() transparently.
     *
     * Uses the same CAS-loop LIFO pattern as the retire stack itself.
     */
    _Atomic(retire_entry_t *) pool_head;

    /**
     * Heap-allocated backing storage for the pool nodes.
     *
     * Allocated in dgc_init(), freed in dgc_stop() after the final sweep.
     * May be NULL if the initial allocation failed (graceful degradation to
     * malloc-only mode).
     */
    retire_entry_t *pool_storage;

    /** Milliseconds to wait before freeing a retired handle. */
    uint32_t  grace_period_ms;

    /** Set to true to request the GC thread to exit. */
    atomic_bool stop_flag;

    /** The GC background thread. */
    pthread_t   thread;

    /** Running count of handles pushed onto the retire stack (stats). */
    atomic_uint_fast64_t total_retired;

    /** Running count of handles that have been dlclose'd (stats). */
    atomic_uint_fast64_t total_freed;
} deferred_gc_t;

/* ─────────────────────────── API ───────────────────────────────────────────── */

/**
 * Initialise a deferred_gc_t.
 *
 * @param dgc              Context to initialise.
 * @param grace_period_ms  Minimum age (ms) before a handle may be freed.
 */
void dgc_init(deferred_gc_t *dgc, uint32_t grace_period_ms);

/**
 * Start the background GC thread.
 *
 * Must be called after dgc_init().
 */
void dgc_start(deferred_gc_t *dgc);

/**
 * Signal the GC thread to stop and wait for it to exit.
 *
 * Any handles still on the retire stack are freed immediately (bypassing
 * the grace period) to avoid leaks at shutdown.
 */
void dgc_stop(deferred_gc_t *dgc);

/**
 * Retire a dlopen handle.
 *
 * Called by compiler threads after an atomic pointer swap.  Wait-free: a
 * single CAS loop that succeeds on the first or second try in practice.
 *
 * @param dgc     Deferred-GC context.
 * @param handle  The dlopen handle returned by dlopen(); may be NULL (no-op).
 */
void dgc_retire(deferred_gc_t *dgc, void *handle);

/**
 * Run a single GC sweep synchronously (for testing / shutdown).
 *
 * Harvests the retire stack and frees any entry older than the grace period.
 * If force is true, all entries are freed unconditionally.
 *
 * @return The minimum milliseconds until the next pending entry will become
 *         freeable, or 0 if nothing is pending (stack was empty or everything
 *         was freed).  The GC thread uses this to schedule its next wake-up
 *         precisely, avoiding both over-sleeping and busy-polling.
 */
uint32_t dgc_sweep(deferred_gc_t *dgc, bool force);
