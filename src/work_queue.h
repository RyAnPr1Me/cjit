/**
 * work_queue.h – Lock-free bounded MPMC (multi-producer / multi-consumer)
 * work queue, based on Dmitry Vyukov's well-known algorithm.
 *
 * Algorithm overview
 * ──────────────────
 * The queue is a ring buffer of fixed capacity (must be a power of two).
 * Each slot carries an atomic sequence counter in addition to the payload.
 *
 *   Enqueue (producer):
 *     1. Load current tail position T with relaxed ordering.
 *     2. Compute the slot index: slot = &buf[T & mask].
 *     3. Load slot->seq with acquire ordering.
 *     4. If seq == T  → slot is free; try CAS(tail, T, T+1).
 *        If seq <  T  → queue is full (return false).
 *        If seq >  T  → another producer raced ahead; retry.
 *     5. Write data, then store seq = T+1 (release), making it visible.
 *
 *   Dequeue (consumer):
 *     1. Load current head position H with relaxed ordering.
 *     2. Compute slot = &buf[H & mask].
 *     3. Load slot->seq with acquire ordering.
 *     4. If seq == H+1 → item ready; try CAS(head, H, H+1).
 *        If seq <  H+1 → queue is empty (return false).
 *        If seq >  H+1 → another consumer raced ahead; retry.
 *     5. Read data, then store seq = H+capacity (release), freeing the slot.
 *
 * No ABA problem: the sequence counter grows monotonically and wraps only
 * after 2^64 operations.  No spurious failures beyond the expected retry.
 *
 * The compile_task payload is small (12 bytes); the full struct fits in one
 * cache line together with its sequence counter.
 *
 * Thread safety: fully lock-free; safe for any number of concurrent
 * producers and consumers.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include "../include/cjit.h"   /* func_id_t, opt_level_t */

/* ───────────────────────── tuneable constants ────────────────────────────── */

/**
 * Ring-buffer capacity.  Must be a power of two so that the modulo reduction
 * collapses to a bitmask.  256 slots give ample depth for 3 compiler threads
 * and one monitor thread under normal load.
 */
#define WQ_CAPACITY 256u

/** Compile-time assertion helper. */
#define WQ_STATIC_ASSERT(cond) typedef char _wq_sa[(cond) ? 1 : -1]
WQ_STATIC_ASSERT((WQ_CAPACITY & (WQ_CAPACITY - 1u)) == 0u); /* power of 2 */

/* ───────────────────────── payload type ─────────────────────────────────── */

/**
 * A single compilation request.
 *
 * Fields
 * ──────
 * func_id      : Identifies which function to recompile.
 * target_level : The desired optimisation tier for this compilation pass.
 * priority     : Higher values are preferred by compiler threads (0 = normal).
 *                Currently informational; threads process in FIFO order.
 * version_req  : The IR version that triggered this request; used to discard
 *                stale tasks if the IR was updated while the task was queued.
 * call_rate    : Observed calls/sec at the time of enqueue.  Carried through
 *                to the compiler thread for verbose logging and future
 *                profile-guided optimisation hints.  Zero for manual requests.
 */
typedef struct {
    func_id_t   func_id;
    opt_level_t target_level;
    uint32_t    priority;
    uint32_t    version_req;
    uint64_t    call_rate;   /* calls/sec at time of enqueue */
} compile_task_t;

/* ───────────────────────── queue structure ───────────────────────────────── */

/**
 * One ring-buffer slot.
 *
 * seq and data are packed so the whole slot fits in one 64-byte cache line.
 * Padding makes the size exactly 64 bytes regardless of platform alignment.
 *
 * Layout on LP64 (x86-64 / AArch64):
 *   seq  : 8 bytes  (atomic_uint_fast64_t)
 *   data : 24 bytes (compile_task_t: 4+4+4+4+8)
 *   pad  : 32 bytes
 *   total: 64 bytes = 1 cache line
 */
typedef struct {
    _Alignas(64) atomic_uint_fast64_t seq;  /* monotonic sequence counter    */
    compile_task_t                    data; /* payload (written before seq)  */
    uint8_t _pad[64 - sizeof(atomic_uint_fast64_t) - sizeof(compile_task_t)];
} wq_slot_t;

/**
 * The MPMC queue itself.
 *
 * enqueue_pos and dequeue_pos are on separate cache lines to eliminate
 * false sharing between producers and consumers.
 */
typedef struct {
    _Alignas(64) atomic_uint_fast64_t enqueue_pos; /* next slot to claim by producer */
    _Alignas(64) atomic_uint_fast64_t dequeue_pos; /* next slot to claim by consumer */
    wq_slot_t                         slots[WQ_CAPACITY];
} mpmc_queue_t;

/* ───────────────────────── public API ────────────────────────────────────── */

/**
 * Initialise the queue.
 *
 * Must be called exactly once before any enqueue/dequeue operations.
 * Sets every slot's sequence counter to its index so slot[0] is ready for
 * the first enqueue immediately.
 */
void mpmc_init(mpmc_queue_t *q);

/**
 * Try to enqueue a task.
 *
 * @return true  if the task was successfully enqueued.
 * @return false if the queue is full (back-pressure to the caller).
 *
 * Lock-free: may spin briefly on CAS contention from competing producers,
 * but never blocks.
 */
bool mpmc_enqueue(mpmc_queue_t *q, const compile_task_t *task);

/**
 * Try to dequeue a task.
 *
 * @return true  if a task was dequeued into *out.
 * @return false if the queue is currently empty.
 *
 * Lock-free: same spin characteristics as mpmc_enqueue.
 */
bool mpmc_dequeue(mpmc_queue_t *q, compile_task_t *out);

/**
 * Return the approximate number of items currently in the queue.
 *
 * This is a non-atomic snapshot: enqueue_pos and dequeue_pos are read
 * separately so the result may be stale by one item.  Useful only for
 * diagnostics (e.g. cjit_get_stats).
 */
uint32_t mpmc_size(const mpmc_queue_t *q);
