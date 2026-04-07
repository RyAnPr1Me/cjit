/**
 * work_queue.c – Lock-free bounded MPMC queue implementation.
 *
 * See work_queue.h for the algorithm description.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "work_queue.h"

#include <string.h>  /* memset */

/* ─────────────────────────── helpers ──────────────────────────────────────── */

static inline uint_fast64_t wq_load_seq(const wq_slot_t *s)
{
    /* Acquire ordering: we need to observe the data written before seq was
     * stored by the producer.                                               */
    return atomic_load_explicit(&s->seq, memory_order_acquire);
}

static inline void wq_store_seq(wq_slot_t *s, uint_fast64_t v)
{
    /* Release ordering: ensures data is visible before the sequence counter
     * that signals "slot is ready".                                         */
    atomic_store_explicit(&s->seq, v, memory_order_release);
}

/* ─────────────────────────── public functions ──────────────────────────────── */

void mpmc_init(mpmc_queue_t *q)
{
    atomic_store_explicit(&q->enqueue_pos, 0, memory_order_relaxed);
    atomic_store_explicit(&q->dequeue_pos, 0, memory_order_relaxed);

    for (uint_fast64_t i = 0; i < WQ_CAPACITY; ++i) {
        /*
         * Initial sequence = slot index.
         * This means slot[0] expects enqueue_pos==0 → seq==0 → it is free.
         * Slot[i] will become free once pos wraps around to i again, at which
         * point dequeue has advanced past it and stored seq = i + WQ_CAPACITY.
         */
        atomic_store_explicit(&q->slots[i].seq, i, memory_order_relaxed);
    }
}

bool mpmc_enqueue(mpmc_queue_t *q, const compile_task_t *task)
{
    wq_slot_t      *slot;
    uint_fast64_t   pos;
    uint_fast64_t   seq;
    intptr_t        diff;

    pos = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);

    for (;;) {
        slot = &q->slots[pos & (WQ_CAPACITY - 1u)];
        seq  = wq_load_seq(slot);
        diff = (intptr_t)(seq) - (intptr_t)(pos);

        if (diff == 0) {
            /*
             * Slot is free for us to claim.  Race with other producers via
             * CAS; only one will win.
             */
            if (atomic_compare_exchange_weak_explicit(
                    &q->enqueue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break; /* we own the slot */
            }
            /* Another producer claimed it first; reload pos and retry. */
        } else if (diff < 0) {
            /* Queue is full (consumer has not freed this slot yet). */
            return false;
        } else {
            /*
             * Another producer just incremented enqueue_pos past us; refresh
             * our view and retry.
             */
            pos = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
        }
    }

    /* Write the payload, then publish by advancing seq to pos+1. */
    slot->data = *task;
    wq_store_seq(slot, pos + 1);
    return true;
}

bool mpmc_dequeue(mpmc_queue_t *q, compile_task_t *out)
{
    wq_slot_t      *slot;
    uint_fast64_t   pos;
    uint_fast64_t   seq;
    intptr_t        diff;

    pos = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);

    for (;;) {
        slot = &q->slots[pos & (WQ_CAPACITY - 1u)];
        seq  = wq_load_seq(slot);
        diff = (intptr_t)(seq) - (intptr_t)(pos + 1);

        if (diff == 0) {
            /* Item is ready.  Race with other consumers via CAS. */
            if (atomic_compare_exchange_weak_explicit(
                    &q->dequeue_pos, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break; /* we own the slot */
            }
        } else if (diff < 0) {
            /* Queue is empty. */
            return false;
        } else {
            /* Another consumer raced ahead; refresh and retry. */
            pos = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
        }
    }

    /* Read the payload, then free the slot by advancing seq to pos+CAPACITY. */
    *out = slot->data;
    wq_store_seq(slot, pos + WQ_CAPACITY);
    return true;
}

uint32_t mpmc_size(const mpmc_queue_t *q)
{
    uint_fast64_t eq = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed);
    uint_fast64_t dq = atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
    if (eq >= dq) {
        uint_fast64_t sz = eq - dq;
        return (uint32_t)(sz < WQ_CAPACITY ? sz : WQ_CAPACITY);
    }
    return 0;
}
