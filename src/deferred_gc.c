/**
 * deferred_gc.c – Grace-period deferred dlclose implementation.
 *
 * See deferred_gc.h for the design rationale.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "deferred_gc.h"

#include <stdlib.h>   /* malloc / free */
#include <string.h>   /* memset        */
#include <time.h>     /* clock_gettime */
#include <dlfcn.h>    /* dlclose       */
#include <errno.h>

/* ─────────────────────── pool helpers ──────────────────────────────────────── */

/**
 * Push a node onto the pre-allocated freelist (lock-free LIFO, same CAS
 * pattern as the retire stack).
 */
static void pool_push(deferred_gc_t *dgc, retire_entry_t *entry)
{
    retire_entry_t *old_head;
    do {
        old_head    = atomic_load_explicit(&dgc->pool_head, memory_order_relaxed);
        entry->next = old_head;
    } while (!atomic_compare_exchange_weak_explicit(
                 &dgc->pool_head, &old_head, entry,
                 memory_order_release, memory_order_relaxed));
}

/**
 * Pop one node from the pre-allocated freelist.
 *
 * Returns NULL if the pool is exhausted (caller falls back to malloc).
 */
static retire_entry_t *pool_pop(deferred_gc_t *dgc)
{
    retire_entry_t *head;
    head = atomic_load_explicit(&dgc->pool_head, memory_order_acquire);
    for (;;) {
        if (!head) return NULL;
        if (atomic_compare_exchange_weak_explicit(
                &dgc->pool_head, &head, head->next,
                memory_order_acquire, memory_order_relaxed))
            return head;
    }
}

/**
 * Return true if entry is within the pre-allocated pool backing storage.
 *
 * Used by dgc_sweep() to decide whether to pool_push() or free() a node.
 */
static bool pool_owns(const deferred_gc_t *dgc, const retire_entry_t *entry)
{
    if (!dgc->pool_storage) return false;
    return entry >= dgc->pool_storage &&
           entry <  dgc->pool_storage + DGC_POOL_SIZE;
}



/* ─────────────────────── internal helpers ──────────────────────────────────── */

/**
 * Return the current monotonic time in milliseconds.
 *
 * Uses CLOCK_MONOTONIC so it is not affected by system-time adjustments.
 */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/**
 * Push a retire_entry onto the lock-free LIFO stack.
 *
 * The CAS loop retries only when another concurrent push wins first.  In
 * practice this converges in one or two iterations.
 *
 * Memory ordering:
 *   - The store of entry->next uses relaxed because it is a freshly allocated
 *     object not yet visible to any other thread.
 *   - The CAS uses release so that any writes to *entry (retire_ms, handle)
 *     are visible to the GC thread once it reads the pointer.
 */
static void stack_push(deferred_gc_t *dgc, retire_entry_t *entry)
{
    retire_entry_t *old_head;
    do {
        old_head   = atomic_load_explicit(&dgc->head, memory_order_relaxed);
        entry->next = old_head;
    } while (!atomic_compare_exchange_weak_explicit(
                 &dgc->head, &old_head, entry,
                 memory_order_release,
                 memory_order_relaxed));
}

/**
 * Atomically drain the entire retire stack in one CAS.
 *
 * Returns the harvested list (singly-linked via ->next, NULL-terminated).
 * If the stack was empty, returns NULL.
 *
 * Memory ordering:
 *   - The CAS uses acquire so that all stores performed by producers before
 *     their release-CAS are visible to the GC thread after this acquire.
 */
static retire_entry_t *stack_drain(deferred_gc_t *dgc)
{
    retire_entry_t *head;
    head = atomic_load_explicit(&dgc->head, memory_order_relaxed);
    if (!head) return NULL;

    while (!atomic_compare_exchange_weak_explicit(
               &dgc->head, &head, NULL,
               memory_order_acquire,
               memory_order_relaxed)) {
        if (!head) return NULL;
    }
    return head;
}

/* ─────────────────────── GC thread ─────────────────────────────────────────── */

static void *gc_thread_fn(void *arg)
{
    deferred_gc_t *dgc = (deferred_gc_t *)arg;

    /*
     * Adaptive sweep interval.
     *
     * The GC thread self-adjusts its sleep duration based on the retire stack
     * state observed after each sweep, rather than using a fixed grace/4 value:
     *
     *   Pending entries remain  →  sleep exactly until the next one is freeable
     *                               (dgc_sweep returns remaining_ms > 0).
     *                               This is optimal: we wake up precisely when
     *                               there is actionable work, never earlier.
     *
     *   Nothing left pending    →  exponential back-off (doubles each idle
     *                               sweep) up to grace_period_ms/2.  At idle
     *                               the GC thread barely consumes CPU.
     *
     *   On the very first sweep  →  start at grace_period_ms/4 (the legacy
     *                               default) so the first batch of retired
     *                               handles is freed promptly.
     *
     * No external inputs (queue depth, compile frequency) are needed: the
     * retire stack itself is the sole signal.  A busy JIT engine retires
     * handles frequently; those handles will have large remaining_ms on the
     * first sweep after retirement, driving the sleep to exactly the right
     * duration.
     */
    uint32_t grace   = dgc->grace_period_ms;
    uint32_t max_ms  = grace / 2 > 0 ? grace / 2 : 1;
    uint32_t next_ms = grace / 4 > 0 ? grace / 4 : 1;

    while (!atomic_load_explicit(&dgc->stop_flag, memory_order_acquire)) {
        struct timespec ts = {
            .tv_sec  = (time_t)(next_ms / 1000),
            .tv_nsec = (long)(next_ms % 1000) * 1000000L,
        };
        nanosleep(&ts, NULL);

        uint32_t remaining = dgc_sweep(dgc, /*force=*/false);

        if (remaining > 0) {
            /* Sleep until the next pending handle is ready to be freed.
             * Add 1 ms so we don't wake a tiny bit too early. */
            next_ms = remaining + 1;
            if (next_ms > max_ms) next_ms = max_ms;
        } else {
            /* Nothing pending: back off to reduce idle CPU usage. */
            next_ms *= 2;
            if (next_ms > max_ms) next_ms = max_ms;
        }
    }

    /* On shutdown: free everything unconditionally. */
    dgc_sweep(dgc, /*force=*/true);
    return NULL;
}

/* ─────────────────────── public API ─────────────────────────────────────────── */

void dgc_init(deferred_gc_t *dgc, uint32_t grace_period_ms)
{
    memset(dgc, 0, sizeof(*dgc));
    atomic_init(&dgc->head,          NULL);
    atomic_init(&dgc->pool_head,     NULL);
    atomic_init(&dgc->stop_flag,     false);
    atomic_init(&dgc->total_retired,  0);
    atomic_init(&dgc->total_freed,    0);

    dgc->grace_period_ms = grace_period_ms;

    /*
     * Pre-allocate the node pool.  Push all DGC_POOL_SIZE nodes onto the
     * freelist so dgc_retire() can pop them without calling malloc().
     * If allocation fails we degrade gracefully to malloc-only mode.
     */
    dgc->pool_storage = malloc(DGC_POOL_SIZE * sizeof(retire_entry_t));
    if (dgc->pool_storage) {
        for (uint32_t i = 0; i < DGC_POOL_SIZE; ++i)
            pool_push(dgc, &dgc->pool_storage[i]);
    }
}

void dgc_start(deferred_gc_t *dgc)
{
    atomic_store_explicit(&dgc->stop_flag, false, memory_order_relaxed);
    pthread_create(&dgc->thread, NULL, gc_thread_fn, dgc);
}

void dgc_stop(deferred_gc_t *dgc)
{
    atomic_store_explicit(&dgc->stop_flag, true, memory_order_release);
    pthread_join(dgc->thread, NULL);
    /* gc_thread_fn already called dgc_sweep(force=true) before exit. */

    /* Free the pre-allocated node pool backing storage. */
    free(dgc->pool_storage);
    dgc->pool_storage = NULL;
}

void dgc_retire(deferred_gc_t *dgc, void *handle)
{
    if (!handle) return;

    /*
     * Obtain a retire_entry_t node from the pre-allocated pool if possible,
     * falling back to malloc() only when the pool is exhausted (extremely
     * rare: requires >DGC_POOL_SIZE concurrent in-flight retirements).
     */
    retire_entry_t *entry = pool_pop(dgc);
    if (!entry) {
        entry = malloc(sizeof(*entry));
        if (!entry) {
            /* Allocation failure is non-fatal; dlclose immediately rather
             * than crash or leak.                                         */
            dlclose(handle);
            return;
        }
    }
    entry->handle     = handle;
    entry->retire_ms  = now_ms();
    entry->next       = NULL;

    stack_push(dgc, entry);
    atomic_fetch_add_explicit(&dgc->total_retired, 1, memory_order_relaxed);
}

uint32_t dgc_sweep(deferred_gc_t *dgc, bool force)
{
    retire_entry_t *list  = stack_drain(dgc);
    retire_entry_t *keep  = NULL;
    retire_entry_t *next;
    uint64_t        now   = now_ms();
    uint32_t        min_remaining = 0;  /* min ms until next pending entry is freeable */

    while (list) {
        next = list->next;

        if (force || (now - list->retire_ms) >= dgc->grace_period_ms) {
            /* Safe to reclaim. */
            if (list->handle) dlclose(list->handle);
            atomic_fetch_add_explicit(&dgc->total_freed, 1, memory_order_relaxed);
            /* Return the node to the pool if it came from there; otherwise
             * free() it (it was malloc'd as a pool-exhaustion fallback).  */
            if (pool_owns(dgc, list))
                pool_push(dgc, list);
            else
                free(list);
        } else {
            /* Not yet ready: compute remaining lifetime with underflow guard. */
            uint64_t age = now - list->retire_ms;
            uint32_t remaining = (age < dgc->grace_period_ms)
                                     ? (uint32_t)(dgc->grace_period_ms - age)
                                     : 1u; /* defensive: treat as almost ready */
            if (min_remaining == 0 || remaining < min_remaining)
                min_remaining = remaining;

            list->next = keep;
            keep       = list;
        }
        list = next;
    }

    /* Re-push items that are not yet ready. */
    while (keep) {
        next = keep->next;
        stack_push(dgc, keep);
        keep = next;
    }

    return min_remaining;
}
