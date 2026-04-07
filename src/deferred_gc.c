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

    struct timespec sleep_ts;
    sleep_ts.tv_sec  = dgc->sweep_interval_ms / 1000;
    sleep_ts.tv_nsec = (long)(dgc->sweep_interval_ms % 1000) * 1000000L;

    while (!atomic_load_explicit(&dgc->stop_flag, memory_order_acquire)) {
        nanosleep(&sleep_ts, NULL);
        dgc_sweep(dgc, /*force=*/false);
    }

    /* On shutdown: free everything unconditionally. */
    dgc_sweep(dgc, /*force=*/true);
    return NULL;
}

/* ─────────────────────── public API ─────────────────────────────────────────── */

void dgc_init(deferred_gc_t *dgc, uint32_t grace_period_ms)
{
    memset(dgc, 0, sizeof(*dgc));
    atomic_init(&dgc->head,         NULL);
    atomic_init(&dgc->stop_flag,    false);
    atomic_init(&dgc->total_retired, 0);
    atomic_init(&dgc->total_freed,   0);

    dgc->grace_period_ms  = grace_period_ms;
    /* Sweep roughly 4 times per grace period so handles are freed promptly. */
    dgc->sweep_interval_ms = (grace_period_ms / 4) ? (grace_period_ms / 4) : 1;
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
}

void dgc_retire(deferred_gc_t *dgc, void *handle)
{
    if (!handle) return;

    retire_entry_t *entry = malloc(sizeof(*entry));
    if (!entry) {
        /* Allocation failure is non-fatal; leak the handle rather than crash.
         * In production, a pre-allocated pool would be used instead.       */
        dlclose(handle);
        return;
    }
    entry->handle     = handle;
    entry->retire_ms  = now_ms();
    entry->next       = NULL;

    stack_push(dgc, entry);
    atomic_fetch_add_explicit(&dgc->total_retired, 1, memory_order_relaxed);
}

void dgc_sweep(deferred_gc_t *dgc, bool force)
{
    retire_entry_t *list  = stack_drain(dgc);
    retire_entry_t *keep  = NULL;  /* items not yet old enough */
    retire_entry_t *next;
    uint64_t        now   = now_ms();

    while (list) {
        next = list->next;

        if (force || (now - list->retire_ms) >= dgc->grace_period_ms) {
            /* Safe to reclaim: all threads that could have loaded the old
             * pointer before the swap have had at least grace_period_ms to
             * finish their calls.                                           */
            if (list->handle) dlclose(list->handle);
            atomic_fetch_add_explicit(&dgc->total_freed, 1, memory_order_relaxed);
            free(list);
        } else {
            /* Still within grace period; push back for the next sweep. */
            list->next = keep;
            keep       = list;
        }
        list = next;
    }

    /* Re-push items that were not yet ready. */
    while (keep) {
        next = keep->next;
        stack_push(dgc, keep);
        keep = next;
    }
}
