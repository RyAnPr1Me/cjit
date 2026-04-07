/**
 * func_table.h – Atomic function-pointer table.
 *
 * Design
 * ──────
 * The table is a flat array of func_table_entry_t structs, indexed by
 * func_id_t (a small integer assigned at registration time).
 *
 * Each entry contains:
 *   • func_ptr  – an _Atomic function pointer; the only field touched on
 *                 the hot path.
 *   • call_cnt  – an atomic call counter incremented by cjit_record_call().
 *   • cur_level – the optimisation level of the currently loaded code.
 *   • version   – incremented every time the function is recompiled; used
 *                 by compiler threads to detect stale queued tasks.
 *   • ir_source – pointer to the registered C-source string (IR).
 *   • name      – human-readable function name (for dlsym and diagnostics).
 *   • dl_handle – dlopen handle of the currently loaded shared object; the
 *                 compiler thread stores this so it can be retired (via
 *                 deferred_gc) when the pointer is swapped out.
 *
 * Hot path (read, called by runtime threads):
 *   jit_func_t f = atomic_load_explicit(&entry->func_ptr, memory_order_acquire);
 *   f(…);
 *
 * Swap path (write, called only by compiler threads):
 *   old_handle = entry->dl_handle;  // NOT atomic – protected by external seq
 *   entry->dl_handle = new_handle;
 *   atomic_store_explicit(&entry->func_ptr, new_fn, memory_order_release);
 *   dgc_retire(dgc, old_handle);
 *
 * Thread safety
 * ─────────────
 * func_ptr and call_cnt are fully atomic; they may be read/written from any
 * thread at any time.
 *
 * cur_level, version, ir_source, dl_handle are written only by compiler
 * threads.  Multiple compiler threads are prevented from compiling the same
 * function simultaneously by a per-entry compile_lock (a simple mutex that
 * is held only during the compilation work, never on the hot path).
 *
 * name is set once during registration and never modified.
 *
 * in_queue is an atomic flag used to prevent duplicate enqueue of the same
 * function when both the monitor thread and a manual cjit_request_recompile()
 * race.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stddef.h>
#include "../include/cjit.h"

/* ─────────────────────────── constants ─────────────────────────────────────── */

/**
 * Maximum length (including NUL) of a function name.
 *
 * 64 bytes covers every real-world C symbol name and keeps both
 * func_table_entry_t and ir_node_t compact.  Reducing from the
 * previous value of 128 saves 64 bytes per entry in each struct.
 */
#define CJIT_NAME_MAX 64

/* ─────────────────────────── entry ─────────────────────────────────────────── */

/**
 * One entry in the function table.
 *
 * Padded to a multiple of 64 bytes so that each entry occupies its own cache
 * line(s), preventing false sharing between entries accessed by different
 * threads.
 */
typedef struct {
    /*
     * HOT FIELDS (cache line 0)
     * Read by runtime threads on every call.
     */
    _Alignas(64)
    _Atomic(jit_func_t)         func_ptr;   /**< Current compiled function.     */
    atomic_uint_fast64_t        call_cnt;   /**< Call counter (relaxed incr.).  */
    _Atomic uint32_t             version;    /**< Recompile generation counter.  */
    atomic_int                  cur_level;  /**< opt_level_t of loaded code.    */

    /*
     * COLD FIELDS (cache line 1+)
     * Written only by registration or compiler threads; never on hot path.
     */
    _Alignas(64)
    const char                 *ir_source;  /**< C-source IR (owned by engine). */
    void                       *dl_handle;  /**< dlopen handle of current .so.  */
    func_id_t                   id;         /**< Self-reference for convenience.*/
    atomic_bool                 in_queue;   /**< True if already enqueued.      */
    /**
     * How long (ms) the most recent compilation of this function took.
     *
     * Written by the compiler thread after each compilation (relaxed store).
     * Read by the monitor thread (relaxed load) to compute an adaptive
     * cooloff: max(cfg.compile_cooloff_ms, 2 × last_compile_duration_ms).
     * This prevents re-enqueuing before the previous compile has likely
     * completed.  Zero overhead on the hot path.
     *
     * uint32_t (not uint_fast32_t) saves 4 bytes on LP64 targets where
     * uint_fast32_t expands to 64 bits.  Max representable value is ~49 days.
     */
    _Atomic uint32_t             last_compile_duration_ms;

    /**
     * Number of successful JIT recompilations of this function.
     *
     * Incremented by the compiler thread in func_table_swap() after each
     * successful compilation and pointer swap.  Read by the monitor thread
     * (relaxed load) to scale promotion thresholds.  uint32_t is sufficient
     * and saves 4 bytes vs uint_fast32_t on LP64.
     */
    _Atomic uint32_t             recompile_count;
    pthread_mutex_t             compile_lock; /**< Serialises concurrent compiles.*/
    char                        name[CJIT_NAME_MAX]; /**< Function symbol name. */
} func_table_entry_t;

/* ─────────────────────────── table ─────────────────────────────────────────── */

/**
 * The full function-pointer table.
 *
 * Managed by the cjit_engine; exposed here for inline access from cjit.c.
 */
typedef struct {
    func_table_entry_t *entries;   /**< Heap-allocated array [0..capacity).  */
    uint32_t            capacity;  /**< Maximum number of entries.           */
    atomic_uint_fast32_t count;    /**< Number of registered functions.      */
} func_table_t;

/* ─────────────────────────── API ───────────────────────────────────────────── */

/** Allocate and initialise a func_table_t. Returns NULL on failure. */
func_table_t *func_table_create(uint32_t capacity);

/** Free all resources held by the table (does not dlclose handles). */
void func_table_destroy(func_table_t *ft);

/**
 * Register a new function.
 *
 * Returns the assigned func_id_t, or CJIT_INVALID_FUNC_ID if the table is
 * full or if name is already registered.
 *
 * Thread safety: must NOT be called concurrently with other registrations.
 */
func_id_t func_table_register(func_table_t *ft,
                               const char   *name,
                               const char   *ir_source,
                               jit_func_t    initial_fn);

/**
 * Retrieve the func_table_entry_t for id.
 *
 * Returns NULL if id is out of range.  The returned pointer is stable for the
 * lifetime of the engine (entries array is never reallocated).
 */
func_table_entry_t *func_table_get(func_table_t *ft, func_id_t id);

/**
 * Atomically replace the function pointer for id.
 *
 * Called by compiler threads after a successful compilation.  Returns the
 * old dlopen handle that must be passed to dgc_retire().
 *
 * @param ft         Function table.
 * @param id         Function to update.
 * @param new_fn     Newly compiled function pointer.
 * @param new_handle dlopen handle for new_fn's shared object.
 * @param new_level  Optimisation level of the new code.
 * @return           The old dlopen handle (may be NULL for the first swap).
 */
void *func_table_swap(func_table_t *ft,
                      func_id_t     id,
                      jit_func_t    new_fn,
                      void         *new_handle,
                      opt_level_t   new_level);
