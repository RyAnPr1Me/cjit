/**
 * ir_cache.h – Multi-generation LRU cache for JIT function IR strings.
 *
 * Motivation
 * ──────────
 * A JIT engine may register hundreds or thousands of functions.  Keeping every
 * function's IR (C source) in memory at all times causes memory churn,
 * pollutes the allocator, and wastes L3 cache on code that is rarely (or
 * never) compiled.  Only a small subset of functions is "hot" at any moment.
 *
 * Solution: multi-generation LRU with disk spill
 * ───────────────────────────────────────────────
 * IR strings are partitioned into three generations based on how recently
 * they were requested for compilation:
 *
 *   HOT  (gen 0) – in memory; at most `hot_capacity` entries.
 *                  These are the most recently compiled functions.
 *
 *   WARM (gen 1) – in memory; at most `warm_capacity` entries.
 *                  Recently compiled but not re-requested since.
 *
 *   COLD (gen 2) – on disk only; IR string freed from heap.
 *                  Written to `ir_disk_dir/<func_id>_<name>.ir` at
 *                  registration time, so the file is always present.
 *
 * Generation transitions
 * ──────────────────────
 *
 *  register()     → place in HOT (evict HOT→WARM→COLD if at capacity)
 *  get_ir()       → if HOT : move to MRU position inside HOT list
 *                   if WARM: promote to HOT MRU (may cascade HOT→WARM eviction)
 *                   if COLD: load from disk, promote to WARM MRU
 *                            (may cascade WARM→COLD eviction)
 *
 *  HOT eviction   → demote LRU-HOT to WARM MRU (IR stays in memory)
 *  WARM eviction  → demote LRU-WARM to COLD   (IR freed; disk file kept)
 *
 * Memory invariant
 * ────────────────
 *   live heap bytes for IR ≤ (hot_capacity + warm_capacity) × avg_ir_size
 *
 * All other functions' IR lives only on disk (a few KB per function).
 *
 * Thread safety
 * ─────────────
 * A single per-cache mutex serialises all LRU list mutations.  The mutex is
 * never held on the runtime hot path (cjit_get_func / cjit_record_call).
 * It is held only when a compiler thread calls ir_cache_get_ir(), which
 * already involves filesystem I/O for cold functions.
 *
 * The per-generation statistics counters are atomic and can be read without
 * holding the lock.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>
#include "../include/cjit.h"  /* func_id_t */

/* ═══════════════════════════ generation tags ══════════════════════════════ */

#define IR_GEN_HOT  0   /**< In memory, most recently used.          */
#define IR_GEN_WARM 1   /**< In memory, less recently used.          */
#define IR_GEN_COLD 2   /**< On disk only; IR freed from heap.       */

typedef uint8_t ir_gen_t;

/* ═══════════════════════════ per-function node ════════════════════════════ */

/**
 * One node per registered function.  Nodes live in a flat array indexed by
 * func_id_t; pointers inside the array are stable for the engine lifetime.
 *
 * A node is in exactly one of three states at any time:
 *   HOT  → linked in the hot  LRU list; ir_source != NULL
 *   WARM → linked in the warm LRU list; ir_source != NULL
 *   COLD → not in any list;             ir_source == NULL; disk file exists
 */
typedef struct ir_node {
    func_id_t        func_id;
    char             name[128];       /**< Function name (for disk filename). */
    char            *ir_source;       /**< Heap copy of IR; NULL when COLD.   */
    char             disk_path[300];  /**< Absolute path to on-disk IR file.  */

    uint64_t         access_cnt;      /**< Times ir_cache_get_ir() was called.*/
    uint64_t         last_access_ms;  /**< Monotonic timestamp of last access.*/
    ir_gen_t         gen;             /**< Current generation (HOT/WARM/COLD).*/
    bool             registered;      /**< True after ir_cache_register().    */

    /* Intrusive doubly-linked list links (within a generation's LRU list).
     * prev → more recently used; next → less recently used.
     * NULL at both ends when COLD or not yet registered.                    */
    struct ir_node  *prev;
    struct ir_node  *next;
} ir_node_t;

/* ═══════════════════════════ cache structure ══════════════════════════════ */

/**
 * The multi-generation LRU IR cache.
 *
 * One instance is embedded in cjit_engine_t.
 */
typedef struct {
    ir_node_t   *nodes;             /**< Flat array [0..max_funcs).           */
    uint32_t     max_funcs;

    /* HOT generation doubly-linked LRU list (head = MRU, tail = LRU). */
    ir_node_t   *hot_head;
    ir_node_t   *hot_tail;
    uint32_t     hot_count;
    uint32_t     hot_capacity;      /**< Max entries allowed in HOT gen.     */

    /* WARM generation doubly-linked LRU list. */
    ir_node_t   *warm_head;
    ir_node_t   *warm_tail;
    uint32_t     warm_count;
    uint32_t     warm_capacity;     /**< Max entries allowed in WARM gen.    */

    uint32_t     cold_count;        /**< Functions on disk only.             */
    uint32_t     total_registered;  /**< Total functions registered.         */

    pthread_mutex_t lock;           /**< Serialises all list mutations.      */

    char         ir_dir[256];       /**< Base directory for .ir disk files.  */

    /* Per-operation statistics (lock-free reads without holding lock). */
    atomic_uint_fast64_t stat_disk_writes;   /**< IR files written to disk.  */
    atomic_uint_fast64_t stat_disk_reads;    /**< IR files read from disk.   */
    atomic_uint_fast64_t stat_evictions;     /**< HOT→WARM or WARM→COLD.     */
    atomic_uint_fast64_t stat_promotions;    /**< COLD→WARM or WARM→HOT.     */
    atomic_uint_fast64_t stat_cache_hits;    /**< get_ir() served from mem.  */
    atomic_uint_fast64_t stat_cache_misses;  /**< get_ir() loaded from disk. */
} ir_lru_cache_t;

/* ═══════════════════════════ public API ═══════════════════════════════════ */

/**
 * Allocate and initialise the cache.
 *
 * @param max_funcs    Upper bound on registered functions (= engine capacity).
 * @param hot_cap      Max functions whose IR is kept in HOT generation.
 * @param warm_cap     Max functions whose IR is kept in WARM generation.
 * @param ir_dir       Directory for on-disk IR files (created if absent).
 *                     Pass NULL to use the default "/tmp/cjit_ir_<pid>".
 * @return Pointer to new cache, or NULL on allocation failure.
 */
ir_lru_cache_t *ir_cache_create(uint32_t    max_funcs,
                                 uint32_t    hot_cap,
                                 uint32_t    warm_cap,
                                 const char *ir_dir);

/**
 * Destroy the cache and release all resources.
 *
 * Frees all heap IR strings.  Does NOT delete the on-disk IR files (they
 * may be useful for post-mortem inspection).  Does NOT dlclose any handles.
 */
void ir_cache_destroy(ir_lru_cache_t *cache);

/**
 * Register a function's IR with the cache.
 *
 * Makes a heap copy of @p ir_source, writes it to disk as a permanent
 * backup, and inserts the node into the HOT generation (cascading any
 * required evictions to maintain capacity invariants).
 *
 * Must be called from a single registration thread before cjit_start().
 *
 * @param cache      The cache.
 * @param func_id    Stable function ID (0-based index into engine table).
 * @param func_name  Symbol name (used to build the on-disk filename).
 * @param ir_source  C source string to store.
 * @return true on success; false if func_id is out of range or disk write
 *         failed.
 */
bool ir_cache_register(ir_lru_cache_t *cache,
                        func_id_t       func_id,
                        const char     *func_name,
                        const char     *ir_source);

/**
 * Return a heap-allocated copy of the IR for @p func_id.
 *
 * The caller MUST free() the returned string after use.
 *
 * Side effects (performed under the cache lock):
 *   • HOT  entry → moved to MRU position in HOT list.
 *   • WARM entry → promoted to HOT MRU; LRU-HOT demoted to WARM if needed.
 *   • COLD entry → IR loaded from disk; promoted to WARM MRU;
 *                  LRU-WARM demoted to COLD if needed.
 *
 * @return Heap-allocated IR string, or NULL on error (unregistered func_id,
 *         disk read failure, allocation failure).
 */
char *ir_cache_get_ir(ir_lru_cache_t *cache, func_id_t func_id);

/**
 * Return the current generation of a function's IR entry.
 *
 * Returns IR_GEN_COLD for any unregistered func_id.
 * Does NOT acquire the lock; the result may be stale by one transition.
 */
ir_gen_t ir_cache_get_generation(const ir_lru_cache_t *cache, func_id_t func_id);

/**
 * Snapshot of IR cache statistics.
 */
typedef struct {
    uint32_t hot_count;
    uint32_t warm_count;
    uint32_t cold_count;
    uint32_t total_registered;
    uint64_t disk_writes;
    uint64_t disk_reads;
    uint64_t evictions;
    uint64_t promotions;
    uint64_t cache_hits;
    uint64_t cache_misses;
} ir_cache_stats_t;

/** Read a lock-free snapshot of cache statistics. */
ir_cache_stats_t ir_cache_get_stats(const ir_lru_cache_t *cache);

/** Print a formatted statistics block to stderr. */
void ir_cache_print_stats(const ir_lru_cache_t *cache);
