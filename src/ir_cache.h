/**
 * ir_cache.h – Multi-generation LRU cache for JIT function IR strings,
 *              with system-memory-pressure awareness.
 *
 * Overview
 * ────────
 * IR strings (C source for JIT compilation) are partitioned across three
 * generations based on recency of access:
 *
 *   HOT  (gen 0) – in memory; at most `hot_capacity` entries.
 *   WARM (gen 1) – in memory; at most `warm_capacity` entries.
 *   COLD (gen 2) – on disk only; IR freed from heap.
 *
 * A permanent copy of every IR string is written to disk at registration time
 * so that COLD entries can be reloaded on demand without data loss.
 *
 * Generation transitions on access (ir_cache_get_ir):
 *   HOT  → stay HOT, move to MRU position.
 *   WARM → promote to HOT MRU; cascade LRU-HOT→WARM if needed.
 *   COLD → load from disk; promote to WARM MRU; cascade LRU-WARM→COLD.
 *
 * Memory-pressure monitoring
 * ──────────────────────────
 * A dedicated background thread reads /proc/meminfo every
 * `mem_check_interval_ms` milliseconds and computes a pressure level:
 *
 *   NORMAL   – MemAvailable ≥ mem_low_pct   % of MemTotal
 *   MEDIUM   – MemAvailable  < mem_low_pct   % (but ≥ mem_high_pct)
 *   HIGH     – MemAvailable  < mem_high_pct  % (but ≥ mem_critical_pct)
 *   CRITICAL – MemAvailable  < mem_critical_pct %
 *
 * When pressure rises the effective capacities are automatically reduced:
 *   NORMAL   → 100 % of configured hot_capacity / warm_capacity
 *   MEDIUM   →  75 %
 *   HIGH     →  50 %
 *   CRITICAL →  25 % (minimum 1)
 *
 * Entries that exceed the new effective capacity are evicted proactively in
 * the background:  HOT→WARM (IR stays in memory) then WARM→COLD (IR freed,
 * disk copy remains).  Subsequent insertions also respect the effective cap.
 *
 * When pressure falls the capacities widen again; the cache refills naturally
 * as functions are next accessed for compilation.
 *
 * Thread safety
 * ─────────────
 * All LRU list mutations are serialised under a single per-cache mutex.
 * Disk I/O and malloc/free are performed outside the lock.
 * The pressure level and mem-info counters are atomic; they are readable
 * from any thread at any time without acquiring the lock.
 * The hot-path (cjit_get_func / cjit_record_call) never touches this cache.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <pthread.h>
#include "../include/cjit.h"   /* func_id_t, mem_pressure_t */

/* ═══════════════════════════ per-function node ════════════════════════════ */

/** IR generation tags. */
#define IRC_GEN_HOT  ((uint8_t)0)  /**< In memory; most-recently-used tier.  */
#define IRC_GEN_WARM ((uint8_t)1)  /**< In memory; less-recently-used tier.  */
#define IRC_GEN_COLD ((uint8_t)2)  /**< On disk only; IR freed from heap.    */

/**
 * One LRU node per registered function.
 *
 * Stored in a flat array indexed by func_id_t; pointers are stable for the
 * engine lifetime.  A node is in exactly one state at any time:
 *   HOT  – linked in hot  list; ir_source != NULL
 *   WARM – linked in warm list; ir_source != NULL
 *   COLD – not in any list;     ir_source == NULL; disk file exists
 */
typedef struct ir_node {
    func_id_t        func_id;
    char             name[128];       /**< Sanitised function name.           */
    char            *ir_source;       /**< Heap copy; NULL when COLD.         */
    char             disk_path[512];  /**< Absolute path to on-disk .ir file. */

    uint64_t         access_cnt;      /**< Times get_ir() was called.         */
    uint64_t         last_access_ms;  /**< Monotonic timestamp (ms).          */
    uint8_t          gen;             /**< IR_GEN_HOT / WARM / COLD           */
    bool             registered;      /**< True after ir_cache_register().    */

    struct ir_node  *prev;            /**< Toward MRU end of list.            */
    struct ir_node  *next;            /**< Toward LRU end of list.            */
} ir_node_t;

/* ═══════════════════════════ cache structure ══════════════════════════════ */

/**
 * Multi-generation LRU IR cache with memory-pressure awareness.
 *
 * One instance is embedded inside cjit_engine_t.
 */
typedef struct {
    ir_node_t   *nodes;              /**< Flat array [0..max_funcs).          */
    uint32_t     max_funcs;

    /* HOT generation LRU (head = MRU, tail = LRU). */
    ir_node_t   *hot_head, *hot_tail;
    uint32_t     hot_count;
    uint32_t     hot_capacity;       /**< Nominal max (pressure may reduce).  */

    /* WARM generation LRU. */
    ir_node_t   *warm_head, *warm_tail;
    uint32_t     warm_count;
    uint32_t     warm_capacity;      /**< Nominal max.                        */

    uint32_t     cold_count;         /**< Entries residing on disk only.      */
    uint32_t     total_registered;

    pthread_mutex_t lock;            /**< Serialises all list mutations.      */

    char         ir_dir[256];        /**< Base directory for .ir disk files.  */

    /* ── Memory pressure monitoring ─────────────────────────────────────── */

    /** Current system-memory pressure level (atomically updated). */
    atomic_int   pressure;           /**< mem_pressure_t cast to int.         */

    /** Most-recent MemAvailable reading from /proc/meminfo (kB). */
    atomic_uint_fast64_t mem_available_kb;

    /** Most-recent MemTotal reading from /proc/meminfo (kB). */
    atomic_uint_fast64_t mem_total_kb;

    /** How often the pressure thread wakes (ms). */
    uint32_t     mem_check_interval_ms;

    /** available% below this → MEDIUM pressure. */
    uint32_t     mem_low_pct;

    /** available% below this → HIGH pressure. */
    uint32_t     mem_high_pct;

    /** available% below this → CRITICAL pressure. */
    uint32_t     mem_critical_pct;

    /** Set to true to stop the pressure-monitor thread. */
    atomic_bool  stop_pressure_flag;

    /** Background pressure-monitor thread handle. */
    pthread_t    pressure_thread;

    /* ── Async I/O prefetch pool ─────────────────────────────────────────── */

    /**
     * Mutex-protected bounded FIFO for prefetch requests.
     *
     * The monitor thread calls ir_cache_prefetch() which non-blockingly pushes
     * a func_id_t onto this queue.  Dedicated I/O threads drain it by calling
     * ir_cache_get_ir() which promotes COLD IR into the WARM generation.
     *
     * Using a mutex FIFO (not a lock-free queue) is correct and efficient here:
     * prefetch submissions are rare (once per function, at warm-up time) so
     * there is no contention.  The mutex is never held on the hot path.
     */
#define IRC_PREFETCH_CAP 128u
    func_id_t        pf_buf[IRC_PREFETCH_CAP]; /**< Ring-buffer of func ids.   */
    uint32_t         pf_head, pf_count;
    pthread_mutex_t  pf_mutex;
    pthread_cond_t   pf_cond;

    /** Background I/O thread pool (heap array of num_io_threads handles). */
    pthread_t       *io_threads;
    uint32_t         num_io_threads;

    /** Set to true to stop all I/O threads. */
    atomic_bool      stop_io_flag;

    /* ── Statistics (lock-free reads at any time) ────────────────────────── */
    atomic_uint_fast64_t stat_disk_writes;
    atomic_uint_fast64_t stat_disk_reads;
    atomic_uint_fast64_t stat_evictions;        /**< HOT→WARM or WARM→COLD.  */
    atomic_uint_fast64_t stat_promotions;       /**< COLD→WARM or WARM→HOT.  */
    atomic_uint_fast64_t stat_cache_hits;       /**< get_ir() from memory.   */
    atomic_uint_fast64_t stat_cache_misses;     /**< get_ir() loaded disk.   */
    atomic_uint_fast64_t stat_pressure_evictions; /**< Evictions due to mem. */
} ir_lru_cache_t;

/* ═══════════════════════════ configuration ════════════════════════════════ */

/**
 * Parameters for ir_cache_create().  All fields are optional; zero/NULL
 * values are replaced with defaults.
 */
typedef struct {
    uint32_t    max_funcs;            /**< Must equal engine max_functions.   */
    uint32_t    hot_cap;              /**< Nominal HOT capacity  (def: 64).   */
    uint32_t    warm_cap;             /**< Nominal WARM capacity (def: 128).  */
    const char *ir_dir;              /**< IR dir; NULL → /tmp/cjit_ir_<pid>.  */
    uint32_t    mem_check_interval_ms;/**< Pressure-check period (def: 500). */
    uint32_t    mem_low_pct;          /**< % avail → MEDIUM  (def: 20).      */
    uint32_t    mem_high_pct;         /**< % avail → HIGH    (def: 10).      */
    uint32_t    mem_critical_pct;     /**< % avail → CRITICAL (def:  5).     */
    /**
     * Number of dedicated async I/O threads for IR prefetch.
     * 0 = disable (reads are done synchronously in the compiler thread).
     * Default: 2.
     */
    uint32_t    num_io_threads;
} ir_cache_config_t;

/* ═══════════════════════════ public API ═══════════════════════════════════ */

/**
 * Allocate and initialise the cache.  Starts the pressure-monitor thread
 * immediately so that a meaningful pressure reading is available before the
 * first compilation.
 *
 * @return Pointer to new cache, or NULL on allocation failure.
 */
ir_lru_cache_t *ir_cache_create(const ir_cache_config_t *cfg);

/**
 * Destroy the cache: stop the pressure thread, free all heap IR strings.
 * Does NOT delete the on-disk .ir files.
 */
void ir_cache_destroy(ir_lru_cache_t *cache);

/**
 * Register a function's IR.  Makes a heap copy, writes it to disk as a
 * permanent backup, and inserts into the HOT generation (evicting LRU
 * entries as needed to maintain capacity invariants).
 *
 * Must be called from a single thread before cjit_start().
 */
bool ir_cache_register(ir_lru_cache_t *cache,
                        func_id_t       func_id,
                        const char     *func_name,
                        const char     *ir_source);

/**
 * Return a heap-allocated copy of the IR for func_id (caller must free()).
 *
 * Promotes the entry toward HOT; loads from disk if COLD.
 * Returns NULL on error.
 */
char *ir_cache_get_ir(ir_lru_cache_t *cache, func_id_t func_id);

/**
 * Submit a non-blocking async prefetch request for a function whose IR may
 * be COLD (on disk only).
 *
 * Returns immediately without doing any I/O.  An I/O thread will pick up
 * the request and promote the IR to WARM in the background.  If the prefetch
 * queue is full the request is silently dropped (the compiler thread will
 * do the disk load synchronously instead — correctness is preserved).
 *
 * @param cache    The IR cache.
 * @param func_id  ID of the function to prefetch.
 * @return true if the request was enqueued, false if the queue was full.
 */
bool ir_cache_prefetch(ir_lru_cache_t *cache, func_id_t func_id);

/**
 * Return the current generation of a function's IR (lock-free, may be stale
 * by one transition).
 */
uint8_t ir_cache_get_generation(const ir_lru_cache_t *cache, func_id_t func_id);

/**
 * Return the current memory-pressure level (lock-free atomic read).
 */
mem_pressure_t ir_cache_get_pressure(const ir_lru_cache_t *cache);

/* ═══════════════════════════ statistics ═══════════════════════════════════ */

/** Snapshot of IR-cache + memory-pressure statistics. */
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
    uint64_t pressure_evictions;
    mem_pressure_t pressure;          /**< Current pressure level.            */
    uint64_t mem_available_mb;        /**< Last observed MemAvailable (MB).   */
    uint64_t mem_total_mb;            /**< Last observed MemTotal (MB).       */
} ir_cache_stats_t;

/** Read a lock-free snapshot of all cache statistics. */
ir_cache_stats_t ir_cache_get_stats(const ir_lru_cache_t *cache);

/** Print a formatted statistics block to stderr. */
void ir_cache_print_stats(const ir_lru_cache_t *cache);
