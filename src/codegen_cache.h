/**
 * codegen_cache.h – Persistent compiled-artifact cache.
 *
 * Design overview
 * ───────────────
 * Stores compiled shared objects on disk, keyed by a 64-bit FNV-1a
 * content-address hash of:
 *
 *   preamble + wrapper_src + spec_define + ir_source
 *   + optimisation-level string + extra compiler flags + compiler binary name
 *
 * Cache hits skip the compiler subprocess entirely – the engine dlopen()s
 * the previously compiled .so directly from the cache directory.  This is
 * the dominant win for warm restarts and for functions whose IR does not
 * change across engine runs.
 *
 * Thread safety
 * ─────────────
 * All operations are safe to call concurrently from multiple threads and
 * even from multiple processes sharing the same cache directory.
 *
 * Cache stores use rename(2) into the final path, which is atomic on all
 * POSIX-compliant filesystems on the same device.  When the source .so is
 * on a different device (e.g., /dev/shm vs a persistent ~/.cache directory),
 * a copy+rename is performed so the final rename is still intra-device and
 * therefore atomic.
 *
 * Cache lookups use access(2) + dlopen(3).  The window between access() and
 * dlopen() is benign: the worst outcome is a spurious miss (file deleted
 * concurrently) or a hit on an unexpected entry (file created concurrently),
 * both of which are handled gracefully.
 *
 * Stale-entry safety
 * ──────────────────
 * If dlopen() fails on a cached path (e.g., wrong architecture, corrupted
 * file, or truncated write from a previous crash), the caller falls back to
 * a normal compilation.  The stale file remains in the cache until explicitly
 * cleaned; it will be overwritten by the next successful compilation of the
 * same hash key.
 *
 * Cache key collisions
 * ────────────────────
 * FNV-1a 64-bit gives a collision probability of ~2^-64 per pair of distinct
 * inputs.  For any realistic number of distinct JIT functions (< 2^20) the
 * probability of even a single collision across the lifetime of any program
 * is negligible.  A collision would result in one function silently using
 * another function's compiled code – undetectable but also astronomically
 * unlikely.  SHA-256 can be substituted if paranoia demands it.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────── opaque type ───────────────────────────────────── */

/** Opaque compiled-artifact cache handle. */
typedef struct codegen_cache codegen_cache_t;

/* ─────────────────────────── lifecycle ─────────────────────────────────────── */

/**
 * Create a cache rooted at `dir`.
 *
 * The directory is created (mode 0700, owner-only) if it does not exist.
 * Returns NULL on failure (dir is NULL/empty, mkdir fails with a fatal error,
 * or memory allocation fails).
 */
codegen_cache_t *codegen_cache_create(const char *dir);

/**
 * Destroy the cache handle.
 *
 * Releases the in-memory state only.  Cached files on disk are not deleted;
 * they persist for future engine instances to reuse.
 */
void codegen_cache_destroy(codegen_cache_t *c);

/* ─────────────────────────── key computation ───────────────────────────────── */

/**
 * Compute the 64-bit content-address key for a compilation unit.
 *
 * All inputs that affect the compiled binary must be included.  NULL and
 * empty-string inputs are each hashed as a single zero byte to distinguish
 * them from absent components while still contributing to the mix.
 *
 * @param preamble    Codegen preamble injected before user source (constant).
 * @param wrapper_src Arg-specialisation wrapper source, or NULL if none.
 * @param spec_define "#define func_name _cjit_i_func_name\n", or NULL/empty.
 * @param ir          User IR (C source string).
 * @param level_str   Optimisation-level string, e.g. "O2".
 * @param cflags      Extra compiler flags (may be NULL or empty).
 * @param cc_binary   Compiler binary name/path (NULL → treated as "cc").
 * @return 64-bit FNV-1a hash of all inputs.
 */
uint64_t codegen_cache_key(const char *preamble,
                            const char *wrapper_src,
                            const char *spec_define,
                            const char *ir,
                            const char *level_str,
                            const char *cflags,
                            const char *cc_binary);

/* ─────────────────────────── lookup / store ────────────────────────────────── */

/**
 * Look up a compiled artifact in the cache.
 *
 * Writes the full cache-file path into `out[0..outsz)` regardless of whether
 * the file exists.  This lets the caller use the same path for both the lookup
 * and a subsequent store.
 *
 * @param c     Cache handle.
 * @param key   Content-address key from codegen_cache_key().
 * @param out   Buffer to receive the cache-file path (≥ 512 bytes recommended).
 * @param outsz Size of `out`.
 * @return true if the file exists and is readable; false on a cache miss.
 */
bool codegen_cache_lookup(codegen_cache_t *c, uint64_t key,
                           char *out, size_t outsz);

/**
 * Absorb a compiled .so file into the cache.
 *
 * Attempts an atomic rename(2) of `so_path` into the cache directory.  On
 * a cross-device rename (EXDEV), falls back to copy-then-rename within the
 * cache directory.
 *
 * If the cache already contains an entry for `key` (concurrent compile race),
 * `so_path` is unlinked and the existing entry wins (identical output for
 * identical input).
 *
 * After a successful return the file at `so_path` no longer exists; its inode
 * lives on either as the new cache entry or (on a copy-path) has been unlinked
 * after copying.
 *
 * @param c       Cache handle.
 * @param key     Content-address key.
 * @param so_path Temporary .so to absorb (must be on a writable filesystem).
 * @return true on success (artifact is cached or a concurrent entry won);
 *         false on unexpected I/O failure.
 */
bool codegen_cache_store(codegen_cache_t *c, uint64_t key, const char *so_path);

/* ─────────────────────────── statistics ────────────────────────────────────── */

/** Number of cache hits (compilations skipped) since the handle was created. */
uint64_t codegen_cache_hits(const codegen_cache_t *c);

/** Number of cache misses (compilations performed) since the handle was created. */
uint64_t codegen_cache_misses(const codegen_cache_t *c);

#ifdef __cplusplus
}
#endif
