/**
 * codegen_cache.c – Persistent compiled-artifact cache implementation.
 *
 * See codegen_cache.h for the full design description.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "codegen_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

/* ─────────────────────────── constants ─────────────────────────────────────── */

/** Maximum length of the cache root directory path (including NUL). */
#define CC_DIR_MAX 256

/** Maximum length of a cache file path (CC_DIR_MAX + "/" + 16 hex + ".so" + NUL). */
#define CC_PATH_MAX (CC_DIR_MAX + 32)

/* ─────────────────────────── FNV-1a 64-bit hash ────────────────────────────── */

/*
 * FNV-1a (Fowler–Noll–Vo) 64-bit variant.
 *
 * Chosen for: simplicity, speed, and avalanche properties adequate for a
 * non-cryptographic content-address key.  The prime and offset are the
 * standard FNV-1a-64 constants.
 *
 * The hash is computed in a streaming fashion: callers can chain multiple
 * fnv1a_feed() calls with the accumulated state to hash composite values.
 */

#define FNV1A_PRIME_64   UINT64_C(1099511628211)
#define FNV1A_OFFSET_64  UINT64_C(14695981039346656037)

/* Update running hash with `len` bytes of `data`. */
static uint64_t fnv1a_feed(uint64_t h, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= FNV1A_PRIME_64;
    }
    return h;
}

/*
 * Hash a C string with normalisation: strips C-style block comments and
 * C++-style line comments, collapses runs of whitespace to a single space,
 * and strips leading and trailing whitespace — all in a single pass.
 *
 * This normalisation ensures that two IR strings that are semantically
 * identical (same tokens, same structure) but differ only in whitespace
 * or comments produce the same cache key, improving artifact-cache hit rates
 * for code that is regenerated with minor formatting differences.
 *
 * Implementation note: spaces are emitted lazily — a pending space is only
 * flushed to the hash when the next non-space character arrives.  This
 * naturally eliminates both leading and trailing whitespace.
 */
static uint64_t fnv1a_norm_ir(uint64_t h, const char *s)
{
    if (!s || !*s) {
        const uint8_t zero = 0;
        return fnv1a_feed(h, &zero, 1);
    }

    const char *p = s;
    bool pending_space = false; /* deferred space, not yet hashed */
    bool any_output    = false; /* suppress leading space */

    while (*p) {
        /* Skip C-style block comment. */
        if (p[0] == '/' && p[1] == '*') {
            p += 2;
            while (*p) {
                if (p[0] == '*' && p[1] == '/') { p += 2; break; }
                p++;
            }
            /* A comment acts as a whitespace separator. */
            if (any_output) pending_space = true;
            continue;
        }

        /* Skip C++ line comment. */
        if (p[0] == '/' && p[1] == '/') {
            p += 2;
            while (*p && *p != '\n') p++;
            if (*p == '\n') p++;
            if (any_output) pending_space = true;
            continue;
        }

        /* Whitespace: defer. */
        if (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            if (any_output) pending_space = true;
            p++;
            continue;
        }

        /* Normal character: flush pending space first, then hash the char. */
        if (pending_space) {
            const uint8_t sp = ' ';
            h = fnv1a_feed(h, &sp, 1);
            pending_space = false;
        }
        uint8_t c = (uint8_t)*p++;
        h = fnv1a_feed(h, &c, 1);
        any_output = true;
    }
    /* Trailing pending_space is intentionally discarded (strip trailing ws). */
    return h;
}

/*
 * Hash a C string.
 *
 * NULL and "" both contribute a single zero byte so that absent components
 * are distinguishable from each other when combined with a per-field separator.
 */
static uint64_t fnv1a_str(uint64_t h, const char *s)
{
    if (s && *s)
        return fnv1a_feed(h, s, strlen(s));
    /* Absent/empty: hash a single zero byte to distinguish from "nothing". */
    const uint8_t zero = 0;
    return fnv1a_feed(h, &zero, 1);
}

/* ─────────────────────────── struct ────────────────────────────────────────── */

struct codegen_cache {
    char                 dir[CC_DIR_MAX]; /**< Cache root (no trailing slash). */
    atomic_uint_fast64_t hits;            /**< Cache hits since creation.      */
    atomic_uint_fast64_t misses;          /**< Cache misses since creation.    */
};

/* ─────────────────────────── helpers ───────────────────────────────────────── */

/*
 * Write the canonical cache file path for `key` into out[0..outsz).
 *
 * Format: "<dir>/<16-hex-digit-key>.so"
 */
static void cc_make_path(const codegen_cache_t *c, uint64_t key,
                          char *out, size_t outsz)
{
    snprintf(out, outsz, "%s/%016llx.so", c->dir,
             (unsigned long long)key);
}

/*
 * Copy a file from `src` to `dst` using a buffered read-write loop.
 *
 * Returns true on success, false on any I/O error.  `dst` is unlinked on
 * failure to avoid leaving a partial file in the cache directory.
 */
static bool cc_copy_file(const char *src, const char *dst)
{
    int fd_src = open(src, O_RDONLY | O_CLOEXEC);
    if (fd_src < 0) return false;

    int fd_dst = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd_dst < 0) {
        close(fd_src);
        return false;
    }

    char buf[65536];
    ssize_t nr;
    bool ok = true;
    while ((nr = read(fd_src, buf, sizeof(buf))) > 0) {
        ssize_t nw = 0;
        while (nw < nr) {
            ssize_t w = write(fd_dst, buf + nw, (size_t)(nr - nw));
            if (w < 0) { ok = false; break; }
            nw += w;
        }
        if (!ok) break;
    }
    if (nr < 0) ok = false;

    close(fd_src);
    if (close(fd_dst) != 0) ok = false;
    if (!ok) unlink(dst);
    return ok;
}

/* ─────────────────────────── public API ────────────────────────────────────── */

codegen_cache_t *codegen_cache_create(const char *dir)
{
    if (!dir || !dir[0]) return NULL;

    codegen_cache_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;

    /* Copy dir, strip trailing slashes. */
    snprintf(c->dir, sizeof(c->dir), "%s", dir);
    size_t len = strlen(c->dir);
    while (len > 1 && c->dir[len - 1] == '/') c->dir[--len] = '\0';

    /* Create the directory if it does not already exist. */
    if (mkdir(c->dir, 0700) != 0 && errno != EEXIST) {
        free(c);
        return NULL;
    }

    atomic_init(&c->hits,   0);
    atomic_init(&c->misses, 0);
    return c;
}

void codegen_cache_destroy(codegen_cache_t *c)
{
    free(c);
}

uint64_t codegen_cache_key(const char *preamble,
                            const char *wrapper_src,
                            const char *spec_define,
                            const char *ir,
                            const char *level_str,
                            const char *cflags,
                            const char *cc_binary)
{
    /*
     * Separator byte injected between fields to prevent trivial length-extension
     * collisions where the concatenation of two distinct (a, b) pairs could
     * produce the same byte sequence.
     *
     * e.g., ("ab", "c") vs ("a", "bc") are different inputs with the same
     * naive concatenation "abc"; the separator makes them ("ab\xFF" "c") vs
     * ("a\xFF" "bc"), which have distinct byte sequences.
     */
    static const uint8_t SEP = 0xFF;

    uint64_t h = FNV1A_OFFSET_64;
    h = fnv1a_str(h, preamble);
    h = fnv1a_feed(h, &SEP, 1);
    h = fnv1a_str(h, wrapper_src);
    h = fnv1a_feed(h, &SEP, 1);
    h = fnv1a_str(h, spec_define);
    h = fnv1a_feed(h, &SEP, 1);
    /* Use normalised hashing for user IR: strip comments + collapse whitespace
     * so that trivially reformatted IR strings map to the same cache entry. */
    h = fnv1a_norm_ir(h, ir);
    h = fnv1a_feed(h, &SEP, 1);
    h = fnv1a_str(h, level_str);
    h = fnv1a_feed(h, &SEP, 1);
    h = fnv1a_str(h, cflags);
    h = fnv1a_feed(h, &SEP, 1);
    h = fnv1a_str(h, cc_binary ? cc_binary : "cc");
    return h;
}

bool codegen_cache_lookup(codegen_cache_t *c, uint64_t key,
                           char *out, size_t outsz)
{
    cc_make_path(c, key, out, outsz);

    if (access(out, R_OK) == 0) {
        atomic_fetch_add_explicit(&c->hits, 1, memory_order_relaxed);
        return true;
    }
    atomic_fetch_add_explicit(&c->misses, 1, memory_order_relaxed);
    return false;
}

bool codegen_cache_store(codegen_cache_t *c, uint64_t key, const char *so_path)
{
    char dest[CC_PATH_MAX];
    cc_make_path(c, key, dest, sizeof(dest));

    /* Fast path: same filesystem – atomic rename. */
    if (rename(so_path, dest) == 0)
        return true;

    if (errno == EEXIST) {
        /*
         * A concurrent compilation (same key) already stored the entry.
         * Both compilations produce identical output for identical input,
         * so the existing entry is authoritative.  Remove our copy.
         */
        unlink(so_path);
        return true;
    }

    if (errno == EXDEV) {
        /*
         * Cross-device rename: the compiled .so is on a different filesystem
         * than the cache directory (e.g., /dev/shm vs ~/.cache/cjit).
         *
         * Strategy: copy to a private temp name inside the cache dir, then
         * rename within the cache dir (same device → atomic).
         */
        char tmp[CC_PATH_MAX + 32];
        snprintf(tmp, sizeof(tmp), "%s/.tmp_%016llx_%d_%lu.so",
                 c->dir, (unsigned long long)key, (int)getpid(),
                 (unsigned long)pthread_self());

        if (!cc_copy_file(so_path, tmp)) {
            /* Copy failed; keep so_path intact for the caller to unlink. */
            return false;
        }

        /* Intra-cache rename – atomic on any single filesystem. */
        if (rename(tmp, dest) != 0) {
            if (errno == EEXIST) {
                /* Concurrent store won; our temp copy is redundant. */
                unlink(tmp);
            } else {
                unlink(tmp);
                unlink(so_path);
                return false;
            }
        }
        unlink(so_path);  /* original is no longer needed after copy */
        return true;
    }

    /* Unexpected error: leave so_path for the caller to handle. */
    return false;
}

uint64_t codegen_cache_hits(const codegen_cache_t *c)
{
    return atomic_load_explicit(&c->hits, memory_order_relaxed);
}

uint64_t codegen_cache_misses(const codegen_cache_t *c)
{
    return atomic_load_explicit(&c->misses, memory_order_relaxed);
}
