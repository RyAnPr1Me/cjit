/**
 * codegen.h – Code-generation backend interface.
 *
 * The backend wraps the system C compiler (cc / gcc / clang) to perform
 * JIT compilation of C source strings into native shared objects that are
 * immediately loaded into the process image.
 *
 * Compilation pipeline
 * ────────────────────
 *
 *  1. Write the IR (C source) to a temporary file on disk.
 *     The file is unlinked immediately after compilation so that it does not
 *     linger in the filesystem.
 *
 *  2. Invoke the C compiler as a subprocess:
 *
 *       cc -shared -fPIC -O<N> [opt-flags] -o <output.so> <input.c>
 *
 *     The output path is a second temp file in /tmp; it is unlinked
 *     immediately after dlopen() so the kernel will reclaim disk space once
 *     the handle is closed.
 *
 *  3. dlopen the output shared object.
 *
 *  4. dlsym to locate the requested symbol (function name).
 *
 *  5. Return the function pointer and dlopen handle to the caller.
 *
 * Why not memfd?
 * ──────────────
 * memfd_create() gives an anonymous in-memory file for the OUTPUT, but most
 * standard C compilers do not accept /proc/self/fd/N as an output path in a
 * portable way.  Using a conventional temp file and unlinking after dlopen()
 * achieves equivalent semantics: the file's inode persists only as long as
 * the mapping does.
 *
 * Optimisation flags by tier
 * ──────────────────────────
 *   OPT_NONE : -O0
 *   OPT_O1   : -O1 -fomit-frame-pointer -fno-semantic-interposition
 *              -mtune=native
 *              -fno-plt            (Linux/ELF only)
 *   OPT_O2   : -O2 -fomit-frame-pointer -fno-semantic-interposition
 *              -mtune=native
 *              -fno-plt            (Linux/ELF only)
 *              -finline-functions -funroll-loops -ftree-vectorize
 *              -funswitch-loops    (hoist invariant conditionals out of loops)
 *              -fpeel-loops        (peel first/last iterations; unroll small
 *                                  known-trip-count loops to straight-line)
 *              -march=native        (if enable_native_arch)
 *   OPT_O3   : -O3 -fomit-frame-pointer -fno-semantic-interposition
 *              -mtune=native
 *              -fno-plt            (Linux/ELF only)
 *              -finline-functions -funroll-loops -ftree-vectorize
 *              -funswitch-loops
 *              -fpeel-loops
 *              -march=native        (if enable_native_arch)
 *              -ffast-math          (if enable_fast_math)
 *
 * Applied at every tier (including OPT_NONE):
 *   -fno-stack-protector        removes stack-canary overhead
 *   -fno-asynchronous-unwind-tables  omits .eh_frame (smaller .so,
 *                                    faster dlopen, better I-cache)
 *   -fno-ident                  omits .comment section (smaller .so,
 *                                    faster dlopen)
 *
 * Source file delivery
 * ────────────────────
 * On Linux, the C source string is written to an anonymous in-memory file
 * created with memfd_create(2) and passed to the compiler via its
 * /proc/self/fd/<fd> path.  This avoids any filesystem writes for the
 * source step: no tmpfs inode, no directory entry, no unlink.  On non-Linux
 * systems (or when memfd_create fails) a traditional /tmp tmpfile is used.
 *
 * Additionally the following hint macros are injected into every translation
 * unit at the top of the source:
 *
 *   #define LIKELY(x)                __builtin_expect(!!(x), 1)
 *   #define UNLIKELY(x)              __builtin_expect(!!(x), 0)
 *   #define HOT                      __attribute__((hot))
 *   #define COLD                     __attribute__((cold))
 *   #define NOINLINE                 __attribute__((noinline))
 *   #define ALWAYS_INLINE            __attribute__((always_inline)) inline
 *   #define PURE                     __attribute__((pure))
 *   #define CONST_FUNC               __attribute__((const))
 *   #define RESTRICT                 __restrict__
 *   #define PREFETCH(addr,rw,loc)    __builtin_prefetch(addr,rw,loc)
 *   #define ASSUME_ALIGNED(ptr,n)    __builtin_assume_aligned(ptr,n)
 *   #define FLATTEN                  __attribute__((flatten))
 *   #define NORETURN                 __attribute__((noreturn))
 *   #define CJIT_EXPORT              __attribute__((visibility("default")))
 *   #define MALLOC_FUNC              __attribute__((malloc))
 *
 * Also included automatically: <stdint.h>, <string.h>, <stdlib.h>,
 * <limits.h>, <stdbool.h>.
 *
 * These macros and headers let user IR code use branch-prediction, aliasing,
 * purity, inlining, and visibility hints portably without adding a compile-
 * time dependency on cjit.h.
 */

#pragma once

#include <stdbool.h>
#include "../include/cjit.h"  /* opt_level_t, jit_func_t */
#include "arg_profile.h"      /* cjit_arg_profile_t */

/* Forward declaration – full definition in codegen_cache.h. */
typedef struct codegen_cache codegen_cache_t;

/* ─────────────────────────── result type ───────────────────────────────────── */

/**
 * Result of one codegen_compile() invocation.
 */
typedef struct {
    jit_func_t  fn;        /**< Pointer to the compiled function (NULL on err). */
    void       *handle;    /**< dlopen handle; pass to dgc_retire() when done.  */
    bool        success;   /**< True iff compilation succeeded.                 */
    bool        timed_out; /**< True iff the compiler subprocess timed out.     */
    bool        cache_hit; /**< True iff the result came from the artifact cache.*/
    char        errmsg[4096]; /**< Human-readable error message on failure.     */
} codegen_result_t;

/**
 * Backend configuration flags (subset of cjit_config_t).
 */
typedef struct {
    bool enable_inlining;      /**< -finline-functions                          */
    bool enable_vectorization; /**< -ftree-vectorize                            */
    bool enable_loop_unroll;   /**< -funroll-loops                              */
    bool enable_native_arch;   /**< -march=native (only at OPT_O3)             */
    bool enable_fast_math;     /**< -ffast-math   (only at OPT_O3)             */
    bool verbose;              /**< Print compiler command to stderr            */

    /**
     * Optional argument profile for specialisation.
     *
     * When non-NULL and the profile contains at least one argument slot with
     * a confident dominant value, codegen_compile() prepends a specialised
     * wrapper to the user's source and renames the original function via a
     * -D preprocessor flag.  The wrapper fast-paths the common argument value
     * through an inlined call that GCC/Clang can constant-fold.
     *
     * NULL (the default) disables argument specialisation entirely.
     */
    const cjit_arg_profile_t  *arg_profile;

    /**
     * Extra compiler flags passed verbatim after all CJIT-generated flags.
     *
     * Space-separated tokens; may contain -I, -D, -l, -L, etc.
     * NULL or empty string → no extra flags.
     */
    const char *extra_cflags;

    /**
     * Compiler binary name or absolute path.
     *
     * NULL or empty string → "cc" found on PATH.
     */
    const char *cc_binary;

    /**
     * Optional compiled-artifact cache handle.
     *
     * When non-NULL, codegen_compile() checks the cache before spawning the
     * compiler.  On a hit the cached .so is dlopen'd directly and the compiler
     * subprocess is skipped entirely.  On a miss the compiled artifact is
     * stored in the cache for future hits.
     *
     * NULL disables caching for this compilation.
     */
    codegen_cache_t *cache;

    /**
     * Maximum wall-clock time (milliseconds) to wait for the compiler
     * subprocess to complete.
     *
     * When non-zero, codegen_compile() polls waitpid(WNOHANG) in a loop with
     * 10 ms sleeps.  If the process has not exited by the deadline, it is
     * sent SIGTERM (with a 50 ms grace period) followed by SIGKILL, and the
     * compilation is marked as timed out (result->timed_out = true).
     *
     * 0 = no timeout (default, backward-compatible).
     */
    uint32_t compile_timeout_ms;
} codegen_opts_t;

/* ─────────────────────────── API ───────────────────────────────────────────── */

/**
 * Compile a C source string into a callable function.
 *
 * @param func_name   The C symbol name to locate with dlsym.
 * @param c_source    Complete C source (may include headers, helpers, etc.).
 * @param level       Desired optimisation tier.
 * @param opts        Backend option flags.
 * @param result      Output: populated with function pointer + handle on success.
 *
 * @return true on success, false on any failure (compile error, dlopen error,
 *         symbol not found).  On failure, result->errmsg is populated.
 *
 * Thread safety: safe to call from multiple threads simultaneously.  Each
 * invocation uses uniquely-named temp files derived from the thread ID and a
 * monotonic counter.
 */
bool codegen_compile(const char          *func_name,
                     const char          *c_source,
                     opt_level_t          level,
                     const codegen_opts_t *opts,
                     codegen_result_t     *result);
