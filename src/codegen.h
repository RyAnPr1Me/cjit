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
 *   OPT_O1   : -O1
 *   OPT_O2   : -O2 -finline-functions
 *   OPT_O3   : -O3 -finline-functions -funroll-loops -ftree-vectorize
 *              -fomit-frame-pointer -march=native  (if enable_native_arch)
 *
 * Additionally the following hint macros are injected into every translation
 * unit at the top of the source:
 *
 *   #define LIKELY(x)   __builtin_expect(!!(x), 1)
 *   #define UNLIKELY(x) __builtin_expect(!!(x), 0)
 *   #define HOT         __attribute__((hot))
 *   #define NOINLINE    __attribute__((noinline))
 *
 * This lets user IR code use branch-prediction and inlining hints portably
 * without adding a compile-time dependency on cjit.h.
 */

#pragma once

#include <stdbool.h>
#include "../include/cjit.h"  /* opt_level_t, jit_func_t */

/* ─────────────────────────── result type ───────────────────────────────────── */

/**
 * Result of one codegen_compile() invocation.
 */
typedef struct {
    jit_func_t  fn;        /**< Pointer to the compiled function (NULL on err). */
    void       *handle;    /**< dlopen handle; pass to dgc_retire() when done.  */
    bool        success;   /**< True iff compilation succeeded.                 */
    char        errmsg[512]; /**< Human-readable error message on failure.      */
} codegen_result_t;

/**
 * Backend configuration flags (subset of cjit_config_t).
 */
typedef struct {
    bool enable_inlining;      /**< -finline-functions                          */
    bool enable_vectorization; /**< -ftree-vectorize                            */
    bool enable_loop_unroll;   /**< -funroll-loops                              */
    bool enable_native_arch;   /**< -march=native (only at OPT_O3)             */
    bool verbose;              /**< Print compiler command to stderr            */
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
