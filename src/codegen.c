/**
 * codegen.c – System-compiler JIT backend.
 *
 * See codegen.h for the design description.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "codegen.h"

#include <stdio.h>       /* FILE, fopen, fclose, snprintf, fprintf */
#include <stdlib.h>      /* free, system                           */
#include <string.h>      /* strlen, strncpy, snprintf              */
#include <stdatomic.h>   /* atomic_fetch_add                       */
#include <unistd.h>      /* unlink, getpid                         */
#include <pthread.h>     /* pthread_self                           */
#include <dlfcn.h>       /* dlopen, dlsym, dlerror                */
#include <sys/types.h>

/* ─────────────────────────── preamble injected before user IR ─────────────── */

/**
 * Helper macros injected at the top of every compiled translation unit.
 *
 * These are injected as raw text prepended to the user's C source so that
 * user code can use LIKELY/UNLIKELY/HOT/NOINLINE without depending on any
 * cjit header.
 *
 * __attribute__((hot)) instructs the compiler to optimise the function more
 * aggressively and place it in the "hot" section for better cache locality.
 */
static const char CODEGEN_PREAMBLE[] =
    "/* cjit auto-generated preamble */\n"
    "#ifndef CJIT_PREAMBLE_H\n"
    "#define CJIT_PREAMBLE_H\n"
    "#ifdef __GNUC__\n"
    "#  define LIKELY(x)                  __builtin_expect(!!(x), 1)\n"
    "#  define UNLIKELY(x)                __builtin_expect(!!(x), 0)\n"
    "#  define HOT                        __attribute__((hot))\n"
    "#  define COLD                       __attribute__((cold))\n"
    "#  define NOINLINE                   __attribute__((noinline))\n"
    "#  define ALWAYS_INLINE              __attribute__((always_inline)) inline\n"
    "#  define ALIGNED(n)                 __attribute__((aligned(n)))\n"
    /* Pure: reads global state but has no side effects; same args → same result. */
    "#  define PURE                       __attribute__((pure))\n"
    /* Const: like PURE but must not read global state – result depends only on args. */
    "#  define CONST_FUNC                 __attribute__((const))\n"
    /* Restrict: pointer aliasing hint – tells the compiler this pointer is unique. */
    "#  define RESTRICT                   __restrict__\n"
    /* Prefetch: software prefetch hint (rw: 0=read,1=write; locality: 0-3). */
    "#  define PREFETCH(addr, rw, loc)    __builtin_prefetch((addr), (rw), (loc))\n"
    /* Assume aligned: lets the compiler omit alignment-fixup code for SIMD. */
    "#  define ASSUME_ALIGNED(ptr, n)     __builtin_assume_aligned((ptr), (n))\n"
    "#else\n"
    "#  define LIKELY(x)                  (x)\n"
    "#  define UNLIKELY(x)                (x)\n"
    "#  define HOT\n"
    "#  define COLD\n"
    "#  define NOINLINE\n"
    "#  define ALWAYS_INLINE              inline\n"
    "#  define ALIGNED(n)\n"
    "#  define PURE\n"
    "#  define CONST_FUNC\n"
    "#  define RESTRICT\n"
    "#  define PREFETCH(addr, rw, loc)    ((void)(addr))\n"
    "#  define ASSUME_ALIGNED(ptr, n)     (ptr)\n"
    "#endif\n"
    "#include <stdint.h>\n"
    "#include <string.h>\n"
    "#include <stdlib.h>\n"
    "#endif /* CJIT_PREAMBLE_H */\n";

/* ─────────────────────────── optimisation flags ───────────────────────────── */

/**
 * Build the compiler flag string for the given optimisation tier.
 *
 * The resulting string is appended to the cc command after "-shared -fPIC".
 */
static void build_opt_flags(char *buf, size_t bufsz,
                             opt_level_t level, const codegen_opts_t *opts)
{
    int n = 0;

    /* Base optimisation level */
    switch (level) {
    case OPT_NONE: n += snprintf(buf + n, bufsz - n, " -O0"); break;
    case OPT_O1:   n += snprintf(buf + n, bufsz - n, " -O1"); break;
    case OPT_O2:   n += snprintf(buf + n, bufsz - n, " -O2"); break;
    case OPT_O3:   n += snprintf(buf + n, bufsz - n, " -O3"); break;
    default:       n += snprintf(buf + n, bufsz - n, " -O2"); break;
    }
    if (n < 0) n = 0;

    /* Optional flags controlled by codegen_opts_t */
    if (opts->enable_inlining && level >= OPT_O2)
        n += snprintf(buf + n, bufsz - n, " -finline-functions");

    if (opts->enable_loop_unroll && level >= OPT_O2)
        n += snprintf(buf + n, bufsz - n, " -funroll-loops");

    if (opts->enable_vectorization && level >= OPT_O2)
        n += snprintf(buf + n, bufsz - n, " -ftree-vectorize");

    if (level >= OPT_O2)
        n += snprintf(buf + n, bufsz - n, " -fomit-frame-pointer");

    /*
     * Disable ELF symbol interposition for shared-library symbols.
     * With -fPIC the compiler is otherwise forced to assume that any
     * call or global access might be redirected by LD_PRELOAD, which
     * prevents inlining of non-static functions and devirtualization.
     * Since each cjit .so is opened with RTLD_LOCAL and contains only
     * a single translation unit, interposition cannot happen and this
     * restriction is safe to lift.  The flag is available in GCC ≥ 8
     * and Clang ≥ 6.
     */
    if (level >= OPT_O2)
        n += snprintf(buf + n, bufsz - n, " -fno-semantic-interposition");

    /* march=native only at the most aggressive tier to avoid ABI issues */
    if (opts->enable_native_arch && level >= OPT_O3)
        n += snprintf(buf + n, bufsz - n, " -march=native");

    /*
     * Fast-math: allows reassociation and approximations that are not
     * IEEE-754 strictly conformant.  Enabled only when the caller opts in
     * and only at the most aggressive tier.
     */
    if (opts->enable_fast_math && level >= OPT_O3)
        n += snprintf(buf + n, bufsz - n, " -ffast-math");

    /* Always hide non-exported symbols to avoid clash across multiple .so.
     * We do NOT use -fvisibility=hidden here: each .so is dlopened with
     * RTLD_LOCAL and contains exactly one function; there is no collision
     * risk, and hiding symbols would make dlsym() fail to find the entry. */

    /* Suppress warnings – we may be compiling user snippets */
    n += snprintf(buf + n, bufsz - n, " -w");

    (void)n;
}

/* ─────────────────────────── unique temp-file naming ───────────────────────── */

static atomic_uint_fast64_t codegen_serial = 0;

/**
 * Generate a unique prefix for temp files of the form:
 *   /tmp/cjit_<pid>_<tid>_<serial>
 *
 * Using both pid and tid ensures uniqueness across processes AND threads.
 */
static void make_temp_prefix(char *buf, size_t sz)
{
    uint64_t serial = atomic_fetch_add_explicit(&codegen_serial, 1,
                                                memory_order_relaxed);
    snprintf(buf, sz, "/tmp/cjit_%d_%lu_%llu",
             (int)getpid(),
             (unsigned long)pthread_self(),
             (unsigned long long)serial);
}

/* ─────────────────────────── main compilation ──────────────────────────────── */

bool codegen_compile(const char          *func_name,
                     const char          *c_source,
                     opt_level_t          level,
                     const codegen_opts_t *opts,
                     codegen_result_t     *result)
{
    result->fn      = NULL;
    result->handle  = NULL;
    result->success = false;
    result->errmsg[0] = '\0';

    /* ── 1. Choose temp-file paths ─────────────────────────────────────── */
    char prefix[256];
    make_temp_prefix(prefix, sizeof(prefix));

    char src_path[300], so_path[300];
    snprintf(src_path, sizeof(src_path), "%s.c",  prefix);
    snprintf(so_path,  sizeof(so_path),  "%s.so", prefix);

    /* ── 2. Write preamble + user source to temp .c file ──────────────── */
    FILE *f = fopen(src_path, "w");
    if (!f) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: cannot create source file %s: (errno)", src_path);
        return false;
    }
    fputs(CODEGEN_PREAMBLE, f);
    fputs(c_source, f);
    fputc('\n', f);
    fclose(f);

    /* ── 3. Build compiler command ────────────────────────────────────── */
    char opt_flags[512] = {0};
    build_opt_flags(opt_flags, sizeof(opt_flags), level, opts);

    char cmd[2048];
    char err_path[304];
    snprintf(err_path, sizeof(err_path), "%s.err", prefix);
    snprintf(cmd, sizeof(cmd),
             "cc -shared -fPIC%s -o '%s' '%s' 2>'%s'",
             opt_flags, so_path, src_path, err_path);

    if (opts->verbose) {
        fprintf(stderr, "[cjit/codegen] compile: %s\n", cmd);
    }

    /* ── 4. Run the compiler ──────────────────────────────────────────── */
    int rc = system(cmd);

    /* Source file is no longer needed. */
    unlink(src_path);

    if (rc != 0) {
        /* Capture compiler error output from the per-invocation error file. */
        FILE *ef = fopen(err_path, "r");
        if (ef) {
            size_t nr = fread(result->errmsg,
                              1, sizeof(result->errmsg) - 1, ef);
            result->errmsg[nr] = '\0';
            fclose(ef);
        } else {
            snprintf(result->errmsg, sizeof(result->errmsg),
                     "codegen: cc exited with code %d", rc);
        }
        unlink(err_path);
        unlink(so_path);
        return false;
    }

    unlink(err_path);  /* remove even on success (may not exist if cc was quiet) */

    /* ── 5. dlopen the shared object ─────────────────────────────────── */
    void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: dlopen('%s'): %s", so_path, dlerror());
        unlink(so_path);
        return false;
    }

    /*
     * Unlink immediately after dlopen.
     * The kernel keeps the inode (and therefore the mapped pages) alive as
     * long as the dlopen handle is open.  This prevents temp-file leaks if
     * the process crashes.
     */
    unlink(so_path);

    /* ── 6. dlsym to find the function ───────────────────────────────── */
    dlerror(); /* clear any old error */
    void *sym = dlsym(handle, func_name);
    const char *err = dlerror();
    if (err) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: dlsym('%s'): %s", func_name, err);
        dlclose(handle);
        return false;
    }

    /* ── 7. Success ──────────────────────────────────────────────────── */
    /*
     * C99 / POSIX: converting between function and data pointers via void*
     * is implementation-defined, but dlsym() returning a function pointer
     * through void* is standard POSIX and works on all Linux/macOS targets.
     */
    result->fn      = (jit_func_t)(uintptr_t)sym;
    result->handle  = handle;
    result->success = true;

    if (opts->verbose) {
        fprintf(stderr, "[cjit/codegen] compiled '%s' at O%d → %p\n",
                func_name, (int)level, (void *)(uintptr_t)result->fn);
    }

    return true;
}
