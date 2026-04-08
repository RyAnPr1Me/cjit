/**
 * codegen.c – System-compiler JIT backend.
 *
 * See codegen.h for the design description.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "codegen.h"
#include "codegen_cache.h"
#include "func_table.h"   /* CJIT_NAME_MAX */

#include <stdio.h>       /* FILE, fopen, fclose, snprintf, fprintf */
#include <stdlib.h>      /* free, system                           */
#include <string.h>      /* strlen, strncpy, snprintf              */
#include <stdatomic.h>   /* atomic_fetch_add                       */
#include <unistd.h>      /* unlink, getpid                         */
#include <pthread.h>     /* pthread_self                           */
#include <dlfcn.h>       /* dlopen, dlsym, dlerror                */
#include <sys/types.h>
#include <sys/stat.h>     /* stat, S_ISDIR                         */
#include <sys/wait.h>     /* waitpid           */
#include <spawn.h>        /* posix_spawnp      */
#include <fcntl.h>        /* O_WRONLY etc.     */
#ifdef __linux__
#include <sys/syscall.h>  /* SYS_memfd_create  */
#endif

/* ─────────────────────────── memfd helper (Linux) ─────────────────────────── */

#ifdef __linux__
/**
 * Thin wrapper around the memfd_create(2) kernel interface.
 *
 * Using a raw syscall avoids a glibc version dependency: the libc wrapper for
 * memfd_create was added in glibc 2.27 (2018), but the kernel syscall has
 * been available since Linux 3.17 (2014).  Both GCC and Clang are happy with
 * syscall() here.
 *
 * We do NOT set MFD_CLOEXEC because the file descriptor must remain open in
 * the compiler child process so that /proc/<ppid>/fd/<fd> continues to refer
 * to a valid, open file while cc reads the source.  The fd is closed
 * explicitly by codegen_compile() after waitpid() returns.
 *
 * Returns a valid file descriptor on success, or -1 on any failure (old
 * kernel, seccomp restriction, etc.).  Callers must check for -1 and fall
 * back to the ordinary /tmp tmpfile path.
 */
#  ifdef __NR_memfd_create
static int cg_memfd_create(const char *name)
{
    return (int)syscall(__NR_memfd_create, name, 0u);
}
#  else
/* Kernel too old to support memfd_create — fall back to tmpfile. */
static int cg_memfd_create(const char *name) { (void)name; return -1; }
#  endif /* __NR_memfd_create */
#endif /* __linux__ */

/* ─────────────────────────── output tmpdir selection ───────────────────────── */

/**
 * Return the best available temporary directory for .so output files.
 *
 * On Linux, /dev/shm is a RAM-backed tmpfs that avoids any real-disk I/O for
 * the compiled shared object.  On systems where /tmp is itself a tmpfs (most
 * modern Linux distributions) the difference is negligible, but on systems
 * with a disk-backed /tmp the saving is measurable: a typical 20-40 KB JIT
 * .so file avoids an inode create, a write, a dlopen (read), and an unlink —
 * all against a spinning disk instead of RAM.
 *
 * The probe is performed once and cached in an atomic flag to avoid repeated
 * stat(2) calls.  The benign race (two threads computing the same result
 * simultaneously) converges within one call.
 *
 * On non-Linux systems /tmp is always returned.
 */
static const char *cg_so_tmpdir(void)
{
#ifdef __linux__
    /* -1 = unknown, 0 = /tmp, 1 = /dev/shm */
    static _Atomic int shm_available = -1;
    int av = atomic_load_explicit(&shm_available, memory_order_relaxed);
    if (av < 0) {
        struct stat st;
        av = (stat("/dev/shm", &st) == 0 && S_ISDIR(st.st_mode)) ? 1 : 0;
        atomic_store_explicit(&shm_available, av, memory_order_relaxed);
    }
    if (av) return "/dev/shm";
#endif
    return "/tmp";
}



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
    /*
     * FLATTEN: forces the compiler to inline every function called from within
     * the annotated function, recursively.  More powerful than ALWAYS_INLINE
     * (which applies only to the immediate callee): a FLATTEN function becomes
     * a single block of straight-line code with no internal call overhead at
     * all.  Particularly useful for hot inner loops that call small helpers —
     * apply it to the outer loop function to guarantee the helpers disappear.
     */
    "#  define FLATTEN                    __attribute__((flatten))\n"
    /*
     * NORETURN: annotates a function guaranteed never to return (e.g. one that
     * calls exit() unconditionally or loops forever).  Allows the compiler to
     * elide the return sequence at call sites and remove dead code after the
     * call, reducing both binary size and branch overhead.
     */
    "#  define NORETURN                   __attribute__((noreturn))\n"
    /*
     * CJIT_EXPORT: explicitly marks a symbol as having default (exported)
     * visibility so it remains accessible via dlsym() even when the
     * translation unit is compiled with -fvisibility=hidden.  Without that
     * flag all symbols have default visibility anyway, so CJIT_EXPORT is a
     * no-op in normal builds.  It becomes useful when user IR passes
     * -fvisibility=hidden via cfg.extra_cflags to hide internal helpers and
     * only export the specific entry-point function the engine looks up.
     */
    "#  define CJIT_EXPORT                __attribute__((visibility(\"default\")))\n"
    /*
     * MALLOC_FUNC: asserts that the annotated function returns a freshly-
     * allocated pointer that is not aliased by any other live pointer in the
     * program.  Enables the alias analyser to eliminate many conservative
     * load/store assumptions when the caller uses the returned pointer,
     * allowing aggressive reordering and vectorisation of subsequent code.
     */
    "#  define MALLOC_FUNC                __attribute__((malloc))\n"
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
    "#  define FLATTEN\n"
    "#  define NORETURN\n"
    "#  define CJIT_EXPORT\n"
    "#  define MALLOC_FUNC\n"
    "#endif\n"
    "#include <stdint.h>\n"
    "#include <string.h>\n"
    "#include <stdlib.h>\n"
    /*
     * <limits.h>: provides INT_MAX, INT_MIN, LONG_MAX, etc.  Commonly
     * required for overflow guards and sentinel values in JIT functions.
     */
    "#include <limits.h>\n"
    /*
     * <stdbool.h>: provides bool, true, false in C99/C11.  In C23 these are
     * built-in keywords and the header is a no-op; including it is always safe.
     */
    "#include <stdbool.h>\n"
    "#endif /* CJIT_PREAMBLE_H */\n";

/* ─────────────────────────── optimisation flags ───────────────────────────── */

/**
 * Maximum number of arguments passed to the compiler subprocess.
 *
 * cc(1) + -shared + -fPIC + opt-level + up to 14 optional flags
 * (including -funswitch-loops, -fpeel-loops) +
 * JIT-specific flags (-fno-stack-protector, -fno-asynchronous-unwind-tables,
 * -fno-ident) + -w + -o + so_path + -x + c + src_path + NULL = at most 26
 * entries.  When argument specialisation is active, two more entries are
 * added (-D <func=_cjit_i_func>).  Extra user flags (extra_cflags) may add
 * up to ~50 more tokens.  96 gives comfortable headroom.
 */
#define MAX_CC_ARGS 96

/**
 * Build the compiler argv array for posix_spawnp().
 *
 * Fills argv[0..n] where argv[n] == NULL and returns n.
 * All string pointers are either string literals (stable) or the caller-
 * supplied so_path / src_path (valid for the duration of codegen_compile).
 */
static int build_compiler_argv(const char *argv[MAX_CC_ARGS],
                                const char *so_path,
                                const char *src_path,
                                opt_level_t level,
                                const codegen_opts_t *opts)
{
    int n = 0;
    argv[n++] = (opts && opts->cc_binary && opts->cc_binary[0])
                    ? opts->cc_binary : "cc";
    argv[n++] = "-shared";
    argv[n++] = "-fPIC";

    /* Base optimisation level */
    switch (level) {
    case OPT_NONE: argv[n++] = "-O0"; break;
    case OPT_O1:   argv[n++] = "-O1"; break;
    case OPT_O2:   argv[n++] = "-O2"; break;
    case OPT_O3:   argv[n++] = "-O3"; break;
    default:       argv[n++] = "-O2"; break;
    }

    /*
     * Flags applied from O1 upward.
     *
     * -fomit-frame-pointer: omit the frame pointer register save/restore in
     *   function prologues and epilogues, freeing an extra general-purpose
     *   register and reducing stack frame setup cost.  Always safe for JIT
     *   functions since we never need to unwind through them at the C level.
     *
     * -fno-semantic-interposition: assume that symbols in this translation
     *   unit cannot be interposed by LD_PRELOAD, allowing the compiler to
     *   inline and devirtualize calls to locally-defined functions.  Correct
     *   because every JIT .so is loaded with RTLD_LOCAL.
     */
    if (level >= OPT_O1) {
        argv[n++] = "-fomit-frame-pointer";
        argv[n++] = "-fno-semantic-interposition";

        /*
         * -mtune=native: schedule instructions for the CPU that is actually
         * running this JIT engine.  Unlike -march=native, this does NOT change
         * the instruction set (the binary is still baseline x86-64, ARM64,
         * etc.), so it is always safe to emit.  It improves throughput for all
         * JIT-compiled code by choosing pipeline-optimal instruction ordering,
         * register allocation hints, and branch alignment for the host µarch.
         */
        argv[n++] = "-mtune=native";

#ifdef __linux__
        /*
         * -fno-plt: replace PLT indirection stubs with direct GOT-relative
         * calls for any external functions called from within JIT code (e.g.
         * malloc, memcpy, libc helpers).
         *
         * Without this flag: call → PLT stub → GOT load → function  (2 hops)
         * With this flag:    call *GOT_entry@GOTPCREL(%rip)           (1 hop)
         *
         * The saving is one indirect-branch instruction and one potential
         * branch-target-buffer miss per unique external call site.  For
         * JIT functions that call many libc routines this is a measurable
         * win at zero correctness risk, because:
         *   (a) We already dlopen() with RTLD_NOW so all GOT entries are
         *       fully resolved before any JIT code executes.
         *   (b) We use RTLD_LOCAL so there is no inter-.so PLT sharing.
         *
         * Linux/ELF only.  Mach-O (macOS) uses a different ABI and does
         * not recognise this flag.  On non-Linux platforms the `#ifdef`
         * simply skips it — no error, just the optimisation is omitted.
         */
        argv[n++] = "-fno-plt";
#endif
    }

    if (opts->enable_inlining && level >= OPT_O2)
        argv[n++] = "-finline-functions";

    if (opts->enable_loop_unroll && level >= OPT_O2)
        argv[n++] = "-funroll-loops";

    if (opts->enable_vectorization && level >= OPT_O2)
        argv[n++] = "-ftree-vectorize";

    if (level >= OPT_O2) {
        /*
         * -funswitch-loops: hoist loop-invariant conditionals out of the loop
         * body.  When a loop contains `if (mode == 0) { … } else { … }` and
         * `mode` does not change across iterations, the compiler emits two
         * separate loops (one per branch) rather than re-testing the condition
         * on every iteration.  The result is fewer branch mispredictions, a
         * tighter inner loop, and better vectorisation opportunities.
         *
         * GCC enables this only at -O3 by default; we apply it at O2 as well
         * since JIT workloads very commonly contain dispatch-style inner loops
         * with invariant flag or mode parameters.
         */
        argv[n++] = "-funswitch-loops";

        /*
         * -fpeel-loops: peel the first (and last) iterations from loops where
         * the compiler has high confidence they execute very few times — either
         * from profile feedback or when the trip count is statically known and
         * small (e.g. a 3-iteration initialisation loop becomes straight-line
         * code).  Peeling the first iteration also often achieves better
         * alignment for the main loop body, reducing the cycle cost of the
         * hot path.
         *
         * GCC enables this only at -O3 by default; we apply it at O2 for the
         * same reasons as -funswitch-loops.
         */
        argv[n++] = "-fpeel-loops";
    }

    if (opts->enable_native_arch && level >= OPT_O3)
        argv[n++] = "-march=native";

    if (opts->enable_fast_math && level >= OPT_O3)
        argv[n++] = "-ffast-math";

    /*
     * JIT-specific flags applied at every optimisation level.
     *
     * -fno-stack-protector: disable the stack-smashing canary.  Every JIT
     *   function otherwise gets 2-4 extra instructions (load guard, compare,
     *   conditional branch) in its prologue and epilogue.  The canary is
     *   irrelevant for JIT code because the IR is trusted/controlled by the
     *   calling application.
     *
     * -fno-asynchronous-unwind-tables: omit the .eh_frame section from the
     *   compiled .so.  This section is required for C++ exception unwinding
     *   and debugger backtraces through JIT frames, neither of which applies
     *   here.  Removing it reduces .so binary size (often 20-30% for small
     *   functions), which lowers dlopen overhead and improves I-cache density.
     */
    argv[n++] = "-fno-stack-protector";
    argv[n++] = "-fno-asynchronous-unwind-tables";

    /*
     * -fno-ident: suppress the `.comment` ELF section that GCC and Clang
     * normally emit into every compiled object.  This section contains the
     * compiler version string (e.g. "GCC: (GNU) 13.2.0") and is never read
     * at runtime.  Removing it makes each JIT .so a few hundred bytes smaller
     * and reduces the amount of data that dlopen() must map and process.
     */
    argv[n++] = "-fno-ident";

    /*
     * -pipe: instruct the compiler to use pipes instead of temporary files for
     * communication between compilation stages (preprocessor → compiler →
     * assembler).  This eliminates multiple temporary-file round-trips through
     * the filesystem for every JIT compilation and is particularly beneficial
     * on systems where /tmp is disk-backed.  The flag is safe and well-supported
     * by both GCC and Clang; it has no effect on the correctness or semantics
     * of the compiled output.
     */
    argv[n++] = "-pipe";

    argv[n++] = "-w";   /* suppress warnings from user snippets */
    argv[n++] = "-o";
    argv[n++] = so_path;
    /*
     * Explicit language specification before the source path.
     *
     * When the source file is an in-memory (memfd) path such as
     * /proc/self/fd/3, the compiler cannot infer "C source" from the file
     * extension.  Adding "-x c" before the path is safe even when the file
     * has a ".c" extension — it is redundant but harmless in that case, so
     * we always emit it to keep the logic unconditional.
     */
    argv[n++] = "-x";
    argv[n++] = "c";
    argv[n++] = src_path;
    argv[n]   = NULL;
    return n;
}

/* ─────────────── argument-profile specialisation helpers ──────────────────── */

/*
 * Per-parameter info extracted from a parsed C function signature.
 *
 * Limits: 64 bytes for the type string + 32 for the name — sufficient for any
 * realistic JIT parameter.  Oversized tokens are truncated; in the worst case
 * the wrapping step simply decides not to specialise that slot.
 */
#define SPEC_MAX_PARAMS   CJIT_MAX_PROFILED_ARGS   /* at most 8 params tracked */
#define SPEC_TYPE_MAX     80
#define SPEC_NAME_MAX     40

typedef struct {
    char type[SPEC_TYPE_MAX];
    char name[SPEC_NAME_MAX];
    bool valid;
} param_info_t;

/*
 * Copy up to `n` non-whitespace-leading, non-whitespace-trailing characters
 * of [begin, end) into `out` (NUL-terminated).
 */
static void copy_trimmed(char *out, size_t outsz, const char *begin, const char *end)
{
    while (begin < end && (*begin == ' ' || *begin == '\t' || *begin == '\n' || *begin == '\r'))
        begin++;
    while (end > begin && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r'))
        end--;
    size_t len = (size_t)(end - begin);
    if (len >= outsz) len = outsz - 1;
    memcpy(out, begin, len);
    out[len] = '\0';
}

/*
 * Return true if the type string is a plain integer type (suitable for
 * constant specialisation).  We look for known integer keywords and reject
 * anything containing '*' (pointer), "float", "double", or "void".
 *
 * This is intentionally conservative: it is safer to miss a specialisation
 * opportunity than to produce a specialisation for a non-integer argument.
 */
static bool is_integer_type(const char *type)
{
    if (strstr(type, "*"))      return false;
    if (strstr(type, "float"))  return false;
    if (strstr(type, "double")) return false;
    if (strstr(type, "void"))   return false;
    /* Accept any of the standard integer-like keywords */
    if (strstr(type, "int"))    return true;
    if (strstr(type, "long"))   return true;
    if (strstr(type, "short"))  return true;
    if (strstr(type, "char"))   return true;
    if (strstr(type, "size_t")) return true;
    if (strstr(type, "uint"))   return true;
    if (strstr(type, "int8"))   return true;
    if (strstr(type, "int16"))  return true;
    if (strstr(type, "int32"))  return true;
    if (strstr(type, "int64"))  return true;
    if (strstr(type, "bool"))   return true;
    return false;
}

/*
 * Parse a comma-separated parameter list of the form:
 *   "int a, const float *b, unsigned long c"
 * into an array of param_info_t structs.
 *
 * Returns the number of params successfully parsed (≥ 0).
 * Stops at SPEC_MAX_PARAMS even if the source has more.
 * A parameter with an unnamed argument (e.g. "int") sets .valid = false.
 */
static int parse_param_list(const char *params, param_info_t *out, int max_out)
{
    int count = 0;
    const char *p = params;
    while (*p && count < max_out) {
        /* Find end of this parameter (next comma at depth 0) */
        const char *start = p;
        int depth = 0;
        while (*p && !(*p == ',' && depth == 0)) {
            if (*p == '(') depth++;
            else if (*p == ')') { if (depth > 0) depth--; else break; }
            p++;
        }
        const char *end = p;
        if (*p == ',') p++; /* consume comma */

        /* Trim the parameter token */
        while (start < end && (*start == ' ' || *start == '\t')) start++;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) end--;
        if (start == end) continue;  /* empty token */

        /*
         * Split "type name" at the last identifier boundary.
         * Scan backwards for the end of the parameter name.
         */
        const char *name_end = end;
        /* Skip trailing [] if any (array params) */
        while (name_end > start && name_end[-1] == ']') {
            while (name_end > start && name_end[-1] != '[') name_end--;
            if (name_end > start) name_end--;  /* skip '[' */
        }
        const char *name_start = name_end;
        while (name_start > start &&
               (*(name_start-1) == '_' ||
                (*(name_start-1) >= 'a' && *(name_start-1) <= 'z') ||
                (*(name_start-1) >= 'A' && *(name_start-1) <= 'Z') ||
                (*(name_start-1) >= '0' && *(name_start-1) <= '9')))
            name_start--;

        if (name_start == start) {
            /* No separate name found — either "int" with no name, or parsing
             * edge case.  Mark as invalid but still count the slot. */
            out[count].valid = false;
            out[count].type[0] = out[count].name[0] = '\0';
            count++;
            continue;
        }

        copy_trimmed(out[count].type, sizeof(out[count].type), start, name_start);
        copy_trimmed(out[count].name, sizeof(out[count].name), name_start, name_end);
        out[count].valid = (out[count].type[0] != '\0' && out[count].name[0] != '\0');
        count++;
    }
    return count;
}

/*
 * Parse the function signature from the C source text.
 *
 * Searches for:   <return_type> <func_name> ( <params> ) {
 * Extracts return type (ret_buf) and parameter list string (par_buf).
 *
 * Returns true on success.  False is returned (and buffers left empty) if:
 *   • func_name is not found in ir_source
 *   • the signature cannot be reliably parsed
 *   • func_name appears too many times (risk of -D substitution side effects)
 */
static bool parse_func_sig(const char *ir, const char *func_name,
                            char *ret_buf, size_t ret_sz,
                            char *par_buf, size_t par_sz)
{
    ret_buf[0] = par_buf[0] = '\0';

    /* Count occurrences of func_name as a whole identifier to detect
     * ambiguous sources (e.g. multiple definitions, recursive helpers). */
    size_t fn_len = strlen(func_name);
    unsigned occurrences = 0;
    const char *scan = ir;
    while ((scan = strstr(scan, func_name)) != NULL) {
        /* Check word boundaries to avoid matching substrings. */
        bool lbound = (scan == ir ||
                       !(*(scan-1) == '_' ||
                         (*(scan-1) >= 'a' && *(scan-1) <= 'z') ||
                         (*(scan-1) >= 'A' && *(scan-1) <= 'Z') ||
                         (*(scan-1) >= '0' && *(scan-1) <= '9')));
        const char *after = scan + fn_len;
        bool rbound = (!(*after == '_' ||
                         (*after >= 'a' && *after <= 'z') ||
                         (*after >= 'A' && *after <= 'Z') ||
                         (*after >= '0' && *after <= '9')));
        if (lbound && rbound) occurrences++;
        scan += fn_len;
    }
    if (occurrences == 0 || occurrences > CJIT_SPEC_MAX_NAME_OCCURRENCES)
        return false;

    /* Find the function DEFINITION: func_name immediately followed by '('. */
    const char *def = ir;
    while ((def = strstr(def, func_name)) != NULL) {
        /* Check left word boundary */
        bool lbound = (def == ir ||
                       !(*(def-1) == '_' ||
                         (*(def-1) >= 'a' && *(def-1) <= 'z') ||
                         (*(def-1) >= 'A' && *(def-1) <= 'Z') ||
                         (*(def-1) >= '0' && *(def-1) <= '9')));
        const char *after = def + fn_len;
        /* Skip whitespace to find '(' */
        while (*after == ' ' || *after == '\t' || *after == '\n') after++;
        if (lbound && *after == '(') break;
        def += fn_len;
    }
    if (!def) return false;

    /* Extract return type: text on the same line before func_name, going back
     * past whitespace to the previous newline or start-of-file. */
    const char *line_start = def;
    while (line_start > ir && *(line_start-1) != '\n') line_start--;
    /* Skip leading whitespace / storage class keywords we don't want to copy */
    while (*line_start == ' ' || *line_start == '\t') line_start++;
    /* The return type is everything from line_start to def, trimmed. */
    copy_trimmed(ret_buf, ret_sz, line_start, def);
    if (ret_buf[0] == '\0') return false;  /* couldn't find return type */

    /* Extract parameter list between '(' and matching ')'. */
    const char *after = def + fn_len;
    while (*after == ' ' || *after == '\t' || *after == '\n') after++;
    if (*after != '(') return false;
    after++; /* skip '(' */

    const char *par_start = after;
    int depth = 1;
    while (*after && depth > 0) {
        if (*after == '(') depth++;
        else if (*after == ')') depth--;
        after++;
    }
    if (depth != 0) return false;
    const char *par_end = after - 1; /* points just after ')' → back up one */

    copy_trimmed(par_buf, par_sz, par_start, par_end);

    /* Verify there is a '{' following (it's a definition, not a declaration). */
    const char *body = after;
    while (*body == ' ' || *body == '\t' || *body == '\n' || *body == '\r') body++;
    return (*body == '{');
}

/*
 * Generate a specialisation wrapper for func_name.
 *
 * Writes the generated C source into out_buf.
 * Sets spec_define (e.g. "my_func=_cjit_i_my_func") if a -D flag is needed.
 *
 * Returns number of bytes written, 0 if no specialisation is possible.
 */
static size_t generate_spec_wrapper(const char *func_name,
                                     const char *ret_type,
                                     const char *params_str,
                                     const cjit_arg_profile_t *prof,
                                     char *out_buf, size_t out_sz,
                                     char *spec_define_buf, size_t define_sz)
{
    spec_define_buf[0] = '\0';

    /* Parse the parameter list. */
    param_info_t pinfo[SPEC_MAX_PARAMS];
    memset(pinfo, 0, sizeof(pinfo));
    int nparams = parse_param_list(params_str, pinfo, SPEC_MAX_PARAMS);
    if (nparams <= 0) return 0;

    /* Identify which params have a confident dominant integer value. */
    bool specialise[SPEC_MAX_PARAMS];
    memset(specialise, 0, sizeof(specialise));
    bool any = false;
    int n_to_check = nparams < (int)prof->n_profiled ? nparams : (int)prof->n_profiled;
    for (int i = 0; i < n_to_check; ++i) {
        if (!pinfo[i].valid) continue;
        if (!is_integer_type(pinfo[i].type)) continue;
        if (!cjit_arg_slot_confident(&prof->slots[i])) continue;
        specialise[i] = true;
        any = true;
    }
    if (!any) return 0;

    /* Build the internal symbol name: _cjit_i_<func_name> (max CJIT_NAME_MAX). */
    char internal_name[CJIT_NAME_MAX + 10];
    snprintf(internal_name, sizeof(internal_name), "_cjit_i_%s", func_name);

    /*
     * Build a preprocessor directive to inject BETWEEN the wrapper and the
     * user's source.  This renames `func_name` to `internal_name` only for
     * the text that follows (i.e. the user's implementation), leaving the
     * wrapper above the directive completely unaffected.
     *
     * Using an inline `#define` rather than a `-D` command-line flag avoids
     * the problem where the `-D` would rename occurrences of `func_name`
     * inside the wrapper itself (causing a redefinition error).
     */
    snprintf(spec_define_buf, define_sz,
             "#define %s %s\n", func_name, internal_name);

    /*
     * Build the parameter string for the forward declaration and call site
     * using offset-tracked snprintf() writes — O(n), no repeated strlen().
     */
    char   forward_params[512];
    char   call_args_generic[512];
    char   call_args_const[512];
    char   cond_expr[256];
    size_t fp  = 0;  /* offset into forward_params   */
    size_t cag = 0;  /* offset into call_args_generic */
    size_t cac = 0;  /* offset into call_args_const   */
    size_t ce  = 0;  /* offset into cond_expr         */

#define APPEND(buf, pos, ...) \
    do { \
        int _n = snprintf((buf) + (pos), sizeof(buf) - (pos), __VA_ARGS__); \
        if (_n > 0) (pos) += (size_t)_n; \
        if ((pos) >= sizeof(buf)) (pos) = sizeof(buf) - 1; \
    } while (0)

    bool first_cond = true;
    for (int i = 0; i < nparams; ++i) {
        if (i > 0) {
            APPEND(forward_params,    fp,  ", ");
            APPEND(call_args_generic, cag, ", ");
            APPEND(call_args_const,   cac, ", ");
        }
        /* Forward declaration: "type name" */
        APPEND(forward_params, fp, "%s %s", pinfo[i].type, pinfo[i].name);
        /* Generic call: just the name */
        APPEND(call_args_generic, cag, "%s", pinfo[i].name);
        /* Specialised call: constant literal for hot slots, name otherwise */
        if (specialise[i]) {
            APPEND(call_args_const, cac, "(%s)%lld",
                   pinfo[i].type,
                   (long long)(int64_t)prof->slots[i].dominant_val);
            APPEND(cond_expr, ce, "%s%s == (%s)%lld",
                   first_cond ? "" : " && ",
                   pinfo[i].name,
                   pinfo[i].type,
                   (long long)(int64_t)prof->slots[i].dominant_val);
            first_cond = false;
        } else {
            APPEND(call_args_const, cac, "%s", pinfo[i].name);
        }
    }
#undef APPEND

    /* bool is_void_return */
    bool is_void = (strncmp(ret_type, "void", 4) == 0 &&
                    (ret_type[4] == '\0' || ret_type[4] == ' '));

    /* Emit the wrapper. */
    int written;
    if (is_void) {
        written = snprintf(out_buf, out_sz,
            "/* [JIT arg-specialisation wrapper: %s] */\n"
            "%s %s(%s);\n"  /* forward decl of internal name using parsed params */
            "%s %s(%s) {\n"
            "    if (__builtin_expect(%s, 1)) { %s(%s); return; }\n"
            "    %s(%s);\n"
            "}\n",
            func_name,
            ret_type, internal_name, forward_params,
            ret_type, func_name,     params_str,
            cond_expr, internal_name, call_args_const,
            internal_name, call_args_generic);
    } else {
        written = snprintf(out_buf, out_sz,
            "/* [JIT arg-specialisation wrapper: %s] */\n"
            "%s %s(%s);\n"  /* forward decl of internal name using parsed params */
            "%s %s(%s) {\n"
            "    if (__builtin_expect(%s, 1)) return %s(%s);\n"
            "    return %s(%s);\n"
            "}\n",
            func_name,
            ret_type, internal_name, forward_params,
            ret_type, func_name,     params_str,
            cond_expr, internal_name, call_args_const,
            internal_name, call_args_generic);
    }
    return (written > 0 && (size_t)written < out_sz) ? (size_t)written : 0;
}

/* ─────────────────────────── unique temp-file naming ───────────────────────── */

static atomic_uint_fast64_t codegen_serial = 0;

/**
 * Generate a unique prefix for source temp files (src, err) of the form:
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

/**
 * Generate a unique path for the compiled .so output file.
 *
 * Uses /dev/shm when available on Linux (RAM-backed tmpfs, zero disk I/O).
 * Falls back to /tmp on any other platform or when /dev/shm is unavailable.
 *
 * The serial counter is shared with make_temp_prefix() so each compilation
 * invocation produces a consistent pair of uniquely numbered temp paths.
 */
static void make_so_path(char *buf, size_t sz)
{
    uint64_t serial = atomic_fetch_add_explicit(&codegen_serial, 1,
                                                memory_order_relaxed);
    snprintf(buf, sz, "%s/cjit_%d_%lu_%llu.so",
             cg_so_tmpdir(),
             (int)getpid(),
             (unsigned long)pthread_self(),
             (unsigned long long)serial);
}

/* ─────────────────────────── level-to-string helper ───────────────────────── */

static const char *level_to_str(opt_level_t level)
{
    switch (level) {
    case OPT_NONE: return "O0";
    case OPT_O1:   return "O1";
    case OPT_O2:   return "O2";
    case OPT_O3:   return "O3";
    default:       return "O2";
    }
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

    /* ── 1. Optionally generate an argument-specialisation wrapper ─────── */
    /*
     * The wrapper must be generated BEFORE the cache check so that the cache
     * key includes the wrapper text.  Specialised and unspecialised
     * compilations for the same IR at the same level produce different
     * binaries; they must have different cache entries.
     *
     * If opts->arg_profile is set and parse_func_sig() finds a confident
     * dominant value on any integer-typed parameter, we:
     *   (a) prepend a specialised wrapper that fast-paths the common value;
     *   (b) inject a `#define func_name _cjit_i_func_name` directive between
     *       the wrapper and the user's source, so only the user's implementation
     *       is renamed.  Using an inline #define (rather than a -D command-line
     *       flag) ensures the wrapper text above the directive is unaffected —
     *       a -D flag would rename func_name everywhere in the file, including
     *       in the wrapper itself, causing a redefinition error.
     *
     * With -finline-functions (O2+) the compiler inlines the renamed helper
     * into the hot path, constant-folds through the dominant argument, and
     * eliminates dead branches.
     *
     * On parse failure or when no confident slot is found, wrapper_src stays
     * NULL and we fall through to the ordinary (unspecialised) path.
     */
    char *wrapper_src    = NULL;   /* heap-allocated on success */
    /* spec_define holds the `#define name internal\n` text to inject. */
    char  spec_define[CJIT_NAME_MAX * 2 + 16];
    spec_define[0] = '\0';

    if (opts->arg_profile && opts->arg_profile->n_profiled > 0 &&
        level >= OPT_O2) {
        char ret_type[128], params_str[512];
        if (parse_func_sig(c_source, func_name,
                           ret_type, sizeof(ret_type),
                           params_str, sizeof(params_str))) {
            /* Allocate a generous buffer; typical wrapper is < 512 bytes. */
            size_t wsz = strlen(c_source) + 1024;
            char  *wbuf = malloc(wsz);
            if (wbuf) {
                size_t wlen = generate_spec_wrapper(
                    func_name, ret_type, params_str,
                    opts->arg_profile, wbuf, wsz,
                    spec_define, sizeof(spec_define));
                if (wlen > 0) {
                    wrapper_src = wbuf;
                    if (opts->verbose)
                        fprintf(stderr,
                                "[cjit/codegen] specialising '%s': %s\n",
                                func_name, spec_define);
                } else {
                    free(wbuf);
                }
            }
        }
    }

    /* ── 2. Compiled-artifact cache lookup ────────────────────────────── */
    /*
     * Check whether this exact compilation (same IR, same preamble, same
     * flags, same level) has been compiled before.  On a hit we dlopen the
     * cached .so directly, skipping the compiler spawn entirely.
     *
     * The cache key is computed from all inputs that affect the compiled
     * binary: preamble, specialisation wrapper + define, user IR, opt level,
     * extra compiler flags, and compiler binary name.  Two compilations that
     * differ in any of these fields produce different keys and never collide.
     */
    char     cache_path[512];
    uint64_t cache_key = 0;
    bool     have_cache = (opts && opts->cache != NULL);

    if (have_cache) {
        const char *cc = (opts->cc_binary && opts->cc_binary[0])
                             ? opts->cc_binary : "cc";
        cache_key = codegen_cache_key(
            CODEGEN_PREAMBLE,
            wrapper_src,
            spec_define[0] ? spec_define : NULL,
            c_source,
            level_to_str(level),
            (opts->extra_cflags && opts->extra_cflags[0]) ? opts->extra_cflags : NULL,
            cc);

        if (codegen_cache_lookup(opts->cache, cache_key,
                                  cache_path, sizeof(cache_path))) {
            /*
             * Cache hit: try to dlopen directly from the cached path.
             *
             * If dlopen fails (stale entry, wrong architecture, file removed
             * between access() and dlopen()), fall through to normal
             * compilation.  The stale file will be overwritten by the next
             * successful store of the same key.
             */
            dlerror();
            void *handle = dlopen(cache_path, RTLD_NOW | RTLD_LOCAL);
            if (handle) {
                dlerror();
                void *sym = dlsym(handle, func_name);
                const char *err = dlerror();
                if (!err && sym) {
                    free(wrapper_src);
                    result->fn      = (jit_func_t)(uintptr_t)sym;
                    result->handle  = handle;
                    result->success = true;
                    if (opts->verbose)
                        fprintf(stderr,
                                "[cjit/codegen] cache hit '%s' O%d → %p\n",
                                func_name, (int)level,
                                (void *)(uintptr_t)result->fn);
                    return true;
                }
                /* Symbol not found in cached .so (unlikely). */
                dlclose(handle);
            }
            /* dlopen failed: log and fall through to recompile. */
            if (opts->verbose)
                fprintf(stderr,
                        "[cjit/codegen] cache hit stale for '%s' O%d, recompiling\n",
                        func_name, (int)level);
        }
    }

    /* ── 3. Choose temp-file paths ─────────────────────────────────────── */
    char prefix[256];
    make_temp_prefix(prefix, sizeof(prefix));

    char so_path[300];
    make_so_path(so_path, sizeof(so_path));

    /* ── 4. Write preamble [+ wrapper] + user source ─────────────────── */
    /*
     * On Linux, write the source into an anonymous in-memory file created
     * with memfd_create(2).  The file is exposed to the compiler via its
     * /proc/self/fd/<fd> symlink, eliminating all filesystem writes for the
     * source step (no inode creation, no unlink, no tmpfs I/O).
     *
     * memfd_create is a raw syscall (no glibc dependency) that has been
     * present since Linux 3.17 (2014).  On failure (very old kernel or
     * seccomp restriction) we fall back to the traditional /tmp tmpfile.
     *
     * On non-Linux systems the traditional tmpfile path is used directly.
     */
    char src_path[300];
    bool src_is_memfd = false;
    int  src_fd = -1;

#ifdef __linux__
    src_fd = cg_memfd_create("cjit_src");
    if (src_fd >= 0) {
        /*
         * Use /proc/<parent_pid>/fd/<fd> rather than /proc/self/fd/<fd>.
         *
         * /proc/self resolves in the CALLING process's context.  When the
         * compiler subprocess opens the path, "self" refers to the compiler
         * process, which has no fd <src_fd>.  Using the parent's PID
         * (/proc/<ppid>/fd/<fd>) makes the path valid from any process that
         * has permission to inspect the parent — which the compiler child
         * always does (same UID, no seccomp restriction on /proc reads).
         */
        snprintf(src_path, sizeof(src_path), "/proc/%d/fd/%d",
                 (int)getpid(), src_fd);
        src_is_memfd = true;
        size_t plen = strlen(CODEGEN_PREAMBLE);
        size_t slen = strlen(c_source);
        bool write_ok =
            ((size_t)write(src_fd, CODEGEN_PREAMBLE, plen) == plen);
        if (write_ok && wrapper_src) {
            size_t wlen = strlen(wrapper_src);
            write_ok = ((size_t)write(src_fd, wrapper_src, wlen) == wlen);
        }
        /* Inject the `#define func_name _cjit_i_func_name` between wrapper
         * and user source so only the implementation is renamed. */
        if (write_ok && spec_define[0] != '\0') {
            size_t dlen = strlen(spec_define);
            write_ok = ((size_t)write(src_fd, spec_define, dlen) == dlen);
        }
        if (write_ok)
            write_ok = ((size_t)write(src_fd, c_source, slen) == slen &&
                        write(src_fd, "\n", 1) == 1);
        if (!write_ok) {
            snprintf(result->errmsg, sizeof(result->errmsg),
                     "codegen: write to in-memory source file failed");
            close(src_fd);
            free(wrapper_src);
            return false;
        }
    }
#endif /* __linux__ */

    if (!src_is_memfd) {
        /* Traditional /tmp tmpfile fallback. */
        snprintf(src_path, sizeof(src_path), "%s.c", prefix);
        FILE *f = fopen(src_path, "w");
        if (!f) {
            snprintf(result->errmsg, sizeof(result->errmsg),
                     "codegen: cannot create source file %s", src_path);
            free(wrapper_src);
            return false;
        }
        fputs(CODEGEN_PREAMBLE, f);
        if (wrapper_src) fputs(wrapper_src, f);
        /* Inject the `#define func_name _cjit_i_func_name` between wrapper
         * and user source so only the implementation is renamed. */
        if (spec_define[0] != '\0') fputs(spec_define, f);
        fputs(c_source, f);
        fputc('\n', f);
        fclose(f);
    }

    /* ── 5. Build compiler argv ──────────────────────────────────────── */
    const char *cc_argv[MAX_CC_ARGS];
    char err_path[304];
    snprintf(err_path, sizeof(err_path), "%s.err", prefix);
    int n_argv = build_compiler_argv(cc_argv, so_path, src_path, level, opts);

    /* Append extra_cflags tokens (e.g. -I/path, -DFOO, -lm).
     * strtok_r is used (not strtok) because codegen_compile() may be invoked
     * concurrently from multiple compiler threads; strtok's internal static
     * state is not thread-safe.
     * Tokens point into extra_flags_buf (stack) and are valid until after
     * waitpid completes. */
    char extra_flags_buf[CJIT_MAX_EXTRA_CFLAGS];
    extra_flags_buf[0] = '\0';
    if (opts && opts->extra_cflags && opts->extra_cflags[0]) {
        strncpy(extra_flags_buf, opts->extra_cflags, sizeof(extra_flags_buf) - 1);
        extra_flags_buf[sizeof(extra_flags_buf) - 1] = '\0';
        char *saveptr = NULL;
        char *tok = strtok_r(extra_flags_buf, " \t", &saveptr);
        while (tok && n_argv < MAX_CC_ARGS - 1) {
            cc_argv[n_argv++] = tok;
            tok = strtok_r(NULL, " \t", &saveptr);
        }
    }
    cc_argv[n_argv] = NULL;  /* always ensure NULL-terminator for posix_spawnp */

    /* Determine which compiler binary to invoke. */
    const char *cc_bin = (opts && opts->cc_binary && opts->cc_binary[0])
                             ? opts->cc_binary : "cc";

    if (opts && opts->verbose) {
        fprintf(stderr, "[cjit/codegen] compile:");
        for (int _i = 0; cc_argv[_i]; ++_i)
            fprintf(stderr, " %s", cc_argv[_i]);
        fprintf(stderr, " 2>%s\n", err_path);
    }

    /* ── 6. Spawn the compiler (no shell intermediate) ────────────────── */
    /*
     * posix_spawnp avoids the /bin/sh intermediate step that system() requires
     * (fork → exec /bin/sh → exec cc → waitpid becomes fork → exec cc → waitpid).
     * stderr is redirected to err_path via posix_spawn_file_actions; stdout
     * is inherited from the engine process (same behaviour as system(cmd)).
     *
     * extern char **environ is declared in <unistd.h> when _GNU_SOURCE is set.
     */
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, err_path,
                                     O_WRONLY | O_CREAT | O_TRUNC, 0600);

    pid_t cc_pid;
    int spawn_err = posix_spawnp(&cc_pid, cc_bin, &fa, NULL,
                                  (char *const *)cc_argv, environ);
    posix_spawn_file_actions_destroy(&fa);

    int rc;
    if (spawn_err != 0) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: posix_spawnp(%s) failed: %s", cc_bin,
                 strerror(spawn_err));
        if (src_is_memfd) close(src_fd); else unlink(src_path);
        free(wrapper_src);
        return false;
    }
    {
        int status = 0;
        waitpid(cc_pid, &status, 0);
        rc = (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : 1;
    }

    /* Source file is no longer needed. */
    if (src_is_memfd) close(src_fd); else unlink(src_path);
    free(wrapper_src);
    wrapper_src = NULL;

    if (rc != 0) {
        /*
         * Print the full compiler error output to stderr immediately so the
         * user sees the complete diagnostics.  Additionally store up to
         * sizeof(errmsg)-1 bytes in result->errmsg for the caller's log line.
         */
        FILE *ef = fopen(err_path, "r");
        if (ef) {
            /* Stream full output to stderr first. */
            char line[256];
            while (fgets(line, sizeof(line), ef))
                fputs(line, stderr);
            /* Rewind and store first chunk in errmsg for structured logging. */
            rewind(ef);
            size_t nr = fread(result->errmsg,
                              1, sizeof(result->errmsg) - 1, ef);
            result->errmsg[nr] = '\0';
            fclose(ef);
        } else {
            snprintf(result->errmsg, sizeof(result->errmsg),
                     "codegen: %s exited with error %d", cc_bin, rc);
        }
        unlink(err_path);
        unlink(so_path);
        return false;
    }

    unlink(err_path);  /* remove even on success (may not exist if cc was quiet) */

    /* ── 7. dlopen the shared object ─────────────────────────────────── */
    void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: dlopen('%s'): %s", so_path, dlerror());
        unlink(so_path);
        return false;
    }

    /*
     * Persist the compiled artifact to the cache (if enabled) BEFORE
     * unlinking so_path.
     *
     * codegen_cache_store() performs an atomic rename() of so_path into the
     * cache directory.  After the rename:
     *   • so_path no longer exists on the filesystem.
     *   • The cached entry at cache_path points to the same inode.
     *   • The dlopen handle (acquired above) is inode-based and remains valid
     *     regardless of the file's directory entry — the kernel keeps the
     *     mapping alive as long as any handle is open.
     *   • Future dlopen(cache_path) calls will use the persisted file.
     *
     * On failure (cross-device copy fails, or any other error), fall back to
     * the normal unlink so the inode is reclaimed on close.
     */
    bool so_disposed = false;
    if (have_cache) {
        if (codegen_cache_store(opts->cache, cache_key, so_path))
            so_disposed = true;  /* store consumed so_path (rename or copy+unlink) */
    }
    if (!so_disposed) {
        /*
         * Either caching is disabled, or the store failed.  Unlink the temp
         * file: the kernel keeps the inode alive through the dlopen handle,
         * and reclaims it automatically when the handle is closed (process
         * crash included).
         */
        unlink(so_path);
    }

    /* ── 8. dlsym to find the function ───────────────────────────────── */
    dlerror(); /* clear any old error */
    void *sym = dlsym(handle, func_name);
    const char *err = dlerror();
    if (err) {
        snprintf(result->errmsg, sizeof(result->errmsg),
                 "codegen: dlsym('%s'): %s", func_name, err);
        dlclose(handle);
        return false;
    }

    /* ── 9. Success ──────────────────────────────────────────────────── */
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
