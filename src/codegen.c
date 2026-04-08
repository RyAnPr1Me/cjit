/**
 * codegen.c – System-compiler JIT backend.
 *
 * See codegen.h for the design description.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "codegen.h"
#include "func_table.h"   /* CJIT_NAME_MAX */

#include <stdio.h>       /* FILE, fopen, fclose, snprintf, fprintf */
#include <stdlib.h>      /* free, system                           */
#include <string.h>      /* strlen, strncpy, snprintf              */
#include <stdatomic.h>   /* atomic_fetch_add                       */
#include <unistd.h>      /* unlink, getpid                         */
#include <pthread.h>     /* pthread_self                           */
#include <dlfcn.h>       /* dlopen, dlsym, dlerror                */
#include <sys/types.h>
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
 * Maximum number of arguments passed to the compiler subprocess.
 *
 * cc(1) + -shared + -fPIC + opt-level + up to 10 optional flags +
 * JIT-specific flags (-fno-stack-protector, -fno-asynchronous-unwind-tables) +
 * -w + -o + so_path + -x + c + src_path + NULL = at most 22 entries.
 * When argument specialisation is active, two more entries are added
 * (-D <func=_cjit_i_func>).  Extra user flags (extra_cflags) may add up to
 * ~50 more tokens.  80 gives comfortable headroom.
 */
#define MAX_CC_ARGS 80

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
    }

    if (opts->enable_inlining && level >= OPT_O2)
        argv[n++] = "-finline-functions";

    if (opts->enable_loop_unroll && level >= OPT_O2)
        argv[n++] = "-funroll-loops";

    if (opts->enable_vectorization && level >= OPT_O2)
        argv[n++] = "-ftree-vectorize";

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

    char so_path[300];
    snprintf(so_path, sizeof(so_path), "%s.so", prefix);

    /* ── 2. Optionally generate an argument-specialisation wrapper ────── */
    /*
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

    /* ── 3. Write preamble [+ wrapper] + user source ─────────────────── */
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

    /* ── 4. Build compiler argv ──────────────────────────────────────── */
    const char *cc_argv[MAX_CC_ARGS];
    char err_path[304];
    snprintf(err_path, sizeof(err_path), "%s.err", prefix);
    int n_argv = build_compiler_argv(cc_argv, so_path, src_path, level, opts);

    /* Append extra_cflags tokens (e.g. -I/path, -DFOO, -lm).
     * strtok needs a mutable buffer; we copy into a local array on the stack.
     * Tokens point into this buffer so they are valid until after waitpid. */
    char extra_flags_buf[512];
    extra_flags_buf[0] = '\0';
    if (opts && opts->extra_cflags && opts->extra_cflags[0]) {
        strncpy(extra_flags_buf, opts->extra_cflags, sizeof(extra_flags_buf) - 1);
        extra_flags_buf[sizeof(extra_flags_buf) - 1] = '\0';
        char *tok = strtok(extra_flags_buf, " \t");
        while (tok && n_argv < MAX_CC_ARGS - 1) {
            cc_argv[n_argv++] = tok;
            tok = strtok(NULL, " \t");
        }
        cc_argv[n_argv] = NULL;
    }

    /* Determine which compiler binary to invoke. */
    const char *cc_bin = (opts && opts->cc_binary && opts->cc_binary[0])
                             ? opts->cc_binary : "cc";

    if (opts && opts->verbose) {
        fprintf(stderr, "[cjit/codegen] compile:");
        for (int _i = 0; cc_argv[_i]; ++_i)
            fprintf(stderr, " %s", cc_argv[_i]);
        fprintf(stderr, " 2>%s\n", err_path);
    }

    /* ── 5. Spawn the compiler (no shell intermediate) ────────────────── */
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
