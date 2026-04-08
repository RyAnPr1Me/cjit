/**
 * cli.c – Command-line interface for the CJIT JIT engine.
 *
 * Usage
 * ─────
 *   cjit [OPTIONS] <source.c> [-- <args>...]
 *
 * Options
 * ───────
 *   -f <name>          Entry function name (default: main).
 *   -O0|-O1|-O2|-O3    Optimisation level (default: -O2).
 *   -I <dir>           Add include search path (passed to the compiler).
 *   -D <macro[=val]>   Define a preprocessor macro.
 *   -l <lib>           Link against <lib> (e.g. -l m → -lm).
 *   -L <dir>           Add library search path.
 *   --cc <compiler>    Use a specific compiler binary (default: cc).
 *   --timeout <ms>     Compilation timeout in milliseconds (default: 30000).
 *   -v, --verbose      Print compilation events to stderr.
 *   --stats            Print engine statistics after execution.
 *   -h, --help         Print this help and exit.
 *
 * How it works
 * ────────────
 * The ENTIRE source file is compiled as a single translation unit into a
 * shared object by the system C compiler, then loaded with dlopen and the
 * entry function is located with dlsym and called.  This is equivalent to:
 *
 *   cc -shared -fPIC -O2 -o /tmp/cjit_<pid>.so source.c
 *   dlopen + dlsym("main") + call
 *
 * but performed at runtime through the CJIT engine so that the JIT can
 * re-optimise the code in the background at higher tiers as it runs.
 *
 * Arguments after the optional -- separator are forwarded to the JIT-compiled
 * function as argc/argv (argv[0] is set to the source file path).
 *
 * The entry function must have one of these signatures:
 *   int <name>(void)
 *   int <name>(int argc, char **argv)
 *
 * Example
 * ───────
 *   $ cat hello.c
 *   #include <stdio.h>
 *   int main(int argc, char **argv) {
 *       printf("hello from JIT! got %d arg(s)\n", argc - 1);
 *       return 0;
 *   }
 *
 *   $ cjit hello.c -- foo bar
 *   hello from JIT! got 2 arg(s)
 *
 *   $ cjit -O3 -v --stats hello.c
 *   $ cjit -I/usr/local/include -lm -DDEBUG hello.c
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>

#include "../include/cjit.h"

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void print_help(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS] <source.c> [-- <args>...]\n"
        "\n"
        "JIT-compile an entire C source file and run it.\n"
        "\n"
        "Options:\n"
        "  -f <name>          Entry function name (default: main)\n"
        "  -O0|-O1|-O2|-O3    Optimisation level  (default: -O2)\n"
        "  -I <dir>           Add include search path (forwarded to cc)\n"
        "  -D <macro[=val]>   Define a preprocessor macro\n"
        "  -l <lib>           Link against a library (e.g. -l m for libm)\n"
        "  -L <dir>           Add library search path\n"
        "  --cc <compiler>    Compiler binary to use (default: cc)\n"
        "  --timeout <ms>     Compilation timeout in ms (default: 30000)\n"
        "  -v, --verbose      Print compilation events to stderr\n"
        "  --stats            Print engine statistics after execution\n"
        "  -h, --help         Show this help and exit\n"
        "\n"
        "Arguments after -- are forwarded to the JIT-compiled function as argv.\n",
        prog);
}

/** Read an entire file into a malloc'd NUL-terminated buffer. */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cjit: cannot open '%s'\n", path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "cjit: cannot seek '%s'\n", path);
        return NULL;
    }
    long sz = ftell(f);
    rewind(f);

    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "cjit: cannot determine size of '%s'\n", path);
        return NULL;
    }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "cjit: out of memory\n");
        return NULL;
    }

    size_t nr = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[nr] = '\0';
    return buf;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    const char *source_path  = NULL;
    const char *func_name    = "main";
    opt_level_t opt_level    = OPT_O2;
    bool        verbose      = false;
    bool        print_stats  = false;
    unsigned    timeout_ms   = 30000;
    int         jit_argc     = 0;
    char      **jit_argv     = NULL;

    /*
     * extra_cflags accumulates space-separated flags collected from -I, -D,
     * -l, and -L options.  cc_binary holds the optional --cc override.
     */
    char extra_cflags[CJIT_MAX_EXTRA_CFLAGS] = "";
    char cc_binary[CJIT_MAX_CC_BINARY]        = "";

    /* Append a single flag token to extra_cflags (with leading space). */
    /* Returns false and prints an error on overflow. */
#define APPEND_CFLAG(flag) do { \
    size_t _cur = strlen(extra_cflags); \
    size_t _add = strlen(flag); \
    size_t _sep = (_cur > 0) ? 1u : 0u; /* space separator */ \
    if (_cur + _sep + _add + 1 > CJIT_MAX_EXTRA_CFLAGS) { \
        fprintf(stderr, "cjit: too many compiler flags (limit %d chars)\n", \
                CJIT_MAX_EXTRA_CFLAGS - 1); \
        return 1; \
    } \
    if (_sep) extra_cflags[_cur++] = ' '; \
    memcpy(extra_cflags + _cur, (flag), _add + 1); \
} while (0)

    /* ── Parse CLI arguments ───────────────────────────────────────────── */
    int i;
    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];

        if (strcmp(a, "--") == 0) {
            /* Everything after -- is forwarded to the JIT function. */
            ++i;
            break;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(a, "-v") == 0 || strcmp(a, "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(a, "--stats") == 0) {
            print_stats = true;
        } else if (strcmp(a, "-O0") == 0) {
            opt_level = OPT_NONE;
        } else if (strcmp(a, "-O1") == 0) {
            opt_level = OPT_O1;
        } else if (strcmp(a, "-O2") == 0) {
            opt_level = OPT_O2;
        } else if (strcmp(a, "-O3") == 0) {
            opt_level = OPT_O3;
        } else if (strcmp(a, "-f") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "cjit: -f requires an argument\n");
                return 1;
            }
            func_name = argv[i];
        } else if (strcmp(a, "--cc") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "cjit: --cc requires an argument\n");
                return 1;
            }
            strncpy(cc_binary, argv[i], sizeof(cc_binary) - 1);
            cc_binary[sizeof(cc_binary) - 1] = '\0';
        } else if (strcmp(a, "--timeout") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "cjit: --timeout requires an argument\n");
                return 1;
            }
            timeout_ms = (unsigned)strtoul(argv[i], NULL, 10);
            if (timeout_ms == 0) timeout_ms = 30000;
        } else if (strncmp(a, "-I", 2) == 0) {
            /* -I<path> or -I <path> */
            const char *path = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!path) { fprintf(stderr, "cjit: -I requires a path\n"); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-I%s", path);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-D", 2) == 0) {
            /* -D<macro> or -D <macro> */
            const char *def = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!def) { fprintf(stderr, "cjit: -D requires a macro\n"); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-D%s", def);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-l", 2) == 0) {
            /* -l<lib> or -l <lib> */
            const char *lib = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!lib) { fprintf(stderr, "cjit: -l requires a library name\n"); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-l%s", lib);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-L", 2) == 0) {
            /* -L<dir> or -L <dir> */
            const char *dir = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!dir) { fprintf(stderr, "cjit: -L requires a directory\n"); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-L%s", dir);
            APPEND_CFLAG(flag);
        } else if (a[0] == '-') {
            fprintf(stderr, "cjit: unknown option '%s'\n", a);
            print_help(argv[0]);
            return 1;
        } else if (!source_path) {
            source_path = a;
        } else {
            fprintf(stderr, "cjit: unexpected argument '%s' (use -- to pass "
                    "args to the JIT function)\n", a);
            return 1;
        }
    }

#undef APPEND_CFLAG

    if (!source_path) {
        print_help(argv[0]);
        return 1;
    }

    /*
     * Build the argv array forwarded to the JIT-compiled function.
     * argv[0] = source file path (mirrors normal C convention).
     * argv[1..] = arguments from after the -- separator.
     */
    int extra = argc - i;             /* number of args after -- */
    jit_argc = 1 + extra;
    jit_argv = malloc(((size_t)jit_argc + 1) * sizeof(char *));
    if (!jit_argv) {
        fprintf(stderr, "cjit: out of memory\n");
        return 1;
    }
    jit_argv[0] = (char *)source_path;
    for (int j = 0; j < extra; ++j)
        jit_argv[1 + j] = argv[i + j];
    jit_argv[jit_argc] = NULL;

    /* ── Read the entire source file ───────────────────────────────────── */
    char *source = read_file(source_path);
    if (!source) { free(jit_argv); return 1; }

    /* ── Configure the JIT engine ──────────────────────────────────────── */
    /*
     * For a one-shot file-run we use a single compiler thread and tight
     * hot thresholds.  The monitor is not useful here, but keeping it alive
     * allows the engine to upgrade to a higher tier if the JIT function
     * calls itself recursively enough to cross the threshold.
     */
    cjit_config_t cfg        = cjit_default_config();
    cfg.verbose              = verbose;
    cfg.compiler_threads     = 1;
    cfg.monitor_interval_ms  = 50;
    cfg.hot_threshold_t1     = 10;
    cfg.hot_threshold_t2     = 100;
    cfg.hot_ir_cache_size    = 4;
    cfg.warm_ir_cache_size   = 8;

    /* Forward extra compiler flags and optional compiler override. */
    if (extra_cflags[0])
        snprintf(cfg.extra_cflags, sizeof(cfg.extra_cflags), "%s", extra_cflags);
    if (cc_binary[0])
        snprintf(cfg.cc_binary, sizeof(cfg.cc_binary), "%s", cc_binary);

    cjit_engine_t *engine = cjit_create(&cfg);
    if (!engine) {
        fprintf(stderr, "cjit: failed to create JIT engine\n");
        free(source);
        free(jit_argv);
        return 1;
    }

    /* ── Register the whole file as a single compilation unit ─────────── */
    /*
     * The entire source string is passed as the IR.  codegen_compile()
     * prepends the CJIT preamble (LIKELY/UNLIKELY/HOT macros + standard
     * headers) and compiles the whole thing as a shared object.  dlsym then
     * locates func_name inside that shared object.
     *
     * No AOT fallback is provided: the engine returns NULL from
     * cjit_get_func() until the first compilation succeeds.
     */
    func_id_t id = cjit_register_function(engine, func_name, source, NULL);
    if (id == CJIT_INVALID_FUNC_ID) {
        fprintf(stderr, "cjit: failed to register '%s'\n", func_name);
        cjit_destroy(engine);
        free(source);
        free(jit_argv);
        return 1;
    }

    /* ── Start the engine and kick off compilation immediately ─────────── */
    cjit_start(engine);
    cjit_request_recompile(engine, id, opt_level);

    /* ── Wait until the compiled function is available ─────────────────── */
    /*
     * cjit_wait_compiled blocks on a condition variable until the compiler
     * thread signals that compilation has completed (success or failure).
     * No busy-waiting.
     */
    if (verbose)
        fprintf(stderr, "[cjit] compiling '%s' at -O%d...\n",
                func_name, (int)opt_level);

    bool compiled = cjit_wait_compiled(engine, id, timeout_ms);
    jit_func_t fn = compiled ? cjit_get_func(engine, id) : NULL;

    if (!fn) {
        fprintf(stderr, "cjit: compilation of '%s' %s after %u ms\n",
                func_name,
                compiled ? "succeeded but returned no function pointer"
                         : "timed out",
                timeout_ms);
        if (print_stats) cjit_print_stats(engine);
        cjit_destroy(engine);
        free(source);
        free(jit_argv);
        return 1;
    }

    if (verbose)
        fprintf(stderr, "[cjit] '%s' ready (O%d), running...\n",
                func_name,
                (int)cjit_get_current_opt_level(engine, id));

    /* ── Call the JIT-compiled function ────────────────────────────────── */
    /*
     * The function is called with (argc, argv) matching the standard C main
     * signature.  If the function is declared as int f(void), the extra
     * arguments are harmlessly ignored on all calling conventions we target
     * (x86-64 SysV ABI, AArch64 ABI).
     */
    typedef int (*main_fn_t)(int, char **);
    int ret = ((main_fn_t)fn)(jit_argc, jit_argv);

    if (print_stats) cjit_print_stats(engine);

    cjit_destroy(engine);
    free(source);
    free(jit_argv);
    return ret;
}
