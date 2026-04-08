/**
 * cli.c – Command-line interface for the CJIT JIT engine.
 *
 * Usage
 * ─────
 *   cjit [OPTIONS] <source.c> [-- <args>...]
 *   cjit [OPTIONS] -e '<C code snippet>' [-- <args>...]
 *
 * Options
 * ───────
 *   -f <name>          Entry function name (default: main).
 *   -O0|-O1|-O2|-O3    Optimisation level (default: -O2).
 *   -e, --eval <code>  Evaluate an inline C snippet instead of a file.
 *   -I <dir>           Add include search path (passed to the compiler).
 *   -D <macro[=val]>   Define a preprocessor macro.
 *   -l <lib>           Link against <lib> (e.g. -l m → -lm).
 *   -L <dir>           Add library search path.
 *   --cc <compiler>    Use a specific compiler binary (default: cc).
 *   --timeout <ms>     Compilation timeout in milliseconds (default: 30000).
 *   --fast-math        Enable -ffast-math (improves float throughput, OPT_O3).
 *   -v, --verbose      Print compilation events to stderr.
 *   --watch            Re-run the program whenever the source file changes.
 *   --stats            Print engine statistics after execution.
 *   --ir-stats         Print IR-cache statistics after execution.
 *   --version          Print version and exit.
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
 * function as argc/argv (argv[0] is set to the source file path or "eval").
 *
 * The entry function must have one of these signatures:
 *   int <name>(void)
 *   int <name>(int argc, char **argv)
 *
 * Example
 * ───────
 *   $ cjit hello.c -- foo bar
 *   $ cjit -O3 -v --stats hello.c
 *   $ cjit -e '#include <stdio.h>\nint main(){puts("hi");return 0;}'
 *   $ cjit --watch hello.c
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <signal.h>

#include "../include/cjit.h"

/* ── ANSI colour helpers (only when stderr is a real tty) ────────────────── */

static bool g_color = false;

#define CLR_RESET   (g_color ? "\033[0m"    : "")
#define CLR_BOLD    (g_color ? "\033[1m"    : "")
#define CLR_RED     (g_color ? "\033[1;31m" : "")
#define CLR_YELLOW  (g_color ? "\033[1;33m" : "")
#define CLR_CYAN    (g_color ? "\033[1;36m" : "")
#define CLR_GREEN   (g_color ? "\033[1;32m" : "")

/* ── graceful shutdown on SIGINT / SIGTERM ──────────────────────────────── */

static volatile sig_atomic_t g_stop = 0;
static void handle_signal(int sig) { (void)sig; g_stop = 1; }

/* ── compile-event callback: print failures to stderr ───────────────────── */

typedef struct {
    bool success;
    bool fired;
} compile_result_t;

static void compile_event_cb(const cjit_compile_event_t *ev, void *ud)
{
    compile_result_t *r = (compile_result_t *)ud;
    r->fired   = true;
    r->success = ev->success;
    if (!ev->success) {
        fprintf(stderr,
                "%scjit:%s compilation of '%s' failed "
                "(level=O%d, duration=%u ms)\n",
                CLR_RED, CLR_RESET,
                ev->func_name, (int)ev->level, ev->duration_ms);
    }
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void print_help(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS] <source.c> [-- <args>...]\n"
        "       %s [OPTIONS] -e '<C code>' [-- <args>...]\n"
        "\n"
        "%sJIT-compile an entire C source file (or inline snippet) and run it.%s\n"
        "\n"
        "Options:\n"
        "  %s-f <name>%s          Entry function name (default: main)\n"
        "  %s-O0|-O1|-O2|-O3%s   Optimisation level  (default: -O2)\n"
        "  %s-e, --eval <code>%s  Evaluate an inline C snippet\n"
        "  -I <dir>           Add include search path (forwarded to cc)\n"
        "  -D <macro[=val]>   Define a preprocessor macro\n"
        "  -l <lib>           Link against a library (e.g. -l m for libm)\n"
        "  -L <dir>           Add library search path\n"
        "  --cc <compiler>    Compiler binary to use (default: cc)\n"
        "  --timeout <ms>     Compilation timeout in ms (default: 30000)\n"
        "  --fast-math        Enable -ffast-math at O3 (float throughput)\n"
        "  -v, --verbose      Print compilation events to stderr\n"
        "  --watch            Re-run whenever the source file changes\n"
        "  --stats            Print engine statistics after execution\n"
        "  --ir-stats         Print IR-cache statistics after execution\n"
        "  --version          Print version and exit\n"
        "  -h, --help         Show this help and exit\n"
        "\n"
        "Arguments after -- are forwarded to the JIT-compiled function as argv.\n",
        prog, prog,
        CLR_BOLD, CLR_RESET,
        CLR_CYAN, CLR_RESET,
        CLR_CYAN, CLR_RESET,
        CLR_CYAN, CLR_RESET);
}

/** Read an entire file into a malloc'd NUL-terminated buffer. */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "%scjit:%s cannot open '%s': %s\n",
                CLR_RED, CLR_RESET, path, strerror(errno));
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "%scjit:%s cannot seek '%s'\n",
                CLR_RED, CLR_RESET, path);
        return NULL;
    }
    long sz = ftell(f);
    rewind(f);

    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "%scjit:%s cannot determine size of '%s'\n",
                CLR_RED, CLR_RESET, path);
        return NULL;
    }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "%scjit:%s out of memory\n", CLR_RED, CLR_RESET);
        return NULL;
    }

    size_t nr = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[nr] = '\0';
    return buf;
}

/** Return last-modification time of a file (seconds). 0 on error. */
static time_t file_mtime(const char *path)
{
    struct stat st;
    return (stat(path, &st) == 0) ? st.st_mtime : 0;
}

/*
 * Expand C escape sequences in a string literal so the user can pass
 * newlines as "\n" on the command line.  The result is malloc'd.
 */
static char *unescape(const char *s)
{
    size_t len = strlen(s);
    char *out  = malloc(len + 1);
    if (!out) return NULL;
    char *p = out;
    for (size_t i = 0; i < len; ) {
        if (s[i] == '\\' && i + 1 < len) {
            switch (s[i + 1]) {
                case 'n':  *p++ = '\n'; i += 2; continue;
                case 't':  *p++ = '\t'; i += 2; continue;
                case 'r':  *p++ = '\r'; i += 2; continue;
                case '\\': *p++ = '\\'; i += 2; continue;
                case '\'': *p++ = '\''; i += 2; continue;
                case '"':  *p++ = '"';  i += 2; continue;
                default: break;
            }
        }
        *p++ = s[i++];
    }
    *p = '\0';
    return out;
}

/* ── one-shot JIT run ────────────────────────────────────────────────────── */
/*
 * Compile `source` under `func_name` and call it with (jit_argc, jit_argv).
 * `display_path` is used in diagnostics.
 *
 * Returns the exit code of the JIT function, or 1 on engine/compile failure.
 */
static int run_once(const char   *source,
                    const char   *func_name,
                    const char   *display_path,
                    opt_level_t   opt_level,
                    bool          verbose,
                    bool          print_stats,
                    bool          print_ir_stats,
                    unsigned      timeout_ms,
                    const char   *extra_cflags,
                    const char   *cc_binary,
                    bool          fast_math,
                    int           jit_argc,
                    char        **jit_argv)
{
    cjit_config_t cfg        = cjit_default_config();
    cfg.verbose              = verbose;
    cfg.compiler_threads     = 1;
    cfg.monitor_interval_ms  = 50;
    cfg.hot_threshold_t1     = 10;
    cfg.hot_threshold_t2     = 100;
    cfg.hot_ir_cache_size    = 4;
    cfg.warm_ir_cache_size   = 8;
    cfg.enable_fast_math     = fast_math;
    cfg.compile_timeout_ms   = timeout_ms;

    if (extra_cflags && extra_cflags[0])
        snprintf(cfg.extra_cflags, sizeof(cfg.extra_cflags), "%s", extra_cflags);
    if (cc_binary && cc_binary[0])
        snprintf(cfg.cc_binary, sizeof(cfg.cc_binary), "%s", cc_binary);

    cjit_engine_t *engine = cjit_create(&cfg);
    if (!engine) {
        fprintf(stderr, "%scjit:%s failed to create JIT engine\n",
                CLR_RED, CLR_RESET);
        return 1;
    }

    /* Hook up compile-event callback so failures are reported clearly. */
    compile_result_t cr = { .success = false, .fired = false };
    cjit_set_compile_callback(engine, compile_event_cb, &cr);

    func_id_t id = cjit_register_function(engine, func_name, source, NULL);
    if (id == CJIT_INVALID_FUNC_ID) {
        fprintf(stderr, "%scjit:%s failed to register '%s'\n",
                CLR_RED, CLR_RESET, func_name);
        cjit_destroy(engine);
        return 1;
    }

    cjit_start(engine);
    cjit_request_recompile(engine, id, opt_level);

    if (verbose)
        fprintf(stderr, "%s[cjit]%s compiling '%s' at -O%d...\n",
                CLR_CYAN, CLR_RESET, func_name, (int)opt_level);

    bool compiled = cjit_wait_compiled(engine, id, timeout_ms);
    jit_func_t fn = compiled ? cjit_get_func(engine, id) : NULL;

    if (!fn) {
        if (!cr.fired) {
            /* No event fired before timeout. */
            fprintf(stderr,
                    "%scjit:%s compilation of '%s' timed out after %u ms\n",
                    CLR_RED, CLR_RESET, func_name, timeout_ms);
        }
        if (print_stats)    cjit_print_stats(engine);
        if (print_ir_stats) cjit_print_ir_cache_stats(engine);
        cjit_destroy(engine);
        return 1;
    }

    if (verbose)
        fprintf(stderr, "%s[cjit]%s '%s' ready (O%d), running %s%s%s...\n",
                CLR_GREEN, CLR_RESET,
                func_name,
                (int)cjit_get_current_opt_level(engine, id),
                CLR_BOLD, display_path, CLR_RESET);

    typedef int (*main_fn_t)(int, char **);
    int ret = ((main_fn_t)(uintptr_t)fn)(jit_argc, jit_argv);

    if (print_stats)    cjit_print_stats(engine);
    if (print_ir_stats) cjit_print_ir_cache_stats(engine);

    cjit_destroy(engine);
    return ret;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    const char *source_path  = NULL;
    const char *eval_snippet = NULL;  /* -e / --eval */
    const char *func_name    = "main";
    opt_level_t opt_level    = OPT_O2;
    bool        verbose      = false;
    bool        print_stats  = false;
    bool        print_ir_stats = false;
    bool        watch_mode   = false;
    bool        fast_math    = false;
    unsigned    timeout_ms   = 30000;
    int         jit_argc     = 0;
    char      **jit_argv     = NULL;

    char extra_cflags[CJIT_MAX_EXTRA_CFLAGS] = "";
    char cc_binary[CJIT_MAX_CC_BINARY]        = "";

    g_color = (isatty(STDERR_FILENO) == 1);

#define APPEND_CFLAG(flag) do { \
    size_t _cur = strlen(extra_cflags); \
    size_t _add = strlen(flag); \
    size_t _sep = (_cur > 0) ? 1u : 0u; \
    if (_cur + _sep + _add + 1 > CJIT_MAX_EXTRA_CFLAGS) { \
        fprintf(stderr, "%scjit:%s too many compiler flags (limit %d chars)\n", \
                CLR_RED, CLR_RESET, CJIT_MAX_EXTRA_CFLAGS - 1); \
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
            ++i;
            break;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(a, "--version") == 0) {
            fprintf(stdout, "cjit %s\n", CJIT_VERSION);
            return 0;
        } else if (strcmp(a, "-v") == 0 || strcmp(a, "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(a, "--stats") == 0) {
            print_stats = true;
        } else if (strcmp(a, "--ir-stats") == 0) {
            print_ir_stats = true;
        } else if (strcmp(a, "--watch") == 0) {
            watch_mode = true;
        } else if (strcmp(a, "--fast-math") == 0) {
            fast_math = true;
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
                fprintf(stderr, "%scjit:%s -f requires an argument\n",
                        CLR_RED, CLR_RESET);
                return 1;
            }
            func_name = argv[i];
        } else if (strcmp(a, "-e") == 0 || strcmp(a, "--eval") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "%scjit:%s %s requires a code argument\n",
                        CLR_RED, CLR_RESET, a);
                return 1;
            }
            eval_snippet = argv[i];
        } else if (strcmp(a, "--cc") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "%scjit:%s --cc requires an argument\n",
                        CLR_RED, CLR_RESET);
                return 1;
            }
            strncpy(cc_binary, argv[i], sizeof(cc_binary) - 1);
            cc_binary[sizeof(cc_binary) - 1] = '\0';
        } else if (strcmp(a, "--timeout") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "%scjit:%s --timeout requires an argument\n",
                        CLR_RED, CLR_RESET);
                return 1;
            }
            timeout_ms = (unsigned)strtoul(argv[i], NULL, 10);
            if (timeout_ms == 0) timeout_ms = 30000;
        } else if (strncmp(a, "-I", 2) == 0) {
            const char *path = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!path) { fprintf(stderr, "%scjit:%s -I requires a path\n",
                                 CLR_RED, CLR_RESET); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-I%s", path);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-D", 2) == 0) {
            const char *def = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!def) { fprintf(stderr, "%scjit:%s -D requires a macro\n",
                                CLR_RED, CLR_RESET); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-D%s", def);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-l", 2) == 0) {
            const char *lib = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!lib) { fprintf(stderr, "%scjit:%s -l requires a library name\n",
                                CLR_RED, CLR_RESET); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-l%s", lib);
            APPEND_CFLAG(flag);
        } else if (strncmp(a, "-L", 2) == 0) {
            const char *dir = (a[2] != '\0') ? &a[2] : (++i < argc ? argv[i] : NULL);
            if (!dir) { fprintf(stderr, "%scjit:%s -L requires a directory\n",
                                CLR_RED, CLR_RESET); return 1; }
            char flag[CJIT_MAX_EXTRA_CFLAGS];
            snprintf(flag, sizeof(flag), "-L%s", dir);
            APPEND_CFLAG(flag);
        } else if (a[0] == '-') {
            fprintf(stderr, "%scjit:%s unknown option '%s'\n",
                    CLR_RED, CLR_RESET, a);
            print_help(argv[0]);
            return 1;
        } else if (!source_path) {
            source_path = a;
        } else {
            fprintf(stderr, "%scjit:%s unexpected argument '%s' "
                    "(use -- to pass args to the JIT function)\n",
                    CLR_RED, CLR_RESET, a);
            return 1;
        }
    }

#undef APPEND_CFLAG

    /* Exactly one of source_path or eval_snippet must be set. */
    if (!source_path && !eval_snippet) {
        print_help(argv[0]);
        return 1;
    }
    if (source_path && eval_snippet) {
        fprintf(stderr,
                "%scjit:%s cannot use both a source file and -e/--eval\n",
                CLR_RED, CLR_RESET);
        return 1;
    }
    if (watch_mode && eval_snippet) {
        fprintf(stderr,
                "%scjit:%s --watch requires a source file, not -e/--eval\n",
                CLR_RED, CLR_RESET);
        return 1;
    }

    /*
     * Build the argv array forwarded to the JIT-compiled function.
     * argv[0] = source_path (or "eval") — mirrors normal C convention.
     */
    int extra = argc - i;
    jit_argc  = 1 + extra;
    jit_argv  = malloc(((size_t)jit_argc + 1) * sizeof(char *));
    if (!jit_argv) {
        fprintf(stderr, "%scjit:%s out of memory\n", CLR_RED, CLR_RESET);
        return 1;
    }
    jit_argv[0] = (char *)(source_path ? source_path : "eval");
    for (int j = 0; j < extra; ++j)
        jit_argv[1 + j] = argv[i + j];
    jit_argv[jit_argc] = NULL;

    /* ── -e / --eval path ──────────────────────────────────────────────── */
    if (eval_snippet) {
        char *source = unescape(eval_snippet);
        if (!source) {
            fprintf(stderr, "%scjit:%s out of memory\n", CLR_RED, CLR_RESET);
            free(jit_argv);
            return 1;
        }
        int ret = run_once(source, func_name, "eval", opt_level,
                           verbose, print_stats, print_ir_stats,
                           timeout_ms, extra_cflags, cc_binary, fast_math,
                           jit_argc, jit_argv);
        free(source);
        free(jit_argv);
        return ret;
    }

    /* ── Normal file path (possibly with --watch) ───────────────────────── */
    /* Install signal handlers for clean shutdown in --watch mode. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    time_t last_mtime = file_mtime(source_path);
    int ret = 0;

    do {
        char *source = read_file(source_path);
        if (!source) { free(jit_argv); return 1; }

        ret = run_once(source, func_name, source_path, opt_level,
                       verbose, print_stats, print_ir_stats,
                       timeout_ms, extra_cflags, cc_binary, fast_math,
                       jit_argc, jit_argv);
        free(source);

        if (!watch_mode || g_stop)
            break;

        /* --watch: poll for file change every 250 ms. */
        if (verbose)
            fprintf(stderr,
                    "%s[cjit/watch]%s watching '%s' for changes "
                    "(Ctrl-C to stop)...\n",
                    CLR_YELLOW, CLR_RESET, source_path);

        while (!g_stop) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 250000000L };
            nanosleep(&ts, NULL);
            time_t mt = file_mtime(source_path);
            if (mt != last_mtime) {
                last_mtime = mt;
                fprintf(stderr,
                        "%s[cjit/watch]%s '%s' changed — rerunning...\n",
                        CLR_YELLOW, CLR_RESET, source_path);
                break;
            }
        }
    } while (!g_stop);

    free(jit_argv);
    return ret;
}
