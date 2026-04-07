# cjit

A non-blocking, background JIT compiler engine for C.  Register C source
strings as "IR", start the engine, and call your functions — the engine
compiles them in the background at increasing optimisation tiers while your
runtime threads keep running without any stalls or locks on the hot path.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Building](#building)
5. [CLI Usage](#cli-usage)
6. [Library API](#library-api)
   - [Engine Lifecycle](#engine-lifecycle)
   - [Registering Functions](#registering-functions)
   - [Dispatching Calls](#dispatching-calls)
   - [Requesting Recompilation](#requesting-recompilation)
   - [Statistics](#statistics)
7. [Configuration Reference](#configuration-reference)
8. [IR Source Conventions](#ir-source-conventions)
9. [Internal Components](#internal-components)
   - [func\_table](#func_table)
   - [work\_queue](#work_queue-mpmc)
   - [codegen](#codegen)
   - [ir\_cache](#ir_cache-multi-generation-lru)
   - [deferred\_gc](#deferred_gc)
10. [Thread Safety](#thread-safety)
11. [Performance Tuning](#performance-tuning)
12. [Requirements](#requirements)

---

## Overview

`cjit` is a C library (plus a thin CLI wrapper) that turns plain C source
strings into optimised native code at runtime.  The hot dispatch path is a
**single atomic load** followed by an indirect call — no mutexes, no
reference counts, no memory barriers beyond that one acquire-load.

Background compiler threads continuously pick up work from a lock-free queue,
invoke the system C compiler (`cc`), load the resulting shared object with
`dlopen`/`dlsym`, and atomically swap the function pointer in the table.  A
monitor thread watches per-function call counters and escalates hot functions
through multiple optimisation tiers automatically.

A multi-generation LRU IR cache (HOT → WARM → COLD/disk) keeps memory usage
bounded even with thousands of registered functions.  A dedicated background
thread monitors `/proc/meminfo` and tightens cache capacity under memory
pressure.

---

## Features

- **Zero-overhead hot path** — one `atomic_load(acquire)` per dispatch.
- **Background compilation** — the system C compiler runs in worker threads;
  runtime threads are never stalled.
- **Tiered optimisation** — functions are automatically re-compiled at `-O1`,
  `-O2`, then `-O3` (with unrolling, vectorisation, and `-march=native`) as
  they warm up.
- **AOT fallback** — register a pre-compiled function pointer that is used
  until the first JIT compilation completes.
- **Multi-generation LRU IR cache** — IR strings cycle through HOT (memory),
  WARM (memory), and COLD (disk) generations; spilled IR is restored on demand.
- **Memory-pressure awareness** — cache capacities shrink under system memory
  pressure and expand when pressure falls.
- **Grace-period deferred GC** — retired `dlopen` handles are only
  `dlclose`d after a configurable quiet period, preventing use-after-free on
  the hot path without requiring hazard pointers or epoch counters in runtime
  threads.
- **Lock-free MPMC work queue** — Dmitry Vyukov's algorithm; O(1) enqueue
  and dequeue with no blocking.
- **C11 / POSIX** — no external dependencies beyond `libpthread` and `libdl`.

---

## Architecture

```
  Runtime threads (any number)
  ────────────────────────────
  Hot path per call:
    fn = atomic_load(&table[id].func_ptr);  ← single acquire-load
    fn(args…);                              ← indirect call
    atomic_fetch_add(&table[id].call_cnt);  ← relaxed increment

  Background threads (started by cjit_start)
  ─────────────────────────────────────────

  ┌──────────────────────────────────────────────────┐
  │  Monitor thread (×1)                             │
  │   • Wakes every monitor_interval_ms              │
  │   • Scans call_cnt of all registered functions   │
  │   • Detects hot functions (cnt ≥ hot_threshold)  │
  │   • Enqueues compile_task onto MPMC work-queue   │
  └──────────────────────────┬───────────────────────┘
                             │  (lock-free MPMC queue)
  ┌──────────────────────────▼───────────────────────┐
  │  Compiler threads (×3, CJIT_COMPILER_THREADS)    │
  │   • Spin-sleep waiting for compile_task          │
  │   • Invoke cc -shared -fPIC -O<N> …              │
  │   • dlopen + dlsym the resulting .so             │
  │   • func_table_swap(new_fn, handle)              │
  │   • Retire old handle via dgc_retire()           │
  └──────────────────────────┬───────────────────────┘
                             │  (lock-free retire stack)
  ┌──────────────────────────▼───────────────────────┐
  │  GC thread (×1, inside deferred_gc)              │
  │   • Wakes every sweep_interval_ms                │
  │   • dlclose handles older than grace_period_ms   │
  └──────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │  Pressure-monitor thread (×1, inside ir_cache)   │
  │   • Reads /proc/meminfo every mem_pressure_check_ms│
  │   • Adjusts effective HOT/WARM cache capacities  │
  │   • Proactively evicts HOT→WARM→COLD as needed   │
  └──────────────────────────────────────────────────┘
```

### RCU-style atomic pointer swap

1. Compiler thread finishes compilation → `new_fn`, `new_handle`.
2. Acquires the per-entry `compile_lock`.
3. Saves `old_handle = entry->dl_handle`.
4. Stores `entry->dl_handle = new_handle`.
5. `atomic_store(&entry->func_ptr, new_fn, release)` — this is the
   "pointer publish"; all subsequent acquire-loads by runtime threads see
   `new_fn`.  Any runtime thread that loaded `old_fn` before this store
   safely finishes its call using old code (still mapped; `dlclose` deferred).
6. Releases `compile_lock`.
7. `dgc_retire(dgc, old_handle)` — GC thread calls `dlclose(old_handle)`
   after `grace_period_ms` milliseconds.

---

## Building

Requirements: CMake ≥ 3.14, a C11 compiler, POSIX threads, and a system C
compiler reachable as `cc` at runtime for JIT compilation.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

This produces:

| Target | Description |
|--------|-------------|
| `build/libcjit_lib.a` | Static library to link into your application |
| `build/demo` | End-to-end demo executable |
| `build/cjit` | Command-line JIT runner |

### Install

```sh
cmake --install build --prefix /usr/local
```

Installs `cjit` to `$prefix/bin` and `cjit.h` to `$prefix/include/cjit/`.

---

## CLI Usage

The `cjit` binary JIT-compiles an entire C source file and runs its entry
function, with optional background tier escalation.

```
cjit [OPTIONS] <source.c> [-- <args>...]

Options:
  -f <name>       Entry function name (default: main)
  -O0|-O1|-O2|-O3 Optimisation level  (default: -O2)
  -v, --verbose   Print compilation events to stderr
  --stats         Print engine statistics after execution
  -h, --help      Show this help and exit

Arguments after -- are forwarded to the JIT-compiled function as argv.
```

> **Note on `-f`:** The `-f` prefix is also used by many C compilers for
> feature flags (e.g. `-ffast-math`).  `cjit` stops argument parsing at `--`,
> so any compiler-style flags that need to reach the *JIT-compiled program's*
> `argv` must be placed after the `--` separator.  `cjit`'s own options are
> always consumed before that separator.

### Examples

```sh
# Compile and run hello.c
cjit hello.c

# Pass arguments to the JIT-compiled program
cjit hello.c -- foo bar baz

# Compile at -O3 with verbose output and stats
cjit -O3 -v --stats hello.c

# Run a non-main entry point
cjit -f my_entry program.c
```

The entry function may have either of these signatures:

```c
int main(void)
int main(int argc, char **argv)
```

`argv[0]` is set to the source file path; any arguments after `--` become
`argv[1]` onwards.

---

## Library API

Include the single public header:

```c
#include <cjit.h>
```

Link with `-lcjit_lib -lpthread -ldl`.

### Engine Lifecycle

```c
// 1. Get default config (all fields pre-filled with sensible values)
cjit_config_t cfg = cjit_default_config();

// 2. Optionally customise
cfg.compiler_threads  = 4;
cfg.hot_threshold_t1  = 1000;
cfg.verbose           = true;

// 3. Create engine (does NOT start threads yet)
cjit_engine_t *engine = cjit_create(&cfg);   // NULL on failure
if (!engine) abort();

// 4. Register all functions (single-threaded setup phase)
func_id_t id = cjit_register_function(engine, "my_func", ir_source, aot_fn);

// 5. Start background threads (compiler + monitor + GC + pressure)
cjit_start(engine);

// 6. ... use the engine ...

// 7. Stop threads and free resources
cjit_destroy(engine);  // calls cjit_stop() internally if needed
```

You may call `cjit_stop()` separately if you want to quiesce the engine
without freeing it (e.g. to inspect final statistics):

```c
cjit_stop(engine);
cjit_print_stats(engine);
cjit_destroy(engine);
```

### Registering Functions

```c
func_id_t cjit_register_function(
    cjit_engine_t *engine,
    const char    *name,         // unique symbol name (used in dlsym)
    const char    *ir_source,    // C source code of the function
    jit_func_t     aot_fallback  // pre-compiled fallback; may be NULL
);
```

- Must be called **before** `cjit_start()`.
- Do not call it concurrently with other registrations.
- Returns a stable `func_id_t` (a small integer index).
- Returns `CJIT_INVALID_FUNC_ID` on failure (table full, duplicate name).

The `aot_fallback` is returned by `cjit_get_func()` until the first JIT
compilation completes.  If it is `NULL`, `cjit_get_func()` returns `NULL`
until compilation finishes (you must poll or use `cjit_request_recompile()`
and wait).

### Dispatching Calls

**Manual dispatch (explicit cast):**

```c
typedef int (*add_fn_t)(int, int);

// Always record the call so the monitor can track hotness
cjit_record_call(engine, id);
int result = ((add_fn_t)cjit_get_func(engine, id))(a, b);
```

**Convenience macro (recommended):**

```c
int result = CJIT_DISPATCH(engine, id, add_fn_t, a, b);
// Expands to: record_call + get_func + cast + call
```

`cjit_get_func()` is a single `atomic_load(acquire)` — it is essentially
free on modern CPUs and is the only synchronisation cost on the hot path.

`cjit_record_call()` is a single `atomic_fetch_add(relaxed)` — also
essentially free.  Without it the monitor thread cannot detect hot functions.

### Requesting Recompilation

```c
// Force (re)compilation at a specific tier right now
cjit_request_recompile(engine, id, OPT_O1);

// Optimisation tiers
typedef enum {
    OPT_NONE = 0,   // -O0 – unoptimised baseline
    OPT_O1   = 1,   // -O1 – basic optimisations
    OPT_O2   = 2,   // -O2 – standard + inlining
    OPT_O3   = 3,   // -O3 – aggressive: unroll, vectorise, march=native
} opt_level_t;
```

`cjit_request_recompile()` is a no-op if the function is already enqueued at
an equal or higher tier.  It is useful for **pre-warming** — call it just
after `cjit_start()` to kick off compilation before the first dispatch:

```c
cjit_start(engine);
cjit_request_recompile(engine, id_hot_fn, OPT_O2);
```

### Statistics

```c
// Snapshot of all engine metrics
cjit_stats_t s = cjit_get_stats(engine);

printf("registered: %u\n",   s.registered_functions);
printf("compiled:   %llu\n", (unsigned long long)s.total_compilations);
printf("failed:     %llu\n", (unsigned long long)s.failed_compilations);
printf("IR HOT/WARM/COLD: %u/%u/%u\n",
       s.ir_hot_count, s.ir_warm_count, s.ir_cold_count);
printf("mem pressure: %d  (%llu/%llu MB)\n",
       (int)s.mem_pressure,
       (unsigned long long)s.mem_available_mb,
       (unsigned long long)s.mem_total_mb);

// Or print a pre-formatted block to stderr
cjit_print_stats(engine);

// Per-function helpers
uint64_t    calls = cjit_get_call_count(engine, id);
opt_level_t level = cjit_get_current_opt_level(engine, id);
```

---

## Configuration Reference

Obtain defaults with `cjit_default_config()` and override individual fields:

| Field | Default | Description |
|-------|---------|-------------|
| `max_functions` | 1024 | Maximum concurrently registered functions (≤ `CJIT_MAX_FUNCTIONS`, defined as `1024` in `include/cjit.h`). |
| `compiler_threads` | 3 | Background compilation worker threads. |
| `hot_threshold_t1` | 500 | Call count that triggers tier-1 recompilation (`-O2`). |
| `hot_threshold_t2` | 5000 | Call count that triggers tier-2 recompilation (`-O3`). |
| `grace_period_ms` | 100 | Milliseconds a retired `dlopen` handle is kept alive before `dlclose`. |
| `monitor_interval_ms` | 200 | How often the monitor thread scans call counters (ms). |
| `enable_inlining` | true | Pass `-finline-functions` to the compiler. |
| `enable_vectorization` | true | Pass `-ftree-vectorize` to the compiler. |
| `enable_loop_unroll` | true | Pass `-funroll-loops` to the compiler. |
| `enable_const_fold` | true | Constant folding (active from `-O1`). |
| `enable_native_arch` | true | Pass `-march=native` at `OPT_O3`. |
| `enable_fast_math` | false | Pass `-ffast-math` at `OPT_O3` (may change floating-point semantics). |
| `verbose` | false | Print compilation events to `stderr`. |
| `hot_ir_cache_size` | 64 | Maximum IR entries in the HOT (memory) generation. |
| `warm_ir_cache_size` | 128 | Maximum IR entries in the WARM (memory) generation. |
| `ir_disk_dir` | `""` | On-disk IR directory; empty → auto-created as `/tmp/cjit_ir_<pid>`. |
| `mem_pressure_check_ms` | 500 | `/proc/meminfo` poll interval (ms). |
| `mem_pressure_low_pct` | 20 | `MemAvailable` below this % of `MemTotal` → MEDIUM pressure. |
| `mem_pressure_high_pct` | 10 | `MemAvailable` below this % → HIGH pressure. |
| `mem_pressure_critical_pct` | 5 | `MemAvailable` below this % → CRITICAL pressure. |

### Memory-pressure effect on cache capacity

| Pressure level | Effective HOT capacity | Effective WARM capacity |
|----------------|------------------------|-------------------------|
| NORMAL | 100% | 100% |
| MEDIUM | 75% | 75% |
| HIGH | 50% | 50% |
| CRITICAL | 25% (min 1) | 25% (min 1) |

---

## IR Source Conventions

The codegen backend prepends the following preamble to every translation unit
before passing it to the system C compiler.  IR source code may use these
macros freely without including any headers:

| Macro | Expansion | Purpose |
|-------|-----------|---------|
| `LIKELY(x)` | `__builtin_expect(!!(x), 1)` | Branch-prediction hint (likely true). |
| `UNLIKELY(x)` | `__builtin_expect(!!(x), 0)` | Branch-prediction hint (likely false). |
| `HOT` | `__attribute__((hot))` | Mark function as hot (prioritise in instruction cache). |
| `NOINLINE` | `__attribute__((noinline))` | Prevent inlining. |
| `PURE` | `__attribute__((pure))` | No side effects beyond return value; may read globals. |
| `CONST_FUNC` | `__attribute__((const))` | No side effects; depends only on its arguments. |
| `RESTRICT` | `__restrict__` | Pointer aliasing hint. |
| `PREFETCH(addr,rw,loc)` | `__builtin_prefetch(addr,rw,loc)` | Manual cache prefetch. |
| `ASSUME_ALIGNED(ptr,n)` | `__builtin_assume_aligned(ptr,n)` | Alignment hint for vectoriser. |

Example IR string:

```c
static const char FIB_IR[] =
    "HOT long fib(int n) {\n"
    "    if (UNLIKELY(n <= 1)) return n;\n"
    "    long a = 0, b = 1;\n"
    "    for (int i = 2; i <= n; ++i) {\n"
    "        long tmp = a + b; a = b; b = tmp;\n"
    "    }\n"
    "    return b;\n"
    "}\n";
```

The entire IR string is compiled as a single C translation unit.  You may
include helper functions, `#include` directives, and preprocessor macros as
needed — the compiler sees a complete `.c` file.

---

## Internal Components

### func\_table

**Files:** `src/func_table.h`, `src/func_table.c`

A flat array of `func_table_entry_t` structs indexed by `func_id_t`.  Each
entry is padded to a multiple of 64 bytes to occupy its own cache line(s),
eliminating false sharing between entries accessed by different threads.

Key fields per entry:

| Field | Atomic? | Description |
|-------|---------|-------------|
| `func_ptr` | `_Atomic(jit_func_t)` — acquire/release | Current compiled function pointer. |
| `call_cnt` | `atomic_uint_fast64_t` — relaxed | Call counter for hotness detection. |
| `version` | `atomic_uint_fast32_t` | Recompile generation; stale tasks are discarded. |
| `cur_level` | `atomic_int` | `opt_level_t` of the currently loaded code. |
| `in_queue` | `atomic_bool` | Prevents duplicate enqueue of the same function. |
| `compile_lock` | `pthread_mutex_t` | Serialises concurrent compiles of the same entry. |
| `dl_handle` | plain pointer — protected by `compile_lock` | `dlopen` handle of the loaded `.so`. |
| `ir_source` | plain pointer — set at registration | C-source IR string. |

### work\_queue (MPMC)

**Files:** `src/work_queue.h`, `src/work_queue.c`

A lock-free bounded MPMC (multi-producer / multi-consumer) ring buffer based
on [Dmitry Vyukov's well-known bounded MPMC queue algorithm](https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue).
Capacity is 256 slots (a power-of-two, compile-time constant `WQ_CAPACITY`).

Each slot carries an atomic sequence counter alongside the `compile_task_t`
payload.  Producers and consumers use CAS loops on their respective position
counters.  No ABA problem: the sequence counter is monotonically increasing
(wraps only after 2⁶⁴ operations).

Each slot is padded to 64 bytes to prevent false sharing between adjacent
slots.  The `enqueue_pos` and `dequeue_pos` counters are on separate cache
lines to eliminate producer/consumer false sharing.

A `compile_task_t` carries:

| Field | Description |
|-------|-------------|
| `func_id` | Which function to compile. |
| `target_level` | Desired optimisation tier. |
| `priority` | Currently informational; tasks are processed in FIFO order. |
| `version_req` | IR version that triggered the request; stale tasks are discarded. |

### codegen

**Files:** `src/codegen.h`, `src/codegen.c`

Wraps the system C compiler to produce native shared objects from C source
strings at runtime.

Compilation pipeline per invocation:

1. Write the IR string to a uniquely-named temporary file (named with thread
   ID + monotonic counter to avoid races).
2. Invoke `cc -shared -fPIC -O<N> [opt-flags] -o <output.so> <input.c>` as a
   subprocess via `fork`/`exec`.
3. `dlopen` the output `.so`.
4. `dlsym` to locate the requested function symbol.
5. Unlink both temp files immediately (the kernel keeps the mapping alive
   until `dlclose`; this reclaims disk space promptly).

Optimisation flags by tier:

| Tier | Compiler flags |
|------|---------------|
| `OPT_NONE` | `-O0` |
| `OPT_O1` | `-O1` |
| `OPT_O2` | `-O2 -finline-functions -fno-semantic-interposition` |
| `OPT_O3` | `-O3 -finline-functions -funroll-loops -ftree-vectorize -fomit-frame-pointer -fno-semantic-interposition` + optionally `-march=native` and `-ffast-math` |

`-fomit-frame-pointer` is applied only at `OPT_O3` because it makes stack
traces harder to read and is generally not worthwhile at lower tiers.  If you
need debuggable symbols even at `OPT_O3`, you can set `enable_native_arch` and
`enable_fast_math` to `false` and rely on the system debuginfo rather than
frame pointers, or simply stay at `OPT_O2` for production diagnostics.

Thread safe: each call uses uniquely-named temp files and maintains no shared
mutable state.

### ir\_cache (Multi-generation LRU)

**Files:** `src/ir_cache.h`, `src/ir_cache.c`

Manages the lifecycle of IR strings across three generations:

```
  HOT  (gen 0) – in memory; at most hot_capacity  entries  (MRU/LRU doubly linked list)
  WARM (gen 1) – in memory; at most warm_capacity entries  (MRU/LRU doubly linked list)
  COLD (gen 2) – on disk only; IR string freed from heap
```

A permanent copy of every IR string is written to `ir_disk_dir/<name>.ir` at
registration time so COLD entries can be restored on demand.

**Generation transitions on `ir_cache_get_ir()`:**

| Current generation | Result |
|-------------------|--------|
| HOT | Stay HOT; move to MRU position. |
| WARM | Promote to HOT MRU; cascade LRU-HOT → WARM if HOT is full. |
| COLD | Load from disk; promote to WARM MRU; cascade LRU-WARM → COLD if WARM is full. |

All LRU list mutations are serialised under a single per-cache mutex.  Disk
I/O and `malloc`/`free` are performed **outside** the lock to minimise
contention.

The background pressure-monitor thread updates atomic counters for the
current pressure level and memory readings; these are readable lock-free from
any thread.

### deferred\_gc

**Files:** `src/deferred_gc.h`, `src/deferred_gc.c`

Provides RCU-safe reclamation of retired `dlopen` handles.

**Retire stack** — a classic lock-free LIFO (CAS on head pointer):

```
Push:  new->next = head; CAS(head, old, new)       (retry on failure)
Drain: local = CAS(head, head, NULL)                (single CAS to steal all)
```

**GC thread** wakes every `grace_period_ms / 4` milliseconds, harvests the
entire stack in one CAS, and for each entry:
- If `(now - retire_time) ≥ grace_period_ms` → call `dlclose()`.
- Otherwise → push back onto the stack for the next round.

The grace period guarantees no thread is still executing through old code when
`dlclose()` is called.  Runtime threads pay zero overhead: they perform only
an `atomic_load + indirect call`; no epoch registration, no hazard-pointer
publish.

---

## Thread Safety

| Operation | Mechanism | Threads involved |
|-----------|-----------|-----------------|
| `cjit_get_func()` | `atomic_load(acquire)` | Any runtime thread |
| `cjit_record_call()` | `atomic_fetch_add(relaxed)` | Any runtime thread |
| `func_table_swap()` | `atomic_store(release)` | Compiler threads only |
| `call_cnt` read | `atomic_load(relaxed)` | Monitor thread |
| Enqueue compile task | CAS loop (lock-free) | Monitor thread, `cjit_request_recompile()` |
| Dequeue compile task | CAS loop (lock-free) | Compiler threads |
| IR LRU mutations | Per-cache `pthread_mutex_t` | Compiler threads, pressure thread |
| Pressure level read | `atomic_load` | Any thread |
| Retire handle | CAS loop (wait-free) | Compiler threads |
| `dlclose` | GC thread after grace period | GC thread only |
| Registration | No synchronisation needed | Single-threaded setup only |

`cjit_register_function()` must be called from a **single thread** before
`cjit_start()`.  All other public API functions are safe to call from any
thread at any time while the engine is running.

---

## Performance Tuning

### Hot-path latency

The hot-path cost is:

1. One `atomic_load(acquire)` — typically a few nanoseconds; free on x86-64
   (acquire is free; the store barrier is on the write side).
2. One indirect call through the function pointer.
3. One `atomic_fetch_add(relaxed)` — one clock cycle on most microarchitectures.

For absolute minimum overhead, the hot and cold fields of `func_table_entry_t`
are placed on separate 64-byte cache lines so that the CPU does not bring the
compile metadata into the L1 data cache on every call.

### Choosing thresholds

| Scenario | Suggested settings |
|----------|--------------------|
| Long-running server, many hot functions | `hot_threshold_t1 = 500`, `hot_threshold_t2 = 5000` (defaults) |
| Short-lived process, quick warm-up | `hot_threshold_t1 = 10`, `hot_threshold_t2 = 100` |
| Pre-warm a known-hot function | `cjit_request_recompile(engine, id, OPT_O2)` just after `cjit_start()` |
| Batch workload (compile first, run later) | Increase `compiler_threads`; call `cjit_request_recompile()` for all functions; poll `cjit_get_current_opt_level()` or `cjit_get_stats()` before starting the workload |

### Memory-pressure configuration

If the target machine is memory-constrained, lower `mem_pressure_low_pct` and
`mem_pressure_high_pct` to trigger cache eviction earlier.  Increase
`hot_ir_cache_size` and `warm_ir_cache_size` if the number of registered
functions is large and disk I/O is expensive.

### Grace period

The default `grace_period_ms = 100` is conservative.  If your JIT-compiled
functions are short (< 1 ms), you can safely reduce this to 10–20 ms to
reclaim shared-object memory faster.  If functions may run for seconds
(e.g. heavy compute kernels), increase it accordingly or use hazard pointers
instead.

### Compiler threads

Three threads (`CJIT_COMPILER_THREADS = 3`) provide a good balance between
compilation throughput and CPU impact on a busy server.  For a batch warm-up
phase you can temporarily increase `compiler_threads` to match the number of
available CPU cores.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| C11 compiler | GCC ≥ 5 or Clang ≥ 3.6; `_Atomic` and `stdatomic.h` required |
| POSIX threads | `libpthread` |
| `dlopen` / `dlsym` | `libdl` |
| CMake ≥ 3.14 | Build system |
| System C compiler (`cc`) | Present at **runtime** for JIT compilation; may be GCC, Clang, or any compatible compiler |
| Linux | `/proc/meminfo` is read for memory-pressure monitoring; the rest of the code is portable POSIX |