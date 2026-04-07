cmake_minimum_required(VERSION 3.14)
project(cjit C)

# ─────────────────────────── compiler settings ────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)   # need _GNU_SOURCE for pthread_self, etc.

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Flags
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter        # IR source strings contain unused helpers
    -pthread
)

# ─────────────────────────── source files ─────────────────────────────────
set(CJIT_SOURCES
    src/work_queue.c
    src/deferred_gc.c
    src/func_table.c
    src/codegen.c
    src/ir_cache.c
    src/cjit.c
)

# ─────────────────────────── library target ───────────────────────────────
add_library(cjit_lib STATIC ${CJIT_SOURCES})
target_include_directories(cjit_lib PUBLIC include)
target_link_libraries(cjit_lib PUBLIC pthread dl)
target_compile_definitions(cjit_lib PRIVATE _GNU_SOURCE)

# ─────────────────────────── demo executable ──────────────────────────────
add_executable(demo examples/demo.c)
target_link_libraries(demo PRIVATE cjit_lib)
target_compile_definitions(demo PRIVATE _GNU_SOURCE)

# ─────────────────────────── cjit CLI ─────────────────────────────────────
add_executable(cjit src/cli.c)
target_link_libraries(cjit PRIVATE cjit_lib)
target_compile_definitions(cjit PRIVATE _GNU_SOURCE)

# ─────────────────────────── integration tests ────────────────────────────
add_executable(test_jit tests/test_jit.c)
target_link_libraries(test_jit PRIVATE cjit_lib)
target_compile_definitions(test_jit PRIVATE _GNU_SOURCE)

enable_testing()
add_test(NAME jit_suite COMMAND test_jit)

# ─────────────────────────── install ──────────────────────────────────────
install(TARGETS cjit RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/cjit)
