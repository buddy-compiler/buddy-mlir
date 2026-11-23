# tools/buddy-bench/CMakeLists.txt

# Pure helpers (no model / runtime state) shared with the unit test.
add_library(buddy-bench-utils STATIC
  BenchUtils.cpp
  Reporter.cpp
  Stats.cpp)
target_compile_features(buddy-bench-utils PUBLIC cxx_std_17)
target_include_directories(buddy-bench-utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(buddy-bench-utils PUBLIC
  buddy_runtime_core
  CLI11
)

add_executable(buddy-bench
  buddy-bench.cpp
  BenchBackend.cpp
  Affinity.cpp)

target_compile_features(buddy-bench PRIVATE cxx_std_17)
target_compile_definitions(buddy-bench PRIVATE
  BUDDY_VERSION="${BUDDY_PACKAGE_VERSION}")

# BenchBackend dlopens the model runner plugin, which resolves its symbols
# against the runtime linked below — same link set as buddy-cli.
target_link_libraries(buddy-bench PRIVATE
  buddy-bench-utils
  buddy_runtime_core
  buddy_runtime_llm
  buddy_runtime_rax_dependencies
  CLI11
  ${CMAKE_DL_LIBS}
)

# ── libnuma (optional; records NUMA config and enables --numa-* flags) ─────
# Mirrors buddy-cli: off by default in cross builds, found on the host in
# native builds. buddy-bench degrades gracefully (records "n/a") without it.
if(CMAKE_CROSSCOMPILING)
  set(_buddy_bench_numa_default OFF)
else()
  set(_buddy_bench_numa_default ON)
endif()
option(BUDDY_BENCH_ENABLE_NUMA
  "Use libnuma in buddy-bench when it is found (--numa-* flags)"
  ${_buddy_bench_numa_default})
if(BUDDY_BENCH_ENABLE_NUMA)
  find_library(NUMA_LIB numa)
  find_path(NUMA_INCLUDE_DIR numa.h)
  if(NUMA_LIB AND NUMA_INCLUDE_DIR)
    target_compile_definitions(buddy-bench PRIVATE BUDDY_BENCH_HAVE_NUMA)
    target_include_directories(buddy-bench PRIVATE ${NUMA_INCLUDE_DIR})
    target_link_libraries(buddy-bench PRIVATE ${NUMA_LIB})
  endif()
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-bench PRIVATE mimalloc)
endif()

install(TARGETS buddy-bench
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)
