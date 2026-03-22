# tools/buddy-cli/CMakeLists.txt

add_executable(buddy-cli buddy-cli.cpp)

target_compile_features(buddy-cli PRIVATE cxx_std_17)

target_link_libraries(buddy-cli PRIVATE
  buddy_models_deepseekr1
  mlir_c_runner_utils
)
target_link_directories(buddy-cli PRIVATE ${LLVM_LIBRARY_DIR})

# ── libnuma (optional but strongly recommended on NUMA systems) ────────────
find_library(NUMA_LIB numa)
if(NUMA_LIB)
  target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_NUMA)
  target_link_libraries(buddy-cli PRIVATE ${NUMA_LIB})
  message(STATUS "[buddy-cli] libnuma found: ${NUMA_LIB} — NUMA flags enabled")
else()
  message(STATUS "[buddy-cli] libnuma not found — --numa-* flags will warn at runtime")
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-cli PRIVATE mimalloc)
endif()
