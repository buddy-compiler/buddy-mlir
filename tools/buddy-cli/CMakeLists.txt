# tools/buddy-cli/CMakeLists.txt

add_executable(buddy-cli buddy-cli.cpp)

target_compile_features(buddy-cli PRIVATE cxx_std_17)

# buddy_models_<name> is produced by buddy_add_model(NAME <name> …) in models/
if(BUDDY_BUILD_DEEPSEEK_R1_MODEL)
  target_link_libraries(buddy-cli PRIVATE buddy_models_deepseek_r1)
  target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL)
else()
  target_link_libraries(buddy-cli PRIVATE
    buddy_runtime_core
    buddy_runtime_llm
    ${CMAKE_DL_LIBS}
  )
endif()
target_link_libraries(buddy-cli PRIVATE
  mlir_c_runner_utils
  LLVMSupport
)
target_link_directories(buddy-cli PRIVATE ${LLVM_LIBRARY_DIR})

# ── libnuma (optional but strongly recommended on NUMA systems) ────────────
find_library(NUMA_LIB numa)
if(NUMA_LIB)
  target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_NUMA)
  target_link_libraries(buddy-cli PRIVATE ${NUMA_LIB})
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-cli PRIVATE mimalloc)
endif()

install(TARGETS buddy-cli
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)
