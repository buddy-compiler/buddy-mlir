# -----------------------------------------------------------------------------
# External-call kernel libraries used by Python-driven execution.
# -----------------------------------------------------------------------------
add_subdirectory(external_kernels)

# RAX schema + FlatBuffers interface target (used by rax-pack, rax-inspect,
# and the model-agnostic runtime core).
add_subdirectory(rax)

# Model-agnostic runtime headers: InferenceRunner, ModelManifest, BufferPool.
# Public includes live under runtime/include/ → #include "buddy/runtime/core/*.h"
# (CMake stays here — not under include/, which should remain installable headers only.)
add_library(buddy_runtime_core INTERFACE)
target_include_directories(buddy_runtime_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/buddy-mlir>
)
target_link_libraries(buddy_runtime_core INTERFACE
  buddy_rax
)
target_compile_features(buddy_runtime_core INTERFACE cxx_std_17)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include/buddy-mlir
        COMPONENT buddy_runtime)
install(TARGETS buddy_runtime_core
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_runtime)

# LLM runtime utilities (text generation, interactive REPL).
add_subdirectory(llm)
