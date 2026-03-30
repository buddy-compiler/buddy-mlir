# -----------------------------------------------------------------------------
# Runtime helper libraries (shared) for Python-driven execution.
# -----------------------------------------------------------------------------

add_library(buddy_rng_utils SHARED buddy_rng_utils.cpp)
target_compile_features(buddy_rng_utils PRIVATE cxx_std_17)
set_target_properties(
  buddy_rng_utils
  PROPERTIES
    OUTPUT_NAME "buddy_rng_utils"
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
)

# RAX schema + FlatBuffers interface target (used by rax-pack, rax-inspect,
# and the model-agnostic runtime core).
add_subdirectory(rax)

# Model-agnostic runtime headers: InferenceRunner, ModelManifest, BufferPool.
# Public includes live under runtime/include/ → #include "buddy/runtime/core/*.h"
# (CMake stays here — not under include/, which should remain installable headers only.)
add_library(buddy_runtime_core INTERFACE)
target_include_directories(buddy_runtime_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(buddy_runtime_core INTERFACE
  buddy_rax
)
target_compile_features(buddy_runtime_core INTERFACE cxx_std_17)

# LLM runtime utilities (text generation, interactive REPL).
add_subdirectory(llm)
