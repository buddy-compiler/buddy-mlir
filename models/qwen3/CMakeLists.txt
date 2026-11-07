include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_QWEN3_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/specs/f16.json"
    CACHE FILEPATH "Qwen3 buddy-codegen specification")
set(BUDDY_QWEN3_LOCAL_MODEL "" CACHE PATH "Local Qwen3 HuggingFace checkpoint")
set(BUDDY_QWEN3_HF_CONFIG "" CACHE FILEPATH "Qwen3 HuggingFace config.json")
set(QWEN3_MLIR_DIR "" CACHE PATH "Optional pre-generated Qwen3 MLIR")
set(QWEN3_BUILD_DIR "" CACHE PATH "Optional precompiled Qwen3 objects")

buddy_add_model(
  NAME qwen3
  SPEC "${BUDDY_QWEN3_SPEC}"
  RUNNER_SRC Qwen3Runner.cpp
  RUNNER_PLUGIN_SRC Qwen3RunnerPlugin.cpp
  HF_CONFIG "${BUDDY_QWEN3_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_QWEN3_LOCAL_MODEL}"
  BUILD_DIR "${QWEN3_BUILD_DIR}"
  MLIR_DIR "${QWEN3_MLIR_DIR}"
  ASSET_FILES "${CMAKE_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt"
)
