# Qwen3-VL multimodal OCR integration.
#
# Uses the shared buddy-codegen model entry point. The model still needs a
# custom packaged build shape because it compiles separate vision and decoder
# shims, then lets the runner orchestrate preprocessing, embedding splice, and
# decode.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_QWEN3_VL_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/instruct_2b.json"
  CACHE FILEPATH "Qwen3-VL variant spec JSON")
set(BUDDY_QWEN3_VL_MODEL_PATH "" CACHE PATH
  "Local HuggingFace Qwen3-VL-2B-Instruct directory")
set(BUDDY_QWEN3_VL_NUM_THREADS "1" CACHE STRING
  "OpenMP thread count baked into Qwen3-VL compiled kernels")
set(BUDDY_QWEN3_VL_LLC_ATTRS "" CACHE STRING
  "Optional Qwen3-VL LLC target attributes; empty uses buddy_add_model platform defaults")

buddy_add_model(
  NAME qwen3_vl
  MODEL_KIND qwen3_vl_multimodal
  SPEC "${BUDDY_QWEN3_VL_SPEC}"
  RUNNER_SRC Qwen3VLRunner.cpp
  RUNNER_PLUGIN_SRC Qwen3VLRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_QWEN3_VL_MODEL_PATH}"
  NUM_THREADS "${BUDDY_QWEN3_VL_NUM_THREADS}"
  LLC_ATTRS "${BUDDY_QWEN3_VL_LLC_ATTRS}"
)
