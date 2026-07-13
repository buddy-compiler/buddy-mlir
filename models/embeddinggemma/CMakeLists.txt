# ══════════════════════════════════════════════════════════════════════════════
# embeddinggemma-300m — Retrieval/Embedding Model Build Path
#
# Unlike LLM models (prefill/decode), embeddinggemma produces a single
# forward graph (text → 768-dim embedding).  The build uses buddy_add_model
# with a single forward graph mapped to the prefill pipeline.
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_EMBEDDINGGEMMA_DEFAULT_COMPILE_JOBS)
if(_BUDDY_EMBEDDINGGEMMA_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_EMBEDDINGGEMMA_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_EMBEDDINGGEMMA_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/f32.json")
set(BUDDY_EMBEDDINGGEMMA_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py")
set(BUDDY_EMBEDDINGGEMMA_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory for weights import")
set(BUDDY_EMBEDDINGGEMMA_COMPILE_JOBS "${_BUDDY_EMBEDDINGGEMMA_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

set(EMBEDDINGGEMMA_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(EMBEDDINGGEMMA_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

buddy_add_model(
  NAME embeddinggemma
  SPEC "${BUDDY_EMBEDDINGGEMMA_SPEC}"
  RUNNER_SRC EmbeddingGemmaRunner.cpp
  HF_CONFIG "${BUDDY_EMBEDDINGGEMMA_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_EMBEDDINGGEMMA_LOCAL_MODEL}"
  BUILD_DIR "${EMBEDDINGGEMMA_BUILD_DIR}"
  MLIR_DIR "${EMBEDDINGGEMMA_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_EMBEDDINGGEMMA_COMPILE_JOBS}"
)
