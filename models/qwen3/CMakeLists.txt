# ══════════════════════════════════════════════════════════════════════════════
# Qwen3 — Single Build Path (buddy-codegen)
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_QWEN3_DEFAULT_COMPILE_JOBS)
if(_BUDDY_QWEN3_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_QWEN3_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_QWEN3_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/f32.json")
set(BUDDY_QWEN3_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py")
set(BUDDY_QWEN3_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory for weights import")
set(BUDDY_QWEN3_COMPILE_JOBS "${_BUDDY_QWEN3_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

set(QWEN3_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(QWEN3_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

buddy_add_model(
  NAME qwen3
  SPEC "${BUDDY_QWEN3_SPEC}"
  RUNNER_SRC Qwen3Runner.cpp
  HF_CONFIG "${BUDDY_QWEN3_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_QWEN3_LOCAL_MODEL}"
  BUILD_DIR "${QWEN3_BUILD_DIR}"
  MLIR_DIR "${QWEN3_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_QWEN3_COMPILE_JOBS}"
)
