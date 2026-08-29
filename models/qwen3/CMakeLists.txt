# ══════════════════════════════════════════════════════════════════════════════
# Qwen3 (Qwen/Qwen3-0.6B) — single build path (buddy-codegen)
#
# Variant JSON → gen_config + ModelSession + RHAL manifest + import_model +
# compile_pipeline + .so + .rax  (see tools/buddy-codegen/)
#
# Part 1 (shape / compile-time constants): values come from the chosen
# specs/<variant>.json via gen_config.py (not hand-edited CMake cache vars).
# Default --max-tokens / RunConfig::maxNewTokens should stay aligned with
# buddy::runtime::RunConfig in InferenceRunner.h and buddy-cli defaults.
#
# Part 2 (linking the static model library): buddy_model.cmake links
#   buddy_runtime_core + buddy_runtime_llm + CMAKE_DL_LIBS (-ldl) + LLVMSupport
# Part 3 (MLIR pipeline): Pipe is roughly
#   buddy-opt → mlir-opt (TOSA) → buddy-opt (bufferize + vectorize + lower) →
#   mlir-translate → llvm-as → llc → .o → link .so → gen_manifest → rax-pack → .rax
#   Mode A: pre-built .o | Mode B: pre-generated MLIR | Mode C: import_model.py
# Output .rax and vocab.txt live next to the .so so manifest URIs like
# file:model.so and file:vocab.txt resolve relative to the .rax directory.
#
# LOCAL_MODEL (BUDDY_QWEN3_LOCAL_MODEL) is forwarded to the generic importer as
# the DEEPSEEKR1_MODEL_PATH environment variable, i.e. a local HF snapshot dir
# with model.safetensors / vocab files (import_model.py reads it for weights).
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
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
set(BUDDY_QWEN3_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory (weights import; sets DEEPSEEKR1_MODEL_PATH)")
set(BUDDY_QWEN3_COMPILE_JOBS "${_BUDDY_QWEN3_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

# Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
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
  RUNNER_PLUGIN_SRC Qwen3RunnerPlugin.cpp
  HF_CONFIG "${BUDDY_QWEN3_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_QWEN3_LOCAL_MODEL}"
  BUILD_DIR "${QWEN3_BUILD_DIR}"
  MLIR_DIR "${QWEN3_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_QWEN3_COMPILE_JOBS}"
)
