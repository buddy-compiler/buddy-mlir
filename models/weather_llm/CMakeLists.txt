# ══════════════════════════════════════════════════════════════════════════════
# Weather-LLM — LlamaForCausalLM single build path (buddy-codegen)
#
# Variant JSON → gen_config + ModelSession + RHAL manifest + import_model +
# compile_pipeline + .so + .rax  (see tools/buddy-codegen/)
#
# Part 1 (shape / compile-time constants): values come from the chosen
# specs/<variant>.json via gen_config.py (not hand-edited CMake cache vars).
#   - model_family `weather_llm` → macros BUDDY_WEATHER_LLM_* and
#     so_name weather_llm_model.so.
# Part 2 (linking the static model library): buddy_model.cmake links
#   buddy_runtime_core + buddy_runtime_llm + CMAKE_DL_LIBS (-ldl) + LLVMSupport
# Part 3 (MLIR pipeline): Pipe is roughly
#   buddy-opt → mlir-opt (TOSA) → buddy-opt (bufferize + vectorize + lower) →
#   mlir-translate → llvm-as → llc → .o → link .so → gen_manifest → rax-pack → .rax
#   Mode A: pre-built .o | Mode B: pre-generated MLIR | Mode C: import_model.py
# Output .rax, weather_llm_runner.so and vocab.txt live next to the .so so
# manifest URIs like file:model.so and file:vocab.txt resolve relative to the
# .rax directory.
#
# The vocab.txt here is a token-per-line conversion of the model's custom
# SentencePiece tokenizer (tokenizer.model), which is what tokenizeLlama() in
# frontend/Interfaces/buddy/LLM/TextContainer.h expects. It is staged next to
# the .rax via ASSET_FILES so the manifest's file:vocab.txt URI resolves to the
# correct vocabulary.
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_WEATHER_LLM_DEFAULT_COMPILE_JOBS)
if(_BUDDY_WEATHER_LLM_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_WEATHER_LLM_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_WEATHER_LLM_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/f32.json")
set(BUDDY_WEATHER_LLM_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
set(BUDDY_WEATHER_LLM_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory (weights import; sets DEEPSEEKR1_MODEL_PATH)")
set(BUDDY_WEATHER_LLM_COMPILE_JOBS "${_BUDDY_WEATHER_LLM_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

# Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
set(WEATHER_LLM_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(WEATHER_LLM_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

buddy_add_model(
  NAME weather_llm
  SPEC "${BUDDY_WEATHER_LLM_SPEC}"
  RUNNER_SRC WeatherRunner.cpp
  RUNNER_PLUGIN_SRC WeatherRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/WeatherRunner.h
  HF_CONFIG "${BUDDY_WEATHER_LLM_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_WEATHER_LLM_LOCAL_MODEL}"
  BUILD_DIR "${WEATHER_LLM_BUILD_DIR}"
  MLIR_DIR "${WEATHER_LLM_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_WEATHER_LLM_COMPILE_JOBS}"
  ASSET_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/vocab.txt"
)
