# ══════════════════════════════════════════════════════════════════════════════
# Weather-LLM — LlamaForCausalLM Build Path (buddy-codegen)
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
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py")
set(BUDDY_WEATHER_LLM_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory for weights import")
set(BUDDY_WEATHER_LLM_COMPILE_JOBS "${_BUDDY_WEATHER_LLM_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

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
  HF_CONFIG "${BUDDY_WEATHER_LLM_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_WEATHER_LLM_LOCAL_MODEL}"
  BUILD_DIR "${WEATHER_LLM_BUILD_DIR}"
  MLIR_DIR "${WEATHER_LLM_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_WEATHER_LLM_COMPILE_JOBS}"
)
