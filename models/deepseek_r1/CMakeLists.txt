# ══════════════════════════════════════════════════════════════════════════════
# DeepSeek R1 — single build path (buddy-codegen)
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
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS)
if(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_DSR1_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/w8a16.json")
set(BUDDY_DSR1_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
set(BUDDY_DSR1_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory (weights import; sets DEEPSEEKR1_MODEL_PATH)")
set(BUDDY_DSR1_COMPILE_JOBS "${_BUDDY_DSR1_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")
set(BUDDY_DSR1_TIERED_KV_CACHE ""
  CACHE STRING "Auto/ON/OFF: build DeepSeek R1 with tiered KV cache entrypoints")
set(BUDDY_DSR1_TIERED_CACHE_SIZES ""
  CACHE STRING "Semicolon-separated tiered KV cache sizes; auto-detected from spec if empty")

# Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
set(DEEPSEEKR1_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(DEEPSEEKR1_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import json,sys; s=json.load(open(sys.argv[1])); raw=s.get('tiered_kv_cache', False); enabled=False; sizes=[]\nif isinstance(raw, dict):\n  enabled=bool(raw.get('enabled', True)); sizes=raw.get('cache_sizes', s.get('cache_sizes', []))\nelse:\n  enabled=bool(raw); sizes=s.get('cache_sizes', [])\nif isinstance(sizes, str): sizes=[x.strip() for x in sizes.split(',') if x.strip()]\nif enabled and not sizes: sizes=[32,64,128,256,512,1024]\nprint(('ON' if enabled else 'OFF') + '|' + ','.join(str(x) for x in sizes))"
          "${BUDDY_DSR1_SPEC}"
  OUTPUT_VARIABLE _BUDDY_DSR1_TIERED_FROM_SPEC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _BUDDY_DSR1_TIERED_PARSE_RESULT)

set(_BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC OFF)
set(_BUDDY_DSR1_TIERED_SIZES_FROM_SPEC "")
if(_BUDDY_DSR1_TIERED_PARSE_RESULT EQUAL 0)
  string(REPLACE "|" ";" _BUDDY_DSR1_TIERED_FIELDS
         "${_BUDDY_DSR1_TIERED_FROM_SPEC}")
  list(GET _BUDDY_DSR1_TIERED_FIELDS 0 _BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC)
  list(LENGTH _BUDDY_DSR1_TIERED_FIELDS _BUDDY_DSR1_TIERED_FIELD_COUNT)
  if(_BUDDY_DSR1_TIERED_FIELD_COUNT GREATER 1)
    list(GET _BUDDY_DSR1_TIERED_FIELDS 1 _BUDDY_DSR1_TIERED_SIZES_CSV)
    string(REPLACE "," ";" _BUDDY_DSR1_TIERED_SIZES_FROM_SPEC
           "${_BUDDY_DSR1_TIERED_SIZES_CSV}")
  endif()
endif()

if(BUDDY_DSR1_TIERED_KV_CACHE STREQUAL "")
  set(_BUDDY_DSR1_TIERED_ENABLED "${_BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC}")
else()
  set(_BUDDY_DSR1_TIERED_ENABLED "${BUDDY_DSR1_TIERED_KV_CACHE}")
endif()

if(BUDDY_DSR1_TIERED_CACHE_SIZES STREQUAL "")
  set(_BUDDY_DSR1_TIERED_CACHE_SIZES
      "${_BUDDY_DSR1_TIERED_SIZES_FROM_SPEC}")
else()
  set(_BUDDY_DSR1_TIERED_CACHE_SIZES "${BUDDY_DSR1_TIERED_CACHE_SIZES}")
endif()

buddy_add_model(
  NAME deepseek_r1
  SPEC "${BUDDY_DSR1_SPEC}"
  RUNNER_SRC DeepSeekR1Runner.cpp
  RUNNER_PLUGIN_SRC DeepSeekR1RunnerPlugin.cpp
  SERVING_PLUGIN_SRC DeepSeekR1ResidentModelPlugin.cpp
  EXTRA_SRCS DeepSeekR1ResidentModel.cpp
  HF_CONFIG "${BUDDY_DSR1_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_DSR1_LOCAL_MODEL}"
  BUILD_DIR "${DEEPSEEKR1_BUILD_DIR}"
  MLIR_DIR "${DEEPSEEKR1_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_DSR1_COMPILE_JOBS}"
  TIERED_KV_CACHE "${_BUDDY_DSR1_TIERED_ENABLED}"
  TIERED_CACHE_SIZES "${_BUDDY_DSR1_TIERED_CACHE_SIZES}"
)
