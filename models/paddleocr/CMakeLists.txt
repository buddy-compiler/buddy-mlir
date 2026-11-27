# PaddleOCR-VL OCR vision-language model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# PaddleOCR-VL keeps a custom import and manifest generator (like ColBERTv2)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_PADDLEOCR_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "PaddleOCR-VL variant spec JSON, e.g. specs/f32.json")
set(BUDDY_PADDLEOCR_MODEL_PATH "" CACHE PATH
  "Local HuggingFace lvyufeng/PaddleOCR-VL-0.9B snapshot dir")
option(BUDDY_PADDLEOCR_K3_IME
  "Use SpacemiT K3 A100 FP16 IME lowering for PaddleOCR-VL"
  OFF)

file(READ "${BUDDY_PADDLEOCR_SPEC}" _paddleocr_spec_json)
string(JSON _paddleocr_f32_aux_size ERROR_VARIABLE _paddleocr_json_error
  GET "${_paddleocr_spec_json}" params_f32_size)
if(NOT _paddleocr_json_error AND _paddleocr_f32_aux_size GREATER 0)
  set(_paddleocr_extra_weights arg1.data)
endif()

set(_paddleocr_ime OFF)
if(BUDDY_PADDLEOCR_K3_IME)
  set(_paddleocr_ime ON)
else()
  string(JSON _paddleocr_spec_ime ERROR_VARIABLE _paddleocr_ime_err
    GET "${_paddleocr_spec_json}" ime)
  if(NOT _paddleocr_ime_err AND _paddleocr_spec_ime STREQUAL "ON")
    set(_paddleocr_ime ON)
  elseif(NOT _paddleocr_ime_err AND _paddleocr_spec_ime STREQUAL "true")
    set(_paddleocr_ime ON)
  endif()
endif()

set(_paddleocr_ime_marker "${CMAKE_CURRENT_BINARY_DIR}/.paddleocr_ime_enabled")
if(_paddleocr_ime)
  if(NOT IS_RVV_CROSSCOMPILE AND NOT HAVE_LOCAL_RVV)
    message(FATAL_ERROR
      "BUDDY_PADDLEOCR_K3_IME requires RISC-V cross-compilation or local RVV")
  endif()
  if(NOT BUDDY_RISCV_VLEN STREQUAL "1024")
    message(FATAL_ERROR
      "BUDDY_PADDLEOCR_K3_IME requires BUDDY_RISCV_VLEN=1024; got '${BUDDY_RISCV_VLEN}'")
  endif()
  string(JSON _paddleocr_precision ERROR_VARIABLE _paddleocr_prec_err
    GET "${_paddleocr_spec_json}" precision)
  if(_paddleocr_prec_err OR NOT _paddleocr_precision STREQUAL "f16")
    message(FATAL_ERROR
      "BUDDY_PADDLEOCR_K3_IME requires a spec with precision=f16")
  endif()
  file(WRITE "${_paddleocr_ime_marker}" "ON\n")
else()
  file(REMOVE "${_paddleocr_ime_marker}")
endif()

string(JSON _paddleocr_num_threads ERROR_VARIABLE _paddleocr_threads_err
  GET "${_paddleocr_spec_json}" num_threads)
if(_paddleocr_threads_err OR NOT _paddleocr_num_threads)
  set(_paddleocr_num_threads 1)
endif()

buddy_add_model(
  NAME paddleocr
  MODEL_KIND single_forward
  SPEC "${BUDDY_PADDLEOCR_SPEC}"
  NUM_THREADS "${_paddleocr_num_threads}"
  RUNNER_SRC PaddleocrRunner.cpp
  RUNNER_PLUGIN_SRC PaddleocrRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/PaddleocrRunner.h
  LOCAL_MODEL "${BUDDY_PADDLEOCR_MODEL_PATH}"
  LOCAL_MODEL_ENV PADDLEOCR_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import_paddleocr.py"
  LOWER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_paddleocr_manifest.py"
  MODEL_SO_NAME paddleocr_model.so
  EXTRA_WEIGHT_FILES ${_paddleocr_extra_weights}
  ASSET_FILES
    "${BUDDY_PADDLEOCR_MODEL_PATH}/tokenizer.json"
)
