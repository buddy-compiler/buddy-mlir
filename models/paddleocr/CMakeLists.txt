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

buddy_add_model(
  NAME paddleocr
  MODEL_KIND single_forward
  SPEC "${BUDDY_PADDLEOCR_SPEC}"
  RUNNER_SRC PaddleocrRunner.cpp
  RUNNER_PLUGIN_SRC PaddleocrRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/PaddleocrRunner.h
  LOCAL_MODEL "${BUDDY_PADDLEOCR_MODEL_PATH}"
  LOCAL_MODEL_ENV PADDLEOCR_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-paddleocr.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_paddleocr_manifest.py"
  MODEL_SO_NAME paddleocr_model.so
  ASSET_FILES
    "${BUDDY_PADDLEOCR_MODEL_PATH}/tokenizer.json"
)
