# ProteinGLM masked language model integration.
#
# ProteinGLM uses a single forward graph, so it keeps custom import and
# manifest scripts while reusing buddy_add_model for runner/plugin/.rax wiring.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_PROTEINGLM_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/base.json"
  CACHE FILEPATH "ProteinGLM variant spec JSON")
set(BUDDY_PROTEINGLM_MODEL_PATH ""
  CACHE PATH "Local HuggingFace ProteinGLM model directory")

if(NOT BUDDY_PROTEINGLM_MODEL_PATH)
  message(FATAL_ERROR
    "BUDDY_BUILD_PROTEINGLM_MODEL=ON requires "
    "-DBUDDY_PROTEINGLM_MODEL_PATH=/path/to/proteinglm-1b-mlm")
endif()

buddy_add_model(
  NAME proteinglm
  MODEL_KIND single_forward
  SPEC "${BUDDY_PROTEINGLM_SPEC}"
  RUNNER_SRC ProteinGLMRunner.cpp
  RUNNER_PLUGIN_SRC ProteinGLMRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/ProteinGLMRunner.h
  LOCAL_MODEL "${BUDDY_PROTEINGLM_MODEL_PATH}"
  LOCAL_MODEL_ENV PROTEINGLM_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-proteinglm.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_proteinglm_manifest.py"
  MODEL_SO_NAME proteinglm_model.so
  ASSET_FILES
    "${BUDDY_PROTEINGLM_MODEL_PATH}/tokenizer.model"
    "${BUDDY_PROTEINGLM_MODEL_PATH}/tokenizer_config.json"
    "${BUDDY_PROTEINGLM_MODEL_PATH}/special_tokens_map.json"
)
