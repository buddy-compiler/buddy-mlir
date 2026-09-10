# MMS (facebook/mms-tts-eng) vocoder-stage integration.
#
# single_forward model. The compiled kernel is the HiFi-GAN vocoder stage of
# the VITS model (spectrogram latent -> waveform).  The full text-to-speech
# pipeline cannot be AOT-traced into one static graph (data-dependent duration
# alignment), see codegen/import-mms.py and README.md.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_MMS_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "MMS variant spec JSON, e.g. specs/f32.json")
set(BUDDY_MMS_MODEL_PATH "" CACHE PATH
  "Local HuggingFace facebook/mms-tts-eng snapshot dir")

buddy_add_model(
  NAME mms
  MODEL_KIND single_forward
  SPEC "${BUDDY_MMS_SPEC}"
  RUNNER_SRC MMSRunner.cpp
  RUNNER_PLUGIN_SRC MMSRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_MMS_MODEL_PATH}"
  LOCAL_MODEL_ENV MMS_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-mms.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_mms_manifest.py"
  MODEL_SO_NAME mms_model.so
)
