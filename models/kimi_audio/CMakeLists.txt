# Kimi-Audio-7B-Instruct single-forward model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# Kimi-Audio keeps a custom import (codegen/import-kimi_audio.py) and manifest
# generator (codegen/gen_kimi_audio_manifest.py) because the model needs remote
# code patching for CPU fullgraph tracing and its forward is a single fixed-shape
# pass over the whole sequence.
#
# No ASSET_FILES is declared: buddy_add_model stages the shared Qwen
# examples/BuddyDeepSeekR1/vocab.txt by default, which is the byte-level BPE
# vocabulary Kimi-Audio's Qwen2.5-style tokenizer needs (the HF snapshot ships
# no tokenizer files).

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_KIMI_AUDIO_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Kimi-Audio variant spec JSON, e.g. specs/f32.json")
set(BUDDY_KIMI_AUDIO_MODEL_PATH "" CACHE PATH
  "Local HuggingFace moonshotai/Kimi-Audio-7B-Instruct snapshot dir")

buddy_add_model(
  NAME kimi_audio
  MODEL_KIND single_forward
  SPEC "${BUDDY_KIMI_AUDIO_SPEC}"
  RUNNER_SRC KimiAudioRunner.cpp
  RUNNER_PLUGIN_SRC KimiAudioRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/KimiAudioRunner.h
  LOCAL_MODEL "${BUDDY_KIMI_AUDIO_MODEL_PATH}"
  LOCAL_MODEL_ENV KIMI_AUDIO_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-kimi_audio.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_kimi_audio_manifest.py"
  MODEL_SO_NAME kimi_audio_model.so
)
