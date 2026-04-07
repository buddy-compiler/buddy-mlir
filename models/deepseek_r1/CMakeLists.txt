# ══════════════════════════════════════════════════════════════════════════════
# DeepSeek R1 — single build path (buddy-codegen)
#
# Variant JSON → gen_config + ModelSession + RHAL manifest + import_model +
# compile_pipeline + .so + .rax  (see tools/buddy-codegen/)
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_DSR1_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/w8a16.json")
set(BUDDY_DSR1_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
set(BUDDY_DSR1_COMPILE_JOBS "1"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

# Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
set(DEEPSEEKR1_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(DEEPSEEKR1_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

buddy_add_model(
  NAME deepseek_r1
  SPEC "${BUDDY_DSR1_SPEC}"
  RUNNER_SRC DeepSeekR1Runner.cpp
  HF_CONFIG "${BUDDY_DSR1_HF_CONFIG}"
  BUILD_DIR "${DEEPSEEKR1_BUILD_DIR}"
  MLIR_DIR "${DEEPSEEKR1_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_DSR1_COMPILE_JOBS}"
)
