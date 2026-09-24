#!/usr/bin/env bash
# ===- set_model_env.sh ------------------------------------------------------
#
# Helper script to set environment variables for vision/multimodal graph
# coverage tests. Each model can be pointed to a local path for offline testing.
#
# Usage:
#   1. Edit the paths below to point to your local model directories
#   2. Source this script: source set_model_env.sh
#   3. Run tests: python tests/Models/BuddyVisionGraphImport/test_import_clip_vit_base_patch32.py
#
# ===---------------------------------------------------------------------------

# --- Edit these paths to your local model directories ---

export CLIP_VIT_BASE_MODEL_PATH="$HOME/model/clip-vit-base-patch32"
export MOBILEVIT_SMALL_MODEL_PATH="$HOME/model/mobilevit-small"
export DINOV2_BASE_MODEL_PATH="$HOME/model/dinov2-base"
export SMOLVLM_256M_MODEL_PATH="$HOME/model/SmolVLM-256M-Instruct"
export QWEN3_VL_2B_MODEL_PATH="$HOME/model/Qwen3-VL-2B-Instruct"
export LLAVA_1.5_7B_MODEL_PATH="$HOME/model/llava-1.5-7b-hf"

# --- Print configured paths ---

echo "Vision Graph Coverage Test - Model Paths:"
echo "  CLIP_VIT_BASE_MODEL_PATH    = $CLIP_VIT_BASE_MODEL_PATH"
echo "  MOBILEVIT_SMALL_MODEL_PATH  = $MOBILEVIT_SMALL_MODEL_PATH"
echo "  DINOV2_BASE_MODEL_PATH      = $DINOV2_BASE_MODEL_PATH"
echo "  SMOLVLM_256M_MODEL_PATH     = $SMOLVLM_256M_MODEL_PATH"
echo "  QWEN3_VL_2B_MODEL_PATH      = $QWEN3_VL_2B_MODEL_PATH"
echo "  LLAVA_1.5_7B_MODEL_PATH     = $LLAVA_1.5_7B_MODEL_PATH"
