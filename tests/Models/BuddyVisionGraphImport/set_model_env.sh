#!/usr/bin/env bash
# ===- set_model_env.sh ------------------------------------------------------
#
# Helper script to set environment variables for vision graph coverage tests.
# Each model can be pointed to a local path for offline testing.
#
# Usage:
#   1. Edit the paths below to point to your local model directories
#   2. Source this script: source set_model_env.sh
#   3. Run tests: python tests/Models/BuddyVisionGraphImport/test_import_vit.py
#
# ===---------------------------------------------------------------------------

# --- Edit these paths to your local model directories ---

export CLIP_MODEL_PATH="$HOME/model/clip-vit-base-patch32"
export DEIT_MODEL_PATH="$HOME/model/deit-base-distilled-patch16-224"
export DETR_MODEL_PATH="$HOME/model/detr-resnet-50"
export MOBILEVIT_MODEL_PATH="$HOME/model/mobilevit-small"
export REGNET_MODEL_PATH="$HOME/model/regnet-y-040"
export RESNET_MODEL_PATH="$HOME/model/resnet-50"
export SEGFORMER_MODEL_PATH="$HOME/model/segformer-b0-finetuned-ade-512-512"
export SWIN_MODEL_PATH="$HOME/model/swin-tiny-patch4-window7-224"
export VIT_MODEL_PATH="$HOME/model/vit-base-patch16-224"

# --- Print configured paths ---

echo "Vision Graph Coverage Test - Model Paths:"
echo "  CLIP_MODEL_PATH       = $CLIP_MODEL_PATH"
echo "  DEIT_MODEL_PATH       = $DEIT_MODEL_PATH"
echo "  DETR_MODEL_PATH       = $DETR_MODEL_PATH"
echo "  MOBILEVIT_MODEL_PATH  = $MOBILEVIT_MODEL_PATH"
echo "  REGNET_MODEL_PATH     = $REGNET_MODEL_PATH"
echo "  RESNET_MODEL_PATH     = $RESNET_MODEL_PATH"
echo "  SEGFORMER_MODEL_PATH  = $SEGFORMER_MODEL_PATH"
echo "  SWIN_MODEL_PATH       = $SWIN_MODEL_PATH"
echo "  VIT_MODEL_PATH        = $VIT_MODEL_PATH"
