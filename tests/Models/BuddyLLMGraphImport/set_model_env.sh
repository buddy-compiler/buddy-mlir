#!/usr/bin/env bash
# ===- set_model_env.sh ------------------------------------------------------
#
# Helper script to set environment variables for LLM graph coverage tests.
# Each model can be pointed to a local path for offline testing.
#
# Usage:
#   1. Edit the paths below to point to your local model directories
#   2. Source this script: source set_model_env.sh
#   3. Run tests: python tests/Models/BuddyLLMGraphImport/test_import_mistral_7b.py
#
# ===---------------------------------------------------------------------------

# --- Edit these paths to your local model directories ---

export MISTRAL_7B_MODEL_PATH="$HOME/model/Mistral-7B-Instruct-v0.3"
export QWEN2_5_7B_MODEL_PATH="$HOME/model/Qwen2.5-7B-Instruct"
export GEMMA2_9B_MODEL_PATH="$HOME/model/Gemma-2-9B-It"
export GEMMA4_E2B_MODEL_PATH="$HOME/model/Gemma-4-E2B-it"
export DEEPSEEK_R1_LLAMA_8B_MODEL_PATH="$HOME/model/DeepSeek-R1-Distill-Llama-8B"
export PHI3_MINI_4K_MODEL_PATH="$HOME/model/Phi-3-mini-4k-instruct"
export LLAMA3_1_8B_MODEL_PATH="$HOME/model/Llama-3.1-8B-Instruct"
export TINYLLAMA_1_1B_MODEL_PATH="$HOME/model/TinyLlama-1.1B-Chat-v1.0"
export CHATGLM3_6B_MODEL_PATH="$HOME/model/chatglm3-6b"
export SOLAR_10_7B_MODEL_PATH="$HOME/model/SOLAR-10.7B-Instruct-v1.0"
export YI_CODER_9B_MODEL_PATH="$HOME/model/Yi-Coder-9B-Chat"
export ALBERT_MODEL_PATH="$HOME/model/albert-base-v2"
export BART_MODEL_PATH="$HOME/model/bart-base"
export DISTILBERT_MODEL_PATH="$HOME/model/distilbert-base-uncased"
export ELECTRA_MODEL_PATH="$HOME/model/electra-small-discriminator"
export GPT2_MODEL_PATH="$HOME/model/gpt2"
export MOBILEBERT_MODEL_PATH="$HOME/model/mobilebert-uncased"
export OPT_MODEL_PATH="$HOME/model/opt-125m"
export PEGASUS_MODEL_PATH="$HOME/model/pegasus-xsum"
export PYTHIA_MODEL_PATH="$HOME/model/pythia-70m"
export ROBERTA_MODEL_PATH="$HOME/model/roberta-base"
export XLM_ROBERTA_MODEL_PATH="$HOME/model/xlm-roberta-base"

# --- Print configured paths ---

echo "LLM Graph Coverage Test - Model Paths:"
echo "  MISTRAL_7B_MODEL_PATH            = $MISTRAL_7B_MODEL_PATH"
echo "  QWEN2_5_7B_MODEL_PATH            = $QWEN2_5_7B_MODEL_PATH"
echo "  GEMMA2_9B_MODEL_PATH             = $GEMMA2_9B_MODEL_PATH"
echo "  GEMMA4_E2B_MODEL_PATH            = $GEMMA4_E2B_MODEL_PATH"
echo "  DEEPSEEK_R1_LLAMA_8B_MODEL_PATH  = $DEEPSEEK_R1_LLAMA_8B_MODEL_PATH"
echo "  PHI3_MINI_4K_MODEL_PATH          = $PHI3_MINI_4K_MODEL_PATH"
echo "  LLAMA3_1_8B_MODEL_PATH           = $LLAMA3_1_8B_MODEL_PATH"
echo "  TINYLLAMA_1_1B_MODEL_PATH        = $TINYLLAMA_1_1B_MODEL_PATH"
echo "  CHATGLM3_6B_MODEL_PATH           = $CHATGLM3_6B_MODEL_PATH"
echo "  SOLAR_10_7B_MODEL_PATH           = $SOLAR_10_7B_MODEL_PATH"
echo "  YI_CODER_9B_MODEL_PATH           = $YI_CODER_9B_MODEL_PATH"
echo "  ALBERT_MODEL_PATH                = $ALBERT_MODEL_PATH"
echo "  BART_MODEL_PATH                  = $BART_MODEL_PATH"
echo "  DISTILBERT_MODEL_PATH            = $DISTILBERT_MODEL_PATH"
echo "  ELECTRA_MODEL_PATH               = $ELECTRA_MODEL_PATH"
echo "  GPT2_MODEL_PATH                  = $GPT2_MODEL_PATH"
echo "  MOBILEBERT_MODEL_PATH            = $MOBILEBERT_MODEL_PATH"
echo "  OPT_MODEL_PATH                   = $OPT_MODEL_PATH"
echo "  PEGASUS_MODEL_PATH               = $PEGASUS_MODEL_PATH"
echo "  PYTHIA_MODEL_PATH                = $PYTHIA_MODEL_PATH"
echo "  ROBERTA_MODEL_PATH               = $ROBERTA_MODEL_PATH"
echo "  XLM_ROBERTA_MODEL_PATH           = $XLM_ROBERTA_MODEL_PATH"
