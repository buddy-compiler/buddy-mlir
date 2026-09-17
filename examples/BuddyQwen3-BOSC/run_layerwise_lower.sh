#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${1:-$SCRIPT_DIR/build-model-w8a8-layerwise}
VECTOR_WIDTH=${VECTOR_WIDTH:-16}
OPTIMIZE_ATTENTION_BMM=${OPTIMIZE_ATTENTION_BMM:-0}
W8A8_PROFILE_PHASES=${W8A8_PROFILE_PHASES:-0}
W8A8_SILU_MUL_QUANTIZE_FUSION=${W8A8_SILU_MUL_QUANTIZE_FUSION:-auto}
W8A8_GATE_UP_LINEAR_FUSION=${W8A8_GATE_UP_LINEAR_FUSION:-auto}

for component in \
  embedding_prefill \
  embedding_decode \
  decoder_layer_prefill \
  decoder_layer_decode \
  final_head; do
  module="subgraph0_$component"
  extra_args=()
  if [[ "$OPTIMIZE_ATTENTION_BMM" == "1" && \
        "$component" == "decoder_layer_prefill" ]]; then
    extra_args+=(--optimize-attention-bmm)
  fi
  if [[ "$W8A8_PROFILE_PHASES" == "1" ]]; then
    extra_args+=(--w8a8-profile-phases)
  fi
  if [[ "$W8A8_SILU_MUL_QUANTIZE_FUSION" == "1" ]]; then
    extra_args+=(--w8a8-enable-silu-mul-quantize-fusion)
  elif [[ "$W8A8_SILU_MUL_QUANTIZE_FUSION" == "0" ]]; then
    extra_args+=(--w8a8-disable-silu-mul-quantize-fusion)
  fi
  if [[ "$W8A8_GATE_UP_LINEAR_FUSION" == "1" ]]; then
    extra_args+=(--w8a8-enable-gate-up-linear-fusion)
  elif [[ "$W8A8_GATE_UP_LINEAR_FUSION" == "0" ]]; then
    extra_args+=(--w8a8-disable-gate-up-linear-fusion)
  fi
  "$SCRIPT_DIR/run_lower.sh" "$BUILD_DIR/$module.mlir" \
    --output-dir "$BUILD_DIR/lowering-$component" \
    --name "$module" --mode vir --vector-width "$VECTOR_WIDTH" \
    --emit-c-wrapper "${extra_args[@]}"
done

echo "Qwen3 layer-wise MLIR lowered to LLVM IR in: $BUILD_DIR"
