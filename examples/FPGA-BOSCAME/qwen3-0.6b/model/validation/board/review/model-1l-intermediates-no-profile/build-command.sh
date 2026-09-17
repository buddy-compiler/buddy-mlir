#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root using a Python environment containing numpy.
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
STAGE="$MODEL/build/review-native-1l"
python3 -B "$MODEL/tools/build_nr_w8a8_image.py" \
  --repo-root "$PWD" \
  --report "$STAGE/replacement/triton-call-replacement.json" \
  --segment "$MODEL/build/board/w8a8-1l-run/w8a8-segment.json" \
  --graph-ir "$STAGE/nr-prefill/forward_prefill.ll" \
  --decode-ir "$STAGE/nr-decode/forward_decode.ll" \
  --archive "$STAGE/model-lib/libqwen_triton.a" \
  --adapters "$STAGE/replacement/qwen_triton_adapters.c" \
  --output "$STAGE/image-intermediates-no-profile" \
  --layers 1 --cache-len 512 --prefill-len 16 --decode-steps 8 \
  --prompt-text 'What is France?' --tokenizer-blob "$MODEL/build/tokenizer.bin" \
  --reference-arrays "$STAGE/quant-nr-trace/arrays.npz" \
  --reference-metadata "$STAGE/quant-nr-trace/quant-reference.json" \
  --intermediate-arrays "$STAGE/quant-nr-trace/arrays.npz" \
  --intermediate-layout "$MODEL/build/import/probe-1layer/weight-layout.json" \
  --intermediate-graph-dir "$STAGE/replacement" \
  --intermediate-progress
python3 -B "$MODEL/tools/prepare_model_run.py" \
  --image "$STAGE/image-intermediates-no-profile/qwen_model.bin" \
  --elf "$STAGE/image-intermediates-no-profile/qwen_model.elf" \
  --weights "$MODEL/build/board/w8a8-1l-run/weights-w8a8.bin" \
  --weight-manifest "$MODEL/build/board/w8a8-1l-run/w8a8-segment.json" \
  --tokenizer "$MODEL/build/tokenizer.bin" \
  --output "$STAGE/prepared-intermediates-no-profile"
