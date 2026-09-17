#!/usr/bin/env bash
set -euo pipefail
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
STAGE="$MODEL/build/review-native-28l"
VARIANT="$MODEL/build/ame-v05/model-28l-phase-probe"
PYTHON=/home/chh/miniconda3/envs/boscame/bin/python
"$PYTHON" -B "$MODEL/tools/build_nr_w8a8_image.py" \
 --repo-root "$PWD" --linker /usr/bin/ld.lld-20 \
 --report "$STAGE/replacement/triton-call-replacement.json" \
 --segment "$MODEL/build/board/w8a8-28l/w8a8-segment.json" \
 --graph-ir "$STAGE/nr-prefill/forward_prefill.ll" \
 --decode-ir "$STAGE/nr-decode/forward_decode.ll" \
 --archive "$VARIANT/model-lib/libqwen_triton.a" \
 --adapters "$STAGE/replacement/qwen_triton_adapters.c" \
 --output "$VARIANT/image" --layers 28 --cache-len 512 --prefill-len 16 --decode-steps 8 \
 --prompt-text 'What is France?' --tokenizer-blob "$MODEL/build/tokenizer.bin" \
 --reference-arrays "$MODEL/build/review-28l/quant-nr-order/arrays.npz" \
 --reference-metadata "$MODEL/build/review-28l/quant-nr-order/quant-reference.json" \
 --profile-kernels --profile-progress --profile-probe _mlir_ciface_kernel_dequantize_1x1024:12
"$PYTHON" -B "$MODEL/tools/prepare_model_run.py" \
 --image "$VARIANT/image/qwen_model.bin" --elf "$VARIANT/image/qwen_model.elf" \
 --weights "$MODEL/build/board/w8a8-28l/weights-w8a8.bin" \
 --weight-manifest "$MODEL/build/board/w8a8-28l/w8a8-segment.json" \
 --tokenizer "$MODEL/build/tokenizer.bin" --output "$VARIANT/prepared"
