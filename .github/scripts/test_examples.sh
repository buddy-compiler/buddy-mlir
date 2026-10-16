#!/usr/bin/env bash
set -euo pipefail

arch="${1:?usage: .github/scripts/test_examples.sh <arch>}"

export PYTHONPATH="$PWD/build/python_packages:${PYTHONPATH:-}"

# Gemma4 is skipped for now: import-gemma4.py patches the decode MLIR with a
# pattern the current lowering no longer emits, and the patch script exits
# non-zero when it applies nothing.

if [ "$arch" = "riscv64" ]; then
  cmake -S . -B build \
    -DBUDDY_QWEN3_EXAMPLES=ON \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
    -DBUDDY_ENABLE_PNG=ON \
    -DDEEPSEEKR1_TEST_VARIANT=w4a16

  ccache -z
  ninja -C build \
    buddy-qwen3-0.6b-run \
    buddy-deepseek-r1-w4a16-run
  ccache -s
else
  # The YOLO26 importer needs ultralytics, which depends on torchvision;
  # riscv64 has no torchvision wheel, so YOLO26 is x86/arm only.
  uv pip install ultralytics

  cmake -S . -B build \
    -DBUDDY_BERT_EXAMPLES=ON \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
    -DBUDDY_LENET_EXAMPLES=ON \
    -DBUDDY_MOBILENETV3_EXAMPLES=ON \
    -DBUDDY_QWEN3_EXAMPLES=ON \
    -DBUDDY_RESNET_EXAMPLES=ON \
    -DBUDDY_STABLE_DIFFUSION_EXAMPLES=ON \
    -DBUDDY_TRANSFORMER_EXAMPLES=ON \
    -DBUDDY_YOLO26_EXAMPLES=ON \
    -DBUDDY_ENABLE_PNG=ON \
    -DDEEPSEEKR1_TEST_VARIANT=w4a16

  ccache -z
  ninja -C build \
    buddy-deepseek-r1-w4a16-run \
    buddy-qwen3-0.6b-run \
    transformer-runner \
    buddy-bert-run \
    buddy-lenet-run \
    buddy-stable-diffusion-run \
    buddy-mobilenetv3-run \
    buddy-resnet-run \
    buddy-yolo26n-run
  ccache -s
fi

ctest --test-dir "$PWD/build" \
  -L example \
  --output-on-failure \
  --no-tests=error
