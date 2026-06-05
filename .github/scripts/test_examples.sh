#!/usr/bin/env bash
set -euo pipefail

arch="${1:?usage: .github/scripts/test_examples.sh <arch>}"

export PYTHONPATH="$PWD/build/python_packages:${PYTHONPATH:-}"

if [ "$arch" = "riscv64" ]; then
  # The installed torchvision on riscv64 is outdated and causes compatibility
  # issues with other modules.
  pip uninstall torchvision -y

  cmake -S . -B build \
    -DBUDDY_QWEN3_EXAMPLES=ON \
    -DBUDDY_GEMMA4_EXAMPLES=ON \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
    -DBUDDY_ENABLE_PNG=ON

  ccache -z
  ninja -C build -j1 \
    buddy-qwen3-0.6b-run \
    buddy-gemma4-e2b-run \
    buddy-deepseek-r1-cli
  ccache -s
else
  cmake -S . -B build \
    -DBUDDY_BERT_EXAMPLES=ON \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
    -DBUDDY_GEMMA4_EXAMPLES=ON \
    -DBUDDY_LENET_EXAMPLES=ON \
    -DBUDDY_MOBILENETV3_EXAMPLES=ON \
    -DBUDDY_QWEN3_EXAMPLES=ON \
    -DBUDDY_RESNET_EXAMPLES=ON \
    -DBUDDY_TRANSFORMER_EXAMPLES=ON \
    -DBUDDY_ENABLE_PNG=ON

  ccache -z
  ninja -C build -j1 \
    buddy-deepseek-r1-cli \
    buddy-qwen3-0.6b-run \
    buddy-gemma4-e2b-run \
    transformer-runner \
    buddy-bert-run \
    buddy-lenet-run \
    buddy-mobilenetv3-run \
    buddy-resnet-run
  ccache -s
fi

ctest --test-dir "$PWD/build" \
  -L example \
  --output-on-failure \
  --no-tests=error
