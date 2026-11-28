#!/usr/bin/env bash
# 构建机：从空目录交叉构建并打包；板端执行入口为 verify_board.py。
set -euo pipefail
: "${ARTIFACT_DIR:?Set a new empty ARTIFACT_DIR}"
: "${HOST_PYTHON:?Set HOST_PYTHON}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
buddy_build=${BUDDY_BUILD_DIR:-$repo/build}
llvm_build=${LLVM_BUILD_DIR:-$repo/llvm/build}
if [[ -e "$ARTIFACT_DIR" ]]; then
  shopt -s nullglob dotglob
  contents=("$ARTIFACT_DIR"/*)
  if (( ${#contents[@]} )); then
    echo 'ARTIFACT_DIR must be empty; existing outputs will not be reused' >&2
    exit 1
  fi
fi
bash "$repo/examples/TritonSoftmax/run.sh" all
"$HOST_PYTHON" "$repo/examples/TritonSoftmax/build_contract_tests.py" \
  --build-dir "$buddy_build" --artifact-dir "$ARTIFACT_DIR"
"$HOST_PYTHON" "$repo/examples/TritonSoftmax/package_tests.py" \
  --build-dir "$buddy_build" --artifact-dir "$ARTIFACT_DIR" --llvm-bin "$llvm_build/bin"
