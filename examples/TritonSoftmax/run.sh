#!/usr/bin/env bash
# 构建机专用：交叉编译，不执行任何目标程序；板端执行需单独批准。
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
buddy_build=${BUDDY_BUILD_DIR:-$repo/build}
llvm_build=${LLVM_BUILD_DIR:-$repo/llvm/build}
artifacts=${ARTIFACT_DIR:-$buddy_build/triton-softmax-riscv}
stage=${1:-all}
case "$stage" in
  export|standalone|buddy|integration|rax|all) ;;
  *) echo 'usage: run.sh [export|standalone|buddy|integration|rax|all] (cross build only)' >&2; exit 1 ;;
esac
: "${CROSS_CXX:?Set CROSS_CXX to the riscv64 Linux cross C++ compiler or wrapper}"
if [[ $("$CROSS_CXX" -dumpmachine) != riscv64*linux* ]]; then
  echo 'CROSS_CXX must target riscv64 Linux' >&2
  exit 1
fi
mkdir -p "$artifacts"
artifacts=$(cd "$artifacts" && pwd)
source_dir=$repo/examples/TritonSoftmax
# Query/Adapter/Runtime code must use the baseline ISA, not the kernel ISA.
common=(-std=c++17 -O2 -march=rv64gc -mabi=lp64d -fPIC -pthread
  -I"$repo/runtime/include" -I"$artifacts")
if [[ -n ${TARGET_SYSROOT:-} ]]; then
  common+=(--sysroot="$TARGET_SYSROOT")
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRITON_RISCV_OPENMP_THREADS=1 TRITON_RISCV_CROSS_COMPILE=1
export TRITON_CACHE_DIR="$artifacts/triton-cache"
export TRITON_DUMP_DIR="$artifacts/dump" TRITON_OVERRIDE_DIR="$artifacts/override"
export TRITON_SHARED_DUMP_PATH="$artifacts/ir"
export BUDDY_MLIR_BINARY_DIR="$buddy_build/bin" LLVM_BINARY_DIR="$llvm_build/bin"
unset TRITON_RISCV_USE_IME TRITON_SHARED_SANITIZER_TYPE

if [[ $stage == all || $stage == export ]]; then
  : "${TRITON_REPO:?Set TRITON_REPO}"
  : "${TRITON_PYTHON:?Set TRITON_PYTHON}"
  "$TRITON_PYTHON" "$source_dir/export.py" --triton-repo "$TRITON_REPO" \
    --output "$artifacts" | tee "$artifacts/export.log"
fi
if [[ $stage == all || $stage == standalone ]]; then
  "$CROSS_CXX" "${common[@]}" -shared "$source_dir/Adapter.cpp" \
    "$artifacts/softmax.obj" -Wl,-z,defs \
    "-Wl,--version-script=$source_dir/exports.map" -lm \
    -o "$artifacts/libtriton_softmax.so"
  "$CROSS_CXX" "${common[@]}" "$source_dir/Standalone.cpp" -L"$artifacts" \
    -ltriton_softmax '-Wl,-rpath,$ORIGIN' -o "$artifacts/standalone"
fi
if [[ $stage == all || $stage == buddy || $stage == integration ]]; then
  if [[ $stage != integration ]]; then
    : "${BUDDY_PYTHON:?Set BUDDY_PYTHON}"
    export PYTHONPATH="$buddy_build/python_packages:$llvm_build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
    "$BUDDY_PYTHON" "$source_dir/import_buddy.py" --output "$artifacts/buddy.mlir" \
      --buddy-opt "${LOWERING_OPT:-$buddy_build/bin/buddy-opt}" --llvm-bin "$llvm_build/bin" \
      | tee "$artifacts/import.log"
  fi
  "$CROSS_CXX" "${common[@]}" "$source_dir/Run.cpp" \
    "$source_dir/SoftmaxSession.cpp" "$repo/runtime/core/ExternalKernel.cpp" \
    "$artifacts/buddy.o" "$artifacts/computation.o" -ldl -lm -o "$artifacts/run"
fi
if [[ $stage == all || $stage == rax ]]; then
  : "${FLATBUFFERS_INCLUDE_DIR:?Set FLATBUFFERS_INCLUDE_DIR to the directory containing flatbuffers/}"
  : "${HOST_PYTHON:?Set HOST_PYTHON to a build-machine Python executable}"
  cli_include=${CLI11_INCLUDE_DIR:-$buddy_build/_deps/cli11_hpp-src}
  # Only stage portable headers. Never add the build host's /usr/include to a
  # cross compile's search path, which could select x86 libc headers.
  mkdir -p "$artifacts/include"
  cp -R "$FLATBUFFERS_INCLUDE_DIR/flatbuffers" "$artifacts/include/"
  cp "$cli_include/CLI11.hpp" "$artifacts/include/"
  common+=(-I"$artifacts/include" -I"$buddy_build/runtime/include")
  "$CROSS_CXX" "${common[@]}" -shared "$source_dir/SoftmaxRunner.cpp" \
    "$source_dir/SoftmaxSession.cpp" "$repo/runtime/core/ExternalKernel.cpp" \
    "$repo/runtime/core/RaxDependenciesHandle.cpp" "$artifacts/buddy.o" \
    "$artifacts/computation.o" -Wl,-z,defs \
    "-Wl,--version-script=$source_dir/runner-exports.map" -ldl -lm \
    -o "$artifacts/softmax_runner.so"
  # Compile the actual buddy-cli source without built-in model/NUMA adapters.
  "$CROSS_CXX" "${common[@]}" '-DBUDDY_VERSION="issue918-poc"' \
    "$repo/tools/buddy-cli/buddy-cli.cpp" "$repo/runtime/core/RaxDependenciesHandle.cpp" \
    -ldl -lm -o "$artifacts/buddy-cli"
  "$HOST_PYTHON" "$source_dir/run_rax.py" prepare --build-dir "$buddy_build" \
    --artifact-dir "$artifacts"
fi
for file in softmax.obj libtriton_softmax.so standalone buddy.o computation.o run softmax_runner.so buddy-cli; do
  if [[ -f $artifacts/$file ]]; then
    "$llvm_build/bin/llvm-readobj" --file-headers "$artifacts/$file" > "$artifacts/$file.headers.txt"
    "$llvm_build/bin/llvm-nm" "$artifacts/$file" > "$artifacts/$file.symbols.txt"
  fi
done
if [[ -f $artifacts/libtriton_softmax.so ]]; then
  "$llvm_build/bin/llvm-readobj" --needed-libs "$artifacts/libtriton_softmax.so" > "$artifacts/dependencies.txt"
  "$llvm_build/bin/llvm-nm" -D --defined-only -j "$artifacts/libtriton_softmax.so" > "$artifacts/exports.txt"
fi
printf '%s\n' 'Cross build complete. No target executable has been run.'
