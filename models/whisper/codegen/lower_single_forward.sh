#!/usr/bin/env bash
# Compile one Whisper single-forward compute graph with checked pipeline exits.
set -euo pipefail

buddy_opt="$1"
llvm_bin="$2"
input="$3"
output="$4"
use_ime="$5"
shift 5
llc_attrs=("$@")

ime_opts=()
if [[ "$use_ime" == ON ]]; then
  ime_pass='-lower-linalg-to-ime=target=k3 a100-fp16=true a100-gemm=true verify-fp16=true'
  if [[ -n "${BUDDY_SINGLE_FORWARD_IME_SHAPES:-}" ]]; then
    ime_pass+=" matmul-shapes=${BUDDY_SINGLE_FORWARD_IME_SHAPES}"
  fi
  ime_opts=(
    -staticize-memref-layout
    "$ime_pass"
    -lower-ime=target=k3
    -expand-strided-metadata
  )
fi

batch_opts=(-batchmatmul-optimize)
if [[ "${BUDDY_SINGLE_FORWARD_SKIP_BATCH:-OFF}" == ON ]]; then
  batch_opts=()
fi

"$llvm_bin/mlir-opt" "$input" \
  -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))' \
  | "$llvm_bin/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
  | "$buddy_opt" \
      -arith-expand \
      -eliminate-empty-tensors \
      -convert-elementwise-to-linalg \
      -empty-tensor-to-alloc-tensor \
      -one-shot-bufferize=bufferize-function-boundaries \
      -ownership-based-buffer-deallocation \
      -buffer-deallocation-simplification \
      -bufferization-lower-deallocations \
      "${ime_opts[@]}" \
      -matmul-parallel-vectorization-optimize \
      "${batch_opts[@]}" \
      -convert-linalg-to-affine-loops \
      -affine-loop-fusion \
      -affine-parallelize \
      -lower-affine \
      -convert-scf-to-openmp \
      -convert-linalg-to-loops \
      -convert-vector-to-scf \
      -expand-strided-metadata \
      -lower-affine \
      -cse \
      -convert-vector-to-llvm \
      -memref-expand \
      -convert-arith-to-llvm \
      -finalize-memref-to-llvm \
      -convert-scf-to-cf \
      -convert-cf-to-llvm \
      -llvm-request-c-wrappers \
      -convert-openmp-to-llvm \
      -convert-arith-to-llvm \
      -convert-math-to-llvm \
      -convert-math-to-libm \
      -convert-func-to-llvm \
      -reconcile-unrealized-casts \
  | "$llvm_bin/mlir-translate" -mlir-to-llvmir \
  | "$llvm_bin/llvm-as" \
  | "$llvm_bin/llc" -filetype=obj -relocation-model=pic \
      "${llc_attrs[@]}" -O3 -o "$output"
