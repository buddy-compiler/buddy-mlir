#!/usr/bin/env bash
# The default single_forward pipeline, with canonicalization before OpenMP.
# Remove one-trip parallel loops and combine directly nested parallel loops
# so a unit batch dimension does not hide the useful work from OpenMP.
# Keep thread selection at runtime, as in the default single_forward pipeline.
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
set -euo pipefail
BUDDY_OPT="$1"; LLVM_BIN="$2"; IN="$3"; OUT="$4"
shift 4
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")

TOSA="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"

"$LLVM_BIN/mlir-opt" "$IN" -pass-pipeline "$TOSA" \
  | "$LLVM_BIN/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
  | "$BUDDY_OPT" \
      -arith-expand -eliminate-empty-tensors -convert-elementwise-to-linalg \
      -empty-tensor-to-alloc-tensor \
      -one-shot-bufferize=bufferize-function-boundaries \
      -ownership-based-buffer-deallocation \
      -buffer-deallocation-simplification -bufferization-lower-deallocations \
      -matmul-parallel-vectorization-optimize \
      -convert-linalg-to-affine-loops -affine-loop-fusion \
      -affine-parallelize -lower-affine -canonicalize \
      -convert-scf-to-openmp -convert-linalg-to-loops -convert-vector-to-scf \
      -expand-strided-metadata -lower-affine -cse -convert-vector-to-llvm \
      -memref-expand -convert-arith-to-llvm -finalize-memref-to-llvm \
      -convert-scf-to-cf -convert-cf-to-llvm -llvm-request-c-wrappers \
      -convert-openmp-to-llvm -convert-arith-to-llvm -convert-math-to-llvm \
      -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts \
  | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir \
  | "$LLVM_BIN/llvm-as" \
  | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" \
      -filetype=obj -relocation-model=pic -O3 -o "$OUT"
