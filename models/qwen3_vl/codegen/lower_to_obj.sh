#!/usr/bin/env bash
# Lower one buddy/TOSA MLIR module to a relocatable object via the standard
# buddy compile pipeline. Invoked by models/qwen3_vl/CMakeLists.txt (a piped
# pipeline like this cannot be expressed directly in add_custom_command).
#
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
set -euo pipefail
BUDDY_OPT="$1"; LLVM_BIN="$2"; IN="$3"; OUT="$4"
THREADS="${5:-1}"
shift 4 || true
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")

TOSA="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))"

"$BUDDY_OPT" "$IN" -simplify-tosa-reshape \
  | "$LLVM_BIN/mlir-opt" -pass-pipeline "$TOSA" \
  | "$BUDDY_OPT" \
      -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
      -convert-elementwise-to-linalg \
      -one-shot-bufferize=bufferize-function-boundaries \
      -expand-strided-metadata -ownership-based-buffer-deallocation -canonicalize \
      -buffer-deallocation-simplification -bufferization-lower-deallocations -cse \
      -canonicalize -optimize-allocation-liveness \
      -matmul-vectorization-blis -batchmatmul-optimize \
      -convert-linalg-to-affine-loops -affine-parallelize -convert-vector-to-scf \
      -lower-affine -convert-scf-to-openmp=num-threads="${THREADS}" -cse -memref-expand \
      -arith-expand -convert-vector-to-llvm -convert-arith-to-llvm \
      -finalize-memref-to-llvm -convert-scf-to-cf -convert-cf-to-llvm \
      -llvm-request-c-wrappers -convert-openmp-to-llvm -convert-arith-to-llvm \
      -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm \
      -reconcile-unrealized-casts \
  | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir \
  | "$LLVM_BIN/llvm-as" \
  | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" -filetype=obj -relocation-model=pic -O3 -o "$OUT"
