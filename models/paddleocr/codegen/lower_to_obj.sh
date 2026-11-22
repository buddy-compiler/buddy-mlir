#!/usr/bin/env bash
# Lower PaddleOCR-VL subgraph0.mlir to a relocatable object file.
# Invoked by buddy_add_model (LOWER_SCRIPT) from models/paddleocr/CMakeLists.txt.
#
# When BUDDY_PADDLEOCR_IME=ON (or a .paddleocr_ime_enabled marker exists next to
# the output object), lowers every linalg.matmul in the FP16 graph to SpacemiT
# K3 A100 FP16 IME (FP32 accumulation), parallelizes attention batch_matmul ops,
# skips unsafe affine-loop-fusion, and targets -mcpu=spacemit-a100 (+xsmtvdotii).
#
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
set -euo pipefail

BUDDY_OPT="$1"
LLVM_BIN="$2"
IN="$3"
OUT="$4"
shift 4 || true

THREADS=1
if [[ $# -gt 0 ]]; then
  THREADS="$1"
  shift
fi
LLC_ATTRS=("$@")

USE_IME="${BUDDY_PADDLEOCR_IME:-OFF}"
if [[ "$USE_IME" != ON && -f "$(dirname "$OUT")/.paddleocr_ime_enabled" ]]; then
  USE_IME=ON
fi

TOSA_PIPELINE="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"

if [[ "$USE_IME" == ON ]]; then
  has_flag() {
    local needle="$1"
    for arg in "${LLC_ATTRS[@]}"; do
      if [[ "$arg" == *"$needle"* ]]; then
        return 0
      fi
    done
    return 1
  }

  A100_LLC_ATTRS=(-mcpu=spacemit-a100 -target-abi=lp64d)
  if ! has_flag "+xsmtvdotii"; then
    A100_LLC_ATTRS+=(-mattr=+v,+zfh,+zvfh,+zvl1024b,+xsmtvdotii)
  fi
  if ! has_flag "-riscv-v-vector-bits-min="; then
    A100_LLC_ATTRS+=(-riscv-v-vector-bits-min=1024)
  fi
  if ! has_flag "-riscv-v-vector-bits-max="; then
    A100_LLC_ATTRS+=(-riscv-v-vector-bits-max=1024)
  fi

  # Decode graphs keep small M=1 softmax reductions where affine-parallelize
  # can race on reduction accumulators, and pass per-layer KV-cache subviews
  # whose strided offsets must not be dropped by staticize-memref-layout;
  # heavy M=1 GEMVs are already parallelized by lower-linalg-to-ime.
  AFFINE_PASSES=(-convert-linalg-to-affine-loops -affine-parallelize -lower-affine)
  STATICIZE_PASSES=(-staticize-memref-layout)
  if [[ "${BUDDY_PADDLEOCR_NO_AFFINE:-OFF}" == ON ]] || grep -q '@subgraph_decode' "$IN"; then
    AFFINE_PASSES=(-convert-linalg-to-loops)
    STATICIZE_PASSES=()
  fi

  "$BUDDY_OPT" "$IN" -simplify-tosa-reshape -silu-fusion \
    | "$LLVM_BIN/mlir-opt" -pass-pipeline "$TOSA_PIPELINE" \
    | "$LLVM_BIN/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
    | "$BUDDY_OPT" \
        -fuse-prefill-elementwise \
        -arith-expand \
        -eliminate-empty-tensors \
        -convert-elementwise-to-linalg \
        -fuse-prefill-elementwise \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize=bufferize-function-boundaries \
        -expand-strided-metadata \
        -ownership-based-buffer-deallocation \
        -buffer-deallocation-simplification \
        -bufferization-lower-deallocations \
        "${STATICIZE_PASSES[@]}" \
        '-lower-linalg-to-ime=target=k3 a100-fp16=true a100-gemm=true verify-fp16=true' \
        -lower-ime \
        -expand-strided-metadata \
        -matmul-vectorization-decode=vector-size=64 \
        '-matmul-vectorization-blis=vector-size=64 target-vlen-bits=1024' \
        -batchmatmul-optimize=vector-size=64 \
        '-batchmatmul-transpose-b-vectorization=vector-size=64 parallel=true' \
        -lower-linalg-to-vir=exp-only=true \
        -lower-vir-to-vector=vector-width=32 \
        -approximate-vector-exp \
        "${AFFINE_PASSES[@]}" \
        "-convert-scf-to-openmp=num-threads=${THREADS}" \
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
    | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir \
    | "$LLVM_BIN/llvm-as" \
    | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" "${A100_LLC_ATTRS[@]}" \
        -filetype=obj -relocation-model=pic -O3 -o "$OUT"
else
  "$LLVM_BIN/mlir-opt" "$IN" -pass-pipeline "$TOSA_PIPELINE" \
    | "$LLVM_BIN/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
    | "$BUDDY_OPT" \
        -arith-expand \
        -eliminate-empty-tensors \
        -convert-elementwise-to-linalg \
        -empty-tensor-to-alloc-tensor \
        -one-shot-bufferize=bufferize-function-boundaries \
        -ownership-based-buffer-deallocation \
        -buffer-deallocation-simplification \
        -bufferization-lower-deallocations \
        -matmul-parallel-vectorization-optimize \
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
    | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir \
    | "$LLVM_BIN/llvm-as" \
    | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" \
        -filetype=obj -relocation-model=pic -O3 -o "$OUT"
fi
