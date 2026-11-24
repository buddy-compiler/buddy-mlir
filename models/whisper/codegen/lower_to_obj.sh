#!/usr/bin/env bash
# Lower one Whisper MLIR module to a relocatable object.
#
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
#
# By default, runs the standard BLIS/batch-matmul pipeline used by
# TEMPLATE_PARTITION_CAPABLE. When BUDDY_WHISPER_PIPELINE=single_forward is
# set (used by K3 split/KV builds), runs the checked Whisper single-forward
# pipeline with opt-in A100 FP16 IME, fused softmax/GELU, and decode passes.
set -euo pipefail

BUDDY_OPT="$1"
LLVM_BIN="$2"
IN="$3"
OUT="$4"
THREADS="${5:-1}"
shift 4 || true
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")

if [[ "${BUDDY_WHISPER_PIPELINE:-default}" != "single_forward" ]]; then
  TOSA="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))"

  "$BUDDY_OPT" "$IN" -simplify-tosa-reshape \
    | "$LLVM_BIN/mlir-opt" -pass-pipeline "$TOSA" \
    | "$BUDDY_OPT" \
        -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
        -convert-elementwise-to-linalg \
        -one-shot-bufferize=bufferize-function-boundaries \
        -expand-strided-metadata -ownership-based-buffer-deallocation -canonicalize \
        -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-bufferization-to-memref -cse \
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
  exit 0
fi

use_ime="${BUDDY_SINGLE_FORWARD_IME:-OFF}"

a100_llc_attrs=()
if [[ "$use_ime" == ON || "${BUDDY_SINGLE_FORWARD_DECODE_TB:-OFF}" == ON ]]; then
  codegen_dir="$(dirname "$0")/../../../tools/buddy-codegen"
  a100_out="$("${PYTHON:-python3}" -c '
import sys
sys.path.insert(0, sys.argv[1])
from compile_pipeline import a100_llc_args
try:
    print("\n".join(a100_llc_args(sys.argv[2:])))
except ValueError as e:
    sys.exit(f"lower_to_obj.sh: {e}")
' "$codegen_dir" "${LLC_ATTRS[@]}")"
  mapfile -t a100_llc_attrs <<< "$a100_out"
fi

pre_buf_opts=()
if [[ "${BUDDY_SINGLE_FORWARD_FUSE_TB:-OFF}" == ON ]]; then
  pre_buf_opts+=(-fuse-attention-softmax)
fi
if [[ "${BUDDY_SINGLE_FORWARD_ELIM_ZERO:-OFF}" == ON ]]; then
  pre_buf_opts+=(-eliminate-large-zero-constants)
fi
if [[ "${BUDDY_SINGLE_FORWARD_FUSE_TB:-OFF}" == ON ]]; then
  pre_buf_opts+=(-fuse-matmul-transpose-b)
fi

post_buf_opts=()
if [[ "${BUDDY_SINGLE_FORWARD_ELIM_COPY:-OFF}" == ON ]]; then
  post_buf_opts+=(-eliminate-memref-copy)
fi

ime_opts=()
if [[ "$use_ime" == ON ]]; then
  ime_pass='-lower-linalg-to-ime=target=k3 a100-fp16=true a100-gemm=true verify-fp16=true'
  ime_opts=(
    -expand-strided-metadata
    -assume-tight-memref-layout
    -staticize-memref-layout
    -lower-vector-softmax-erf
    -approximate-vector-exp
    "$ime_pass"
    -lower-ime=target=k3
    -expand-strided-metadata
  )
elif [[ "${BUDDY_SINGLE_FORWARD_DECODE_TB:-OFF}" == ON ]]; then
  ime_opts=(
    -expand-strided-metadata
    -assume-tight-memref-layout
    -staticize-memref-layout
    -lower-vector-softmax-erf
    -approximate-vector-exp
  )
fi

decode_tb_opts=()
if [[ "${BUDDY_SINGLE_FORWARD_DECODE_TB:-OFF}" == ON ]]; then
  decode_tb_opts+=(-matmul-transpose-b-vectorization-decode="vector-size=32 unroll=4 n-tile=4")
fi

batch_opts=(-batchmatmul-optimize)
if [[ "${BUDDY_SINGLE_FORWARD_SKIP_BATCH:-OFF}" == ON ]]; then
  batch_opts=()
fi

affine_fusion_opts=(-affine-loop-fusion)
if [[ "${BUDDY_SINGLE_FORWARD_SKIP_AFFINE_FUSION:-OFF}" == ON ]]; then
  affine_fusion_opts=()
fi

"$LLVM_BIN/mlir-opt" "$IN" \
  -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))' \
  | "$LLVM_BIN/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
  | "$BUDDY_OPT" \
      "${pre_buf_opts[@]}" \
      -arith-expand \
      -eliminate-empty-tensors \
      -convert-elementwise-to-linalg \
      -empty-tensor-to-alloc-tensor \
      -one-shot-bufferize=bufferize-function-boundaries \
      -ownership-based-buffer-deallocation \
      -buffer-deallocation-simplification \
      -bufferization-lower-deallocations \
      "${post_buf_opts[@]}" \
      "${ime_opts[@]}" \
      "${decode_tb_opts[@]}" \
      -matmul-parallel-vectorization-optimize \
      "${batch_opts[@]}" \
      -convert-linalg-to-affine-loops \
      "${affine_fusion_opts[@]}" \
      -affine-parallelize \
      -lower-affine \
      -canonicalize \
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
  | "$LLVM_BIN/llc" -filetype=obj -relocation-model=pic \
      "${LLC_ATTRS[@]}" "${a100_llc_attrs[@]}" -O3 -o "$OUT"
