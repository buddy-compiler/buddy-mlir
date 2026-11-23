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
    sys.exit(f"lower_single_forward.sh: {e}")
' "$codegen_dir" "${llc_attrs[@]}")"
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
  if [[ -n "${BUDDY_SINGLE_FORWARD_IME_SHAPES:-}" ]]; then
    ime_pass+=" matmul-shapes=${BUDDY_SINGLE_FORWARD_IME_SHAPES}"
  fi
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

"$llvm_bin/mlir-opt" "$input" \
  -pass-pipeline 'builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith))' \
  | "$llvm_bin/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
  | "$buddy_opt" \
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
  | "$llvm_bin/mlir-translate" -mlir-to-llvmir \
  | "$llvm_bin/llvm-as" \
  | "$llvm_bin/llc" -filetype=obj -relocation-model=pic \
      "${llc_attrs[@]}" "${a100_llc_attrs[@]}" -O3 -o "$output"
