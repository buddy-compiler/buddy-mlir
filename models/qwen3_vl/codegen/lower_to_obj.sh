#!/usr/bin/env bash
# ===- lower_to_obj.sh - Lower Qwen3-VL MLIR to a relocatable object -----===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# Lower one buddy/TOSA MLIR module to a relocatable object.
# Invoked by models/qwen3_vl CMake (a piped pipeline like this cannot be
# expressed directly in add_custom_command).
#
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> \
#          [threads] [llc-attrs...]
#
# When LLC attrs include +xsmtime, matmuls are lowered through IME (f16
# vfmadot) and assembled with buddy-llc. That path is for boards that
# implement XSMTIME. SpacemiT X100 (K3) implements XSMTVDot (smt.vmadot,
# integer only) and SIGILLs on vfmadot — keep USE_IME off for those builds
# (mattr may list +xsmtvdot; that alone does not enable IME). Host builds
# keep the OpenMP/BLIS pipeline and llvm llc.
#
# ===----------------------------------------------------------------------===//
set -euo pipefail
BUDDY_OPT="$1"; LLVM_BIN="$2"; IN="$3"; OUT="$4"
THREADS="${5:-1}"
shift 4 || true
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")

# IME / XSMTIME is opt-in: llc attrs must mention xsmtime, or QWEN3_VL_USE_IME=1.
# +xsmtvdot alone is not IME — it only advertises the X100 integer matrix ISA.
USE_IME=0
if [[ "${QWEN3_VL_USE_IME:-0}" == "1" ]]; then
  USE_IME=1
fi
for attr in "${LLC_ATTRS[@]+"${LLC_ATTRS[@]}"}"; do
  case "$attr" in
    *xsmtime*) USE_IME=1 ;;
  esac
done

# X100 boards must not silently emit vfmadot.
if [[ "$USE_IME" -eq 1 ]]; then
  for attr in "${LLC_ATTRS[@]+"${LLC_ATTRS[@]}"}"; do
    case "$attr" in
      *xsmtvdot*)
        echo "error: refusing IME/XSMTIME lowering with +xsmtvdot (X100); use RVV FP16 or INT8+smt.vmadot" >&2
        exit 1
        ;;
    esac
  done
fi

TOSA="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))"

if [[ "$USE_IME" -eq 1 ]]; then
  BUDDY_BIN="$(dirname "$BUDDY_OPT")"
  if [[ ! -x "$BUDDY_BIN/buddy-llc" || ! -x "$BUDDY_BIN/buddy-translate" ]]; then
    if [[ -n "${BUDDY_MLIR_BUILD_DIR:-}" && -x "${BUDDY_MLIR_BUILD_DIR}/bin/buddy-llc" ]]; then
      BUDDY_BIN="${BUDDY_MLIR_BUILD_DIR}/bin"
    fi
  fi
  if [[ ! -x "$BUDDY_BIN/buddy-llc" || ! -x "$BUDDY_BIN/buddy-translate" ]]; then
    echo "error: buddy-llc/buddy-translate not next to $BUDDY_OPT or in \$BUDDY_MLIR_BUILD_DIR/bin" >&2
    exit 1
  fi
  # IME already owns f16 matmuls (vfmadot). Do not run BLIS/batchmatmul-optimize
  # afterwards — those passes assert on ops IME has already rewritten.
  "$BUDDY_OPT" "$IN" -simplify-tosa-reshape \
    | "$LLVM_BIN/mlir-opt" -pass-pipeline "$TOSA" \
    | "$BUDDY_OPT" \
        -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
        -convert-elementwise-to-linalg \
        -one-shot-bufferize=bufferize-function-boundaries \
        -ownership-based-buffer-deallocation \
        -buffer-deallocation-simplification \
        -bufferization-lower-deallocations \
        -convert-bufferization-to-memref \
        -expand-strided-metadata -canonicalize -cse \
        -canonicalize -optimize-allocation-liveness \
        -lower-linalg-to-ime \
        -lower-ime \
        -convert-linalg-to-affine-loops -affine-parallelize -convert-vector-to-scf \
        -lower-affine -convert-scf-to-openmp=num-threads="${THREADS}" -cse -memref-expand \
        -arith-expand -convert-vector-to-llvm -convert-arith-to-llvm \
        -finalize-memref-to-llvm -convert-scf-to-cf -convert-cf-to-llvm \
        -llvm-request-c-wrappers -convert-openmp-to-llvm -convert-arith-to-llvm \
        -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm \
        -reconcile-unrealized-casts \
    | "$BUDDY_BIN/buddy-translate" -buddy-to-llvmir \
    | "$LLVM_BIN/llvm-as" \
    | "$BUDDY_BIN/buddy-llc" -filetype=obj -relocation-model=pic -O3 \
        -mtriple=riscv64-unknown-linux-gnu \
        -mattr=+m,+d,+v,+zfh,+zvfh,+xsmtime \
        -o "$OUT"
else
  # Decode graphs are M=1 GEMV-dominated; BLIS (mr=8 pack) wastes bandwidth.
  # Prefer decode vectorization (same idea as DeepSeek compile_pipeline).
  # Override with QWEN3_VL_MATMUL_KIND=blis|decode.
  MATMUL_KIND="${QWEN3_VL_MATMUL_KIND:-}"
  if [[ -z "$MATMUL_KIND" ]]; then
    case "$IN" in
      *decode*) MATMUL_KIND=decode ;;
      *) MATMUL_KIND=blis ;;
    esac
  fi
  if [[ "$MATMUL_KIND" == "decode" ]]; then
    # vector-size=16 matches +zvl256b f16. Prefer packed GEMV when shapes file exists.
    PACKED_SHAPES_FILE="${IN%/*}/decoder_decode_packed_shapes.txt"
    if [[ -f "$PACKED_SHAPES_FILE" ]] && [[ -n "$(head -n1 "$PACKED_SHAPES_FILE")" ]]; then
      # Empty packed-shapes ⇒ rewrite every m==1 linalg.matmul (attention is
      # batch_matmul and stays on the batch decode pass). Matches DeepSeek.
      MATMUL_PASSES=(
        -matmul-vectorization-decode-packed=vector-size=16
        -matmul-vectorization-decode=vector-size=16
        -batch-matmul-vectorization-decode=vector-size=16
      )
      echo "[lower] decode packed GEMV (vector-size=16, shapes file present)"
    else
      MATMUL_PASSES=(
        -matmul-vectorization-decode=vector-size=16
        -batch-matmul-vectorization-decode=vector-size=16
      )
      echo "[lower] decode plain GEMV (no packed shapes file)"
    fi
  else
    MATMUL_PASSES=(
      -matmul-vectorization-blis
      -batchmatmul-optimize=vector-size=16
    )
  fi
  "$BUDDY_OPT" "$IN" -simplify-tosa-reshape \
    | "$LLVM_BIN/mlir-opt" -pass-pipeline "$TOSA" \
    | "$BUDDY_OPT" \
        -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
        -convert-elementwise-to-linalg \
        -one-shot-bufferize=bufferize-function-boundaries \
        -ownership-based-buffer-deallocation \
        -buffer-deallocation-simplification \
        -bufferization-lower-deallocations \
        -convert-bufferization-to-memref \
        -expand-strided-metadata -canonicalize -cse \
        -canonicalize -optimize-allocation-liveness \
        "${MATMUL_PASSES[@]}" \
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
fi
