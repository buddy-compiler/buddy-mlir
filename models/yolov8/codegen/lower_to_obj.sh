#!/usr/bin/env bash
# Lower subgraph0.mlir of YOLOv8 to a relocatable object. Invoked by
# buddy_add_model (LOWER_SCRIPT) from models/yolov8/CMakeLists.txt.
#
# Differs from the default single_forward pipeline of buddy_model.cmake:
# - the unit batch dimension is folded away first, so that the parallel loop
#   of each elementwise op and transpose is not the batch loop of 1 trip;
# - the linalg ops go through VIR (-lower-linalg-to-vir; the convolutions
#   vectorized along the output channels, their output rows in parallel);
#   -lower-vir-to-vector sizes the vectors for the target of the llc options,
#   as for the generated kernels of the other models
#   (compile_pipeline.kernels_lower_vir; the K3: rvv-vlen=1024);
# - no -affine-loop-fusion, which miscompiles this graph (the outputs of all
#   anchors come out the same).
#
# With the "ime" spec option the importer also writes yolov8_kernels.mlir
# next to subgraph0.mlir (codegen/yolov8_ime.py: the convolutions on the
# matrix engine of the A100 cores). It is lowered with the IME dialect,
# linked with the subgraph, and the whole object compiled for the A100
# (-mcpu=spacemit-a100, VLEN 1024, the IME extension), scheduled top down.
#
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
set -euo pipefail
BUDDY_OPT="$1"; LLVM_BIN="$2"; IN="$3"; OUT="$4"
shift 4 || true
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")
CODEGEN="$(dirname "$0")/../../../tools/buddy-codegen"
LOWER_VIR="$("${PYTHON:-python3}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from compile_pipeline import kernels_lower_vir; print(kernels_lower_vir(sys.argv[2:]))' "$CODEGEN" "${LLC_ATTRS[@]}")"
BUDDY_TRANSLATE="$(dirname "$BUDDY_OPT")/buddy-translate"
KERNELS="$(dirname "$IN")/yolov8_kernels.mlir"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

TOSA="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"

"$LLVM_BIN/mlir-opt" "$IN" -pass-pipeline "$TOSA" \
  | "$BUDDY_OPT" \
      -linalg-fold-unit-extent-dims -canonicalize \
      -arith-expand -eliminate-empty-tensors -convert-elementwise-to-linalg \
      -empty-tensor-to-alloc-tensor \
      -one-shot-bufferize=bufferize-function-boundaries \
      -ownership-based-buffer-deallocation \
      -buffer-deallocation-simplification \
      -bufferization-lower-deallocations \
      -lower-linalg-to-vir "$LOWER_VIR" \
      -scf-forall-to-parallel \
      -convert-linalg-to-affine-loops -affine-parallelize -lower-affine \
      -convert-scf-to-openmp -convert-linalg-to-loops -convert-vector-to-scf \
      -expand-strided-metadata -lower-affine -cse -convert-vector-to-llvm \
      -memref-expand -convert-arith-to-llvm -finalize-memref-to-llvm \
      -convert-scf-to-cf -convert-cf-to-llvm -llvm-request-c-wrappers \
      -convert-openmp-to-llvm -convert-arith-to-llvm -convert-math-to-llvm \
      -convert-math-to-libm -convert-func-to-llvm -reconcile-unrealized-casts \
  | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir -o "$TMP/subgraph.ll"

if [[ ! -f "$KERNELS" ]]; then
  "$LLVM_BIN/llvm-as" "$TMP/subgraph.ll" -o - \
    | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" -filetype=obj -relocation-model=pic -O3 -o "$OUT"
  exit 0
fi

# The IME kernels need exactly VLEN 1024 (vscale 16).
for a in "${LLC_ATTRS[@]}"; do
  if [[ "$a" == -riscv-v-vector-bits-max=* && "$a" != -riscv-v-vector-bits-max=1024 ]]; then
    echo "lower_to_obj.sh: the IME kernels need VLEN 1024, the build gives $a" >&2
    exit 1
  fi
done

"$BUDDY_OPT" "$KERNELS" \
    -lower-ime=target=k3 -convert-scf-to-openmp \
    -expand-strided-metadata -convert-vector-to-scf -lower-affine \
    -expand-strided-metadata -memref-expand -arith-expand \
    -convert-vector-to-llvm -convert-ub-to-llvm -convert-arith-to-llvm \
    -finalize-memref-to-llvm -convert-scf-to-cf -convert-cf-to-llvm \
    -llvm-request-c-wrappers -convert-openmp-to-llvm -convert-arith-to-llvm \
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm \
    -reconcile-unrealized-casts \
  | "$BUDDY_TRANSLATE" --buddy-to-llvmir -o "$TMP/kernels.ll"

"$LLVM_BIN/llvm-link" "$TMP/subgraph.ll" "$TMP/kernels.ll" -o "$TMP/all.bc"
"$LLVM_BIN/llc" "${LLC_ATTRS[@]}" \
    -mattr=+xsmtvdotii,+zvl1024b -mcpu=spacemit-a100 \
    -misched-prera-direction=topdown -riscv-v-vector-bits-max=1024 \
    -filetype=obj -relocation-model=pic -O3 -o "$OUT" "$TMP/all.bc"
