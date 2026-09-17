#!/usr/bin/env bash
# Build a host shared library exposing the same Triton kernels the NR archive
# contains, so the external-call graph can be executed and scored.
#
# The point is to test the *call chain*, not to test the board: the C symbols the
# graph calls (triton_<case>) and the kernels behind them come from the same
# per-case build tree that produced the RISC-V objects, only compiled for the
# host. Numeric agreement therefore says the replacement wired the right tensors
# into the right kernels.
#
# Layers, innermost first:
#   triton_<case>                  host/kernel.ll (the Triton kernel itself)
#   _mlir_ciface_kernel_<case>     build/<case>/adapter.c (descriptor + grid)
#   triton_<case> (graph-facing)   model/tools/triton_call_replace.py adapters
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
QWEN="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b"
TRITON_BUILD="${TRITON_BUILD:-$QWEN/triton/build}"
CC="${HOST_CC:-$REPO/llvm/build-2d26/bin/clang}"
OUT="${OUT:-$MODEL/build/host-bridge}"
ADAPTERS="${ADAPTERS:-$MODEL/build/triton-call/full-28l/qwen_triton_adapters.c}"

if [[ ! -f "$ADAPTERS" ]]; then
  echo "missing generated adapters: $ADAPTERS (run model/tools/triton_call_replace.py)" >&2
  exit 1
fi

# The cases the replacement table can reference: every case named in the
# adapters, discovered from the generated source so the two cannot drift.
CASES=$(grep -o '_mlir_ciface_kernel_[a-z0-9_]*' "$ADAPTERS" \
        | sed 's/^_mlir_ciface_kernel_//' | sort -u)
if [[ -z "$CASES" ]]; then
  echo "no archive entries referenced by $ADAPTERS" >&2
  exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"
FLAGS=(-O2 -fPIC -ffp-contract=off -DHOST_TEST -I"$QWEN")

objects=()
count=0
for case in $CASES; do
  if [[ ! -f "$TRITON_BUILD/$case/host/kernel.ll" ]]; then
    echo "missing host kernel IR for $case" >&2
    exit 1
  fi
  "$CC" "${FLAGS[@]}" -c "$TRITON_BUILD/$case/host/kernel.ll" \
    -o "$OUT/$case.kernel.o"
  "$CC" "${FLAGS[@]}" -c "$TRITON_BUILD/$case/adapter.c" \
    -o "$OUT/$case.ciface.o"
  objects+=("$OUT/$case.kernel.o" "$OUT/$case.ciface.o")
  count=$((count + 1))
done

"$CC" "${FLAGS[@]}" -c "$ADAPTERS" -o "$OUT/qwen_triton_adapters.o"
objects+=("$OUT/qwen_triton_adapters.o")

"$CC" -shared -o "$OUT/libqwen_triton_host.so" "${objects[@]}" -lm

echo "$count kernels -> $OUT/libqwen_triton_host.so"
"$REPO/llvm/build-2d26/bin/llvm-nm" --dynamic --defined-only \
  "$OUT/libqwen_triton_host.so" | grep -c " T triton_" || true
echo "graph-facing symbols exported (count above)"
"$REPO/llvm/build-2d26/bin/llvm-nm" --dynamic --undefined-only \
  "$OUT/libqwen_triton_host.so" | head -10 || true