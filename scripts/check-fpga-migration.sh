#!/usr/bin/env bash
# Automated part of the FPGA migration verification matrix
# (docs/BOSCAMEFPGAValueSemantics.md, section 7b).
#
# Usage: scripts/check-fpga-migration.sh [buddy-build] [buddy-python-build] [llvm-build]
# Defaults: build-migrate, build-python, llvm/build-2d26
#
# It runs every check that does not need external hardware.  The two acceptance
# steps that do need it (GEM5/RTL numeric run, and the ISA answers about
# mmve eew / cross-class index semantics) are printed as pending at the end.
set -uo pipefail
CHECK_TMP=$(mktemp -d)
trap 'rm -rf "$CHECK_TMP"' EXIT

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BM="${1:-$REPO/build-migrate}"
BM_PY="${2:-$REPO/build-python}"
LLVMB="${3:-$REPO/llvm/build-2d26}"
LIT="$LLVMB/bin/llvm-lit"

fail=0
step() { printf '\n=== %s ===\n' "$1"; }

if [ ! -x "$LIT" ]; then
  echo "error: llvm-lit not found at $LIT (build the target LLVM first)" >&2
  exit 2
fi

step "upstream BOSCAMEDialect examples"
"$LIT" -s "$BM/examples" --filter='BOSCAMEDialect' | tail -4 || fail=1

step "BOSCAME/FPGA lit tests"
"$LIT" -s "$BM/tests" --filter='(QwenW8A8|linalg-to-boscame|Target/RISCV)' | tail -4 || fail=1

if [ -d "$BM_PY/tests" ]; then
  step "Python bindings and frontend tests"
  "$LIT" -s "$BM_PY/tests" --filter='Python' | tail -4 || fail=1
else
  echo "FAIL: $BM_PY/tests missing - Python verification is required"
  fail=1
fi

step "default pathway stays upstream (no FPGA markers)"
if ! "$BM/bin/buddy-opt" "$REPO/tests/Conversion/QwenW8A8/qwen-w8a8-default-upstream.mlir" \
     -lower-linalg-to-boscame --lower-bosc-ame -o "$CHECK_TMP/default.mlir"; then
  echo "FAIL: default pathway compilation"
  fail=1
elif grep -qE '65552|65602|xboscame-fpga|bosc_ame\.target|bosc_ame\.msettype %' "$CHECK_TMP/default.mlir"; then
  echo "FAIL: the default pathway emitted an FPGA-only marker"
  fail=1
else
  echo "ok: no bit-field constant, no FPGA target feature, no profile attribute"
fi

step "FPGA assembly smoke test (the target feature travels in the IR)"
"$BM/bin/buddy-opt" "$REPO/tests/Conversion/QwenW8A8/qwen-w8a8-accumulator-chain.mlir" \
    '--lower-linalg-to-boscame=target=qwen3-fpga triton-w8a8-fast-path=true' \
    --lower-bosc-ame -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
    -expand-strided-metadata -lower-affine -convert-cf-to-llvm \
    -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm \
    -finalize-memref-to-llvm -reconcile-unrealized-casts \
  | "$BM/bin/buddy-translate" --buddy-to-llvmir \
  | "$LLVMB/bin/llc" -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame \
      -verify-machineinstrs -o "$CHECK_TMP/fpga.s" - \
  && grep -q 'mqma.b.mm' "$CHECK_TMP/fpga.s" \
  && echo "ok: llc needed no -mattr=+xboscame-fpga and emitted the AME instructions" \
  || { echo "FAIL: FPGA assembly smoke test"; fail=1; }

step "pending external input"
cat <<'EOF'
- GEM5 or RTL/board numeric acceptance: needs a runner command and a golden
  baseline from the hardware owner (plan phase 4).
- ISA answers: `mmve*.t.t` / `.a.a` eew semantics, and the meaning of the GPR
  index of the cross-class moves (plan Q2/Q3).
- General role types: the owner selected role-level separation. The fixed-slot
  W8A8 adapter is implemented; removing the carrier-width heuristic remains
  separate work (docs/BOSCAMEFPGARoleABI.md).
EOF

printf '\n=== summary ===\n'
if [ "$fail" -eq 0 ]; then
  echo "all automated checks passed"
else
  echo "at least one automated check FAILED"
fi
exit "$fail"
