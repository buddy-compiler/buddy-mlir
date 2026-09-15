#!/usr/bin/env bash
# Negative check for the FPGA matrix-register capacity.
#
# The FPGA convention has eight accumulator registers and no lossless spill for
# them (`msce32.m` converts i32 -> f32, so it cannot save the accumulator), so a
# kernel that needs a ninth resident accumulator must be rejected by the
# compiler instead of being silently miscompiled.  The diagnostic terminates the
# process, so the check runs the compiler under a real shell.
#
# Usage: check-matrix-spill.sh <buddy-llc> <input.ll>
set -u
LLC="$1"
SRC="$2"

# The capacity rule must hold at every optimisation level, so each one is
# checked with the same diagnostic expectation.
for OPT in -O0 -O2 -O3; do
  ERR="${SRC}.${OPT}.err"
  "$LLC" "$SRC" -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame "$OPT" \
      -o /dev/null > /dev/null 2> "$ERR"
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    echo "FAIL: $OPT accepted a kernel with nine accumulator chains"
    rm -f "$ERR"
    exit 1
  fi
  if ! grep -q "BOSC AME matrix registers cannot be spilled" "$ERR"; then
    echo "FAIL: $OPT did not emit the expected diagnostic"
    cat "$ERR"
    rm -f "$ERR"
    exit 1
  fi
  rm -f "$ERR"
done

echo "PASS: rejected at -O0, -O2 and -O3 with the FPGA accumulator capacity diagnostic"
