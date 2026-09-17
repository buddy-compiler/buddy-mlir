#!/usr/bin/env bash
# Run the model-shape Triton kernels on the board, one at a time.
#
# These 11 kernels are the specialisations the imported Qwen3 graph needs and
# that the shared 72-case set does not cover (attention over the full 512-slot
# cache, a mask boundary and a KV slot that come from runtime data). They were
# built and passed the full ELF audit, but had never been executed on hardware.
#
# Runs are strictly sequential and go through the shared fpga_run.sh, so only one
# process ever owns the UART and an interrupted session can be reattached with
# --resume-run. Nothing here uploads weights or touches another user's run.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
BUILD="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/triton/build"
OUT="${OUT:-$MODEL/build/fpga-model-kernels}"
FPGA="${FPGA:-5}"
CAPTURE="${CAPTURE:-300}"

CASES=(
  attention_qk_16x16x512x128
  attention_qk_16x1x512x128
  attention_pv_16x16x128x512
  attention_pv_16x1x128x512
  attention_scale_mask_position_16x16x512
  attention_scale_mask_position_16x1x512
  softmax_16x16x512
  softmax_16x1x512
  kv_cache_update_position_16x8x128_cap512
  kv_cache_update_position_1x8x128_cap512
  gqa_repeat_8x512x128_to_16x128x128
)

mkdir -p "$OUT"
pass=0; fail=0; skipped=0
: > "$OUT/summary.txt"

for case in "${CASES[@]}"; do
  image="$BUILD/$case/nr/$case.bin"
  if [[ "$case" == "gqa_repeat_8x512x128_to_16x128x128" ]]; then
    # Name guard: the real case is ..._to_16x512x128. Fail loudly rather than
    # silently skipping a kernel.
    image="$BUILD/gqa_repeat_8x512x128_to_16x512x128/nr/gqa_repeat_8x512x128_to_16x512x128.bin"
    case="gqa_repeat_8x512x128_to_16x512x128"
  fi
  if [[ ! -f "$image" ]]; then
    echo "MISSING $case" | tee -a "$OUT/summary.txt"
    skipped=$((skipped + 1))
    continue
  fi
  log="$OUT/$case.log"
  echo "=== $case ==="
  if timeout 900 "$REPO/examples/FPGA-BOSCAME/fpga_run.sh" "$image" \
       --fpga="$FPGA" --capture-seconds="$CAPTURE" \
       --completion-marker='[nr] RA returned:' > "$log" 2>&1; then
    result=$(grep -oE "verify $case: (PASS|FAIL)[^\"]*" "$log" | head -1)
    cycles=$(grep -oE "launch cycles=0x[0-9A-Fa-f]+" "$log" | head -1)
    runtime=$(grep -oE "verify NR runtime: (PASS|FAIL)" "$log" | head -1)
    runid=$(grep -oE "run-[0-9a-f]{16}" "$log" | head -1)
    if grep -q "verify $case: PASS" "$log"; then
      pass=$((pass + 1))
      echo "PASS  $case  $cycles  $runtime  $runid" | tee -a "$OUT/summary.txt"
    else
      fail=$((fail + 1))
      echo "FAIL  $case  $result  $runid" | tee -a "$OUT/summary.txt"
    fi
  else
    fail=$((fail + 1))
    echo "ERROR $case (fpga_run.sh nonzero; see $log)" | tee -a "$OUT/summary.txt"
  fi
done

echo "---"
echo "pass=$pass fail=$fail skipped=$skipped" | tee -a "$OUT/summary.txt"