#!/usr/bin/env bash
# One runner/UART owner; fpga_run owns upload, verification and reconnect.
set -euo pipefail
model_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 ]]; then
  echo 'usage: run_model.sh PREPARED_IMAGE_DIR [fpga_run options, e.g. --fpga=5]' >&2
  exit 2
fi
run_dir=$1
shift
segments=(--segment "$run_dir/weights-w8a8.bin")
if [[ -f "$run_dir/tokenizer.bin" ]]; then
  segments+=(--segment "$run_dir/tokenizer.bin")
fi
exec "$model_dir/../../fpga_run.sh" "$run_dir/image.bin" \
  --layout-plan "$run_dir/ddr-load.plan" "${segments[@]}" \
  --completion-marker='[nr] RA returned:' "$@"
