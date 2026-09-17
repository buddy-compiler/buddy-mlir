#!/usr/bin/env bash
# Historical unreviewed images are deliberately not reused.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_model.sh" "${MODEL_RUN_DIR:-$script_dir/../build/review-4l/run}" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900 "$@"
