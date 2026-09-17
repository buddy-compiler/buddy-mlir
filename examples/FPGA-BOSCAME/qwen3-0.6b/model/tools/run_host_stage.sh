#!/usr/bin/env bash
# Reproduce the 4-layer host stages used to check the compiled graph numerically.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
PY="${PYTHON:-python3}"
PROMPT=151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
LAYERS="${LAYERS:-4}"

export PYTHONPATH="$REPO/build-python/python_packages"

"$PY" -B "$MODEL/tools/import_model.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --output "$MODEL/build/import/probe-${LAYERS}layer" --layers "$LAYERS" \
  --prefill-len 16 --max-cache-len 512 --fuse none --save-mlir --init-cache 2>&1 | tail -2

python3 -B "$MODEL/tools/weight_layout.py" \
  --mlir "$MODEL/build/import/probe-${LAYERS}layer/forward_decode.mlir" \
  --params "$MODEL/build/import/probe-${LAYERS}layer/params.json" \
  --checkpoint "$MODEL/assets/checkpoint/model.safetensors" \
  --output "$MODEL/build/import/probe-${LAYERS}layer/weight-layout.json" \
  > "$MODEL/build/import/probe-${LAYERS}layer/weight-layout.log" 2>&1 || {
    echo "weight layout failed:"; tail -20 "$MODEL/build/import/probe-${LAYERS}layer/weight-layout.log"; exit 1; }
python3 -c "import json,sys;d=json.load(open('$MODEL/build/import/probe-${LAYERS}layer/weight-layout.json'));print('layout',d['status'],d['section_count'],'sections',d['problems'])"

"$PY" -B "$MODEL/tools/host_reference.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --output "$MODEL/build/reference/fp32-${LAYERS}layer-16plus8" \
  --prompt-ids "$PROMPT" --decode-steps 8 --max-cache-len 512 --layers "$LAYERS" \
  2>&1 | tail -10

"$PY" -B "$MODEL/tools/run_graph_host.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --layout "$MODEL/build/import/probe-${LAYERS}layer/weight-layout.json" \
  --output "$MODEL/build/host-run/full-${LAYERS}l" \
  --reference-dir "$MODEL/build/reference/fp32-${LAYERS}layer-16plus8" \
  --prompt-ids "$PROMPT" --decode-steps 8 --max-cache-len 512 --layers "$LAYERS"
