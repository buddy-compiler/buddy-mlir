#!/usr/bin/env bash
# Reproduce the 4-layer host stages used to check the compiled graph numerically.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
PY="${PYTHON:-python3}"
PROMPT=151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
LAYERS="${LAYERS:-4}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-128}"
IMPORT_DIR="$MODEL/build/import/probe-${LAYERS}layer-cap${MAX_CACHE_LEN}"
REFERENCE_DIR="$MODEL/build/reference/fp32-${LAYERS}layer-cap${MAX_CACHE_LEN}-16plus8"
HOST_RUN_DIR="$MODEL/build/host-run/full-${LAYERS}l-cap${MAX_CACHE_LEN}"

export PYTHONPATH="$REPO/build-python/python_packages"

"$PY" -B "$MODEL/tools/import_model.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --output "$IMPORT_DIR" --layers "$LAYERS" \
  --prefill-len 16 --max-cache-len "$MAX_CACHE_LEN" --fuse none --save-mlir --init-cache 2>&1 | tail -2

python3 -B "$MODEL/tools/weight_layout.py" \
  --mlir "$IMPORT_DIR/forward_decode.mlir" \
  --params "$IMPORT_DIR/params.json" \
  --checkpoint "$MODEL/assets/checkpoint/model.safetensors" \
  --output "$IMPORT_DIR/weight-layout.json" \
  > "$IMPORT_DIR/weight-layout.log" 2>&1 || {
    echo "weight layout failed:"; tail -20 "$IMPORT_DIR/weight-layout.log"; exit 1; }
python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("layout",d["status"],d["section_count"],"sections",d["problems"])' "$IMPORT_DIR/weight-layout.json"

"$PY" -B "$MODEL/tools/host_reference.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --output "$REFERENCE_DIR" \
  --prompt-ids "$PROMPT" --decode-steps 8 --max-cache-len "$MAX_CACHE_LEN" --layers "$LAYERS" \
  2>&1 | tail -10

"$PY" -B "$MODEL/tools/run_graph_host.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --layout "$IMPORT_DIR/weight-layout.json" \
  --output "$HOST_RUN_DIR" \
  --reference-dir "$REFERENCE_DIR" \
  --prompt-ids "$PROMPT" --decode-steps 8 --max-cache-len "$MAX_CACHE_LEN" --layers "$LAYERS"
