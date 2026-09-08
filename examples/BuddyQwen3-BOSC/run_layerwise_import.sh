#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUDDY_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUDDY_DIR="$BUDDY_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python with torch/transformers not found: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$BUDDY_DIR/build/python_packages${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/import_qwen3_layerwise.py" "$@"
