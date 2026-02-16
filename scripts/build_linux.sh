#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "PROJECT_ROOT=$PROJECT_ROOT"

BUDDY_BUILD_DIR=${BUDDY_BUILD_DIR:-"$PROJECT_ROOT/build"}
export BUDDY_BUILD_DIR

for path in "$BUDDY_BUILD_DIR/python_packages" "$BUDDY_BUILD_DIR/bin" "$BUDDY_BUILD_DIR/lib"; do
  if [[ ! -d "$path" ]]; then
    echo "Expected build output missing: $path"
    echo "Configure & build with -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON before packaging."
    exit 1
  fi
done

# python3 -m build --wheel --outdir "$PROJECT_ROOT/build/dist"
auditwheel repair "${BUDDY_BUILD_DIR}/dist"/buddy-*.whl -w "${BUDDY_BUILD_DIR}/dist"
