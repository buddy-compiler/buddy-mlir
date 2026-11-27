#!/usr/bin/env bash
# 批准后在构建机运行：复用算子对象，仅交叉构建接入代码及测试并打包。
set -euo pipefail
: "${BASE_ARTIFACT_DIR:?Set the previously verified artifact directory}"
: "${ARTIFACT_DIR:?Set a new empty output directory}"
: "${HOST_PYTHON:?Set the build-machine Python}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -e $ARTIFACT_DIR ]]; then
  echo 'ARTIFACT_DIR must not exist; never overwrite old evidence' >&2
  exit 1
fi
mkdir -p "$ARTIFACT_DIR"
for name in softmax.obj buddy.o computation.o SoftmaxTarget.h; do
  cp "$BASE_ARTIFACT_DIR/$name" "$ARTIFACT_DIR/$name"
done
cp "$BASE_ARTIFACT_DIR/export.json" "$ARTIFACT_DIR/source-export.json"
"$HOST_PYTHON" - "$repo/examples/TritonSoftmax" "$ARTIFACT_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from softmax_metadata import target_header, validate_export
out = Path(sys.argv[2])
source = out / 'source-export.json'
metadata = json.loads(source.read_text())
old = metadata.get('requires_single_cpu')
if type(old) is not bool:
    raise ValueError('source export must explicitly declare its CPU policy')
# This field describes the Adapter's deployment policy, not Triton lowering.
# Preserve the old record; validate every remaining field and the actual object.
metadata['requires_single_cpu'] = False
validate_export(metadata, out / 'softmax.obj')
metadata['adapter_policy_update'] = {
    'field': 'requires_single_cpu', 'old': old, 'new': False,
    'source_export_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
}
(out / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
if (out / 'SoftmaxTarget.h').read_text() != target_header(metadata):
    raise ValueError('source header does not match the verified kernel profile')
PY
sha256sum "$ARTIFACT_DIR/softmax.obj" "$ARTIFACT_DIR/buddy.o" \
  "$ARTIFACT_DIR/computation.o" > "$ARTIFACT_DIR/reused-objects.sha256"
bash "$repo/examples/TritonSoftmax/run.sh" standalone
bash "$repo/examples/TritonSoftmax/run.sh" integration
bash "$repo/examples/TritonSoftmax/run.sh" rax
"$HOST_PYTHON" "$repo/examples/TritonSoftmax/build_contract_tests.py" \
  --build-dir "${BUDDY_BUILD_DIR:-$repo/build}" --artifact-dir "$ARTIFACT_DIR"
"$HOST_PYTHON" "$repo/examples/TritonSoftmax/package_tests.py" \
  --build-dir "${BUDDY_BUILD_DIR:-$repo/build}" --artifact-dir "$ARTIFACT_DIR" \
  --llvm-bin "${LLVM_BUILD_DIR:-$repo/llvm/build}/bin"
