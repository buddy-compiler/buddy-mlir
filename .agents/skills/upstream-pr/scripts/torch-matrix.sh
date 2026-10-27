#!/bin/bash
# ===- torch-matrix.sh ---------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Runs the tests against every torch version CI tests (TestBuild.yml's
# x86 torch_versions, newest first), like CI: the whole check-buddy for the
# newest, check-tests + check-buddy-examples-buddyjit for the others.
#
#   .agents/skills/upstream-pr/scripts/torch-matrix.sh [build-dir] [scratch-venv]
#
# From the repository root; defaults: build, /tmp/buddy-torch-matrix. The
# build is configured as usual (its own venv untouched): the scratch venv
# gets requirements.txt and then each torch / torchvision from the CPU index,
# and lit's python_executable in <build-dir>/tests/lit.site.cfg.py points at
# it while the script runs (restored on exit). Logs: <scratch-venv>/check-*.log.
# Needs network access to PyPI and download.pytorch.org.
#
# ===---------------------------------------------------------------------------

set -u
ROOT=$(pwd)
BUILD=$(cd "${1:-build}" && pwd)
VENV=${2:-/tmp/buddy-torch-matrix}
CFG=$BUILD/tests/lit.site.cfg.py
[ -f "$CFG" ] || { echo "no $CFG: configure and build $BUILD first" >&2; exit 2; }
[ -f "$ROOT/.github/workflows/TestBuild.yml" ] || { echo "run from the repository root" >&2; exit 2; }

# The x86 list: the last torch_versions=(...) line of the workflow.
VERSIONS=$(grep -o 'torch_versions=([^)]*)' "$ROOT/.github/workflows/TestBuild.yml" |
  tail -n 1 | sed 's/.*(\(.*\))/\1/')
[ -n "$VERSIONS" ] || { echo "no torch_versions in TestBuild.yml" >&2; exit 2; }
echo "torch versions (TestBuild.yml): $VERSIONS"

BUILD_PY=$(sed -n 's/^config.python_executable = "\(.*\)"/\1/p' "$CFG")
if [ ! -x "$VENV/bin/python" ]; then
  "$BUILD_PY" -m venv "$VENV" || exit 2
  "$VENV/bin/pip" install -q -r "$ROOT/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu || exit 2
fi

cp "$CFG" "$VENV/lit.site.cfg.py.orig"
trap 'cp "$VENV/lit.site.cfg.py.orig" "$CFG"' EXIT
sed -i "s#^config.python_executable = .*#config.python_executable = \"$VENV/bin/python\"#" "$CFG"

failed=0
first=1
for v in $VERSIONS; do
  echo "== torch $v"
  "$VENV/bin/pip" install -q --force-reinstall "torch==$v.*" torchvision \
    --index-url https://download.pytorch.org/whl/cpu > "$VENV/pip-$v.log" 2>&1 ||
    { echo "   pip failed, see $VENV/pip-$v.log"; failed=1; continue; }
  "$VENV/bin/python" -c 'import torch, torchvision; print("   torch", torch.__version__, "torchvision", torchvision.__version__)'
  if [ $first = 1 ]; then targets=check-buddy; else targets="check-tests check-buddy-examples-buddyjit"; fi
  first=0
  if ninja -C "$BUILD" $targets > "$VENV/check-$v.log" 2>&1; then
    echo "   passed ($targets)"
  else
    failed=1
    echo "   FAILED ($targets), see $VENV/check-$v.log:"
    grep -E "^  [A-Za-z]+ :: |Failed *:" "$VENV/check-$v.log" | sed 's/^/   /'
  fi
done
exit $failed
