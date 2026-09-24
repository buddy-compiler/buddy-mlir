#!/usr/bin/env bash
# ===- env.sh - K3 environment for BuddyBgeM3 tools (user-mode, no python) ===//
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
# ===----------------------------------------------------------------------===//
#
# Usage: source ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/env.sh
# All k3/ scripts source this file automatically.

export ROOT="${ROOT:-$HOME/buddy-k3}"
export REPO="${REPO:-$ROOT/buddy-mlir}"
export BUILD="$REPO/build"
export LLVM="$REPO/llvm/build"
export LLVMBIN="$LLVM/bin"
export LLVMLIB="$LLVM/lib"
export BUDDYBIN="$BUILD/bin"
export APP="${APP:-$REPO/examples/BuddyBgeM3}"

export SRC="$APP/src"          # MLIR / weights / manifest from x86
export OUT="$APP/out"          # build products (.o / .so / .rax)
export RESULTS="$APP/results"  # raw benchmark data
export PROFILE="$APP/profile"  # RVV instruction statistics

# rv64 llc options (same as buddy_model.cmake cross mode; works natively).
# Default-value semantics: callers may pre-export LLC_ATTRS (e.g. exp_g2.sh).
export LLC_ATTRS="${LLC_ATTRS:--march=riscv64 -mattr=+m,+d,+v \
  -mtriple=riscv64-unknown-linux-gnu}"

# Runtime library lookup for dlopen'ed model .so files: libomp.so and
# libmlir_c_runner_utils.so live in the LLVM build tree; add their dirs to
# LD_LIBRARY_PATH so that the .so can resolve transitive dependencies.
_omp="$(find "$LLVM" -maxdepth 6 -name 'libomp.so' -print -quit 2>/dev/null)"
_runner="$(find "$LLVM" -maxdepth 6 \
  -name 'libmlir_c_runner_utils.so' -print -quit 2>/dev/null)"
for _d in "$LLVM/lib" \
  "$(dirname "$_omp" 2>/dev/null)" \
  "$(dirname "$_runner" 2>/dev/null)"; do
  [ -d "$_d" ] \
    && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_d"
done
unset _omp _runner _d

mkdir -p "$OUT" "$RESULTS" "$PROFILE"

