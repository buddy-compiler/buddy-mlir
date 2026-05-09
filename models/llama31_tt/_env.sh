#!/usr/bin/env bash
# ===- _env.sh ------------------------------------------------------------===//
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
# Shared environment for the Llama-3.1-8B TTIR/Tenstorrent model path.
# Source this file from the repo root or this directory:
#
#   source models/llama31_tt/_env.sh
#
# It does not build tt-mlir. See docs/TenstorrentEnvironment.md for the
# optional tt-mlir/tt-metal setup.
#
# ===----------------------------------------------------------------------===//

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export BUDDY_BUILD="${BUDDY_BUILD:-${REPO_ROOT}/build}"
export TTMLIR_SOURCE="${TTMLIR_SOURCE:-${BUDDY_TT_MLIR_SOURCE_DIR:-${REPO_ROOT}/thirdparty/tt-mlir}}"
export TTMLIR_BUILD="${TTMLIR_BUILD:-${BUDDY_TT_MLIR_BUILD_DIR:-${TTMLIR_SOURCE}/build}}"
export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-${REPO_ROOT}/build-ttmlir-toolchain}"
export TTMLIR_VENV_DIR="${TTMLIR_VENV_DIR:-${TTMLIR_TOOLCHAIN_DIR}/venv}"

if [[ "${BUDDY_TT_SKIP_ACTIVATE:-0}" != "1" && -n "${BUDDY_TT_CONDA_ENV:-}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${BUDDY_TT_CONDA_ENV}"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate "${BUDDY_TT_CONDA_ENV}"
  fi
elif [[ "${BUDDY_TT_SKIP_ACTIVATE:-0}" != "1" && -f "${TTMLIR_SOURCE}/env/activate" ]]; then
  pushd "${TTMLIR_SOURCE}" >/dev/null
  # shellcheck disable=SC1091
  source env/activate
  popd >/dev/null
fi

if [[ -x "${TTMLIR_VENV_DIR}/bin/python" ]]; then
  export PATH="${TTMLIR_TOOLCHAIN_DIR}/bin:${TTMLIR_VENV_DIR}/bin:${PATH}"
fi

export TT_METAL_HOME="${TT_METAL_HOME:-${TTMLIR_SOURCE}/third_party/tt-metal/src/tt-metal}"

export PATH="${BUDDY_BUILD}/bin:${TTMLIR_BUILD}/bin:${PATH}"
export PYTHONPATH="${BUDDY_BUILD}/python_packages:${TTMLIR_BUILD}/python_packages${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${TTMLIR_BUILD}/lib" ]]; then
  export LD_LIBRARY_PATH="${TTMLIR_BUILD}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${TT_METAL_HOME}/build/lib" ]]; then
  export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export LLAMA31_MODEL_PATH="${LLAMA31_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
