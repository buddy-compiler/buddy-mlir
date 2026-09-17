#!/usr/bin/env bash
# Source after activating the Python environment used to build Triton.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source examples/FPGA-BOSCAME/common/triton/triton-env.sh" >&2
  exit 1
fi

_boscame_triton_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDDY_SOURCE_DIR="${BUDDY_SOURCE_DIR:-$(cd "${_boscame_triton_dir}/../../../.." && pwd)}"
export TRITON_PYTHON="${TRITON_PYTHON:-python}"
export TRITON_RISCV_DIR="${TRITON_RISCV_DIR:-${BUDDY_SOURCE_DIR}/thirdparty/triton-riscv}"
export TRITON_DIR="${TRITON_DIR:-${TRITON_RISCV_DIR}/triton}"
export TRITON_PLUGIN_DIRS="${TRITON_RISCV_DIR}"

if [[ -z "${LLVM_SYSPATH:-}" ]]; then
  for _boscame_candidate in "${BUDDY_SOURCE_DIR}/llvm/build-2d26" "${BUDDY_SOURCE_DIR}/llvm/build"; do
    if [[ -f "${_boscame_candidate}/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
      export LLVM_SYSPATH="${_boscame_candidate}"
      break
    fi
  done
fi
if [[ -z "${BUDDY_MLIR_BINARY_DIR:-}" ]]; then
  for _boscame_candidate in "${BUDDY_SOURCE_DIR}/build-migrate/bin" "${BUDDY_SOURCE_DIR}/build/bin"; do
    if [[ -x "${_boscame_candidate}/buddy-opt" ]]; then
      export BUDDY_MLIR_BINARY_DIR="${_boscame_candidate}"
      break
    fi
  done
fi
if [[ -n "${LLVM_SYSPATH:-}" ]]; then
  export LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${LLVM_SYSPATH}/bin}"
fi

# Let Triton's pinned downloader populate JSON when no local copy is available.
if [[ -z "${JSON_SYSPATH:-}" && -f "${HOME}/.triton/json/include/nlohmann/json.hpp" ]]; then
  export JSON_SYSPATH="${HOME}/.triton/json"
fi
if [[ -z "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  _boscame_python_tag="$("${TRITON_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || return 1
  for _boscame_candidate in "${TRITON_DIR}"/build/cmake.*-cpython-"${_boscame_python_tag}"/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt; do
    if [[ -x "${_boscame_candidate}" ]]; then
      export TRITON_SHARED_OPT_PATH="${_boscame_candidate}"
      break
    fi
  done
fi

# Keep the example's directory named "triton" from shadowing the real package.
export PYTHONSAFEPATH=1
unset _boscame_candidate _boscame_python_tag _boscame_triton_dir
