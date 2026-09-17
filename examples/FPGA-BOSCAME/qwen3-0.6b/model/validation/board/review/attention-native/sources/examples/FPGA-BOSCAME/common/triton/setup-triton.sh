#!/usr/bin/env bash
# Build the pinned frontend against this checkout's already-built Buddy/LLVM.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
check_only=0
skip_deps=0
jobs="${MAX_JOBS:-24}"
for arg in "$@"; do
  case "${arg}" in
    --check) check_only=1 ;;
    --skip-deps) skip_deps=1 ;;
    --jobs=*) jobs="${arg#*=}" ;;
    --help|-h)
      echo "Usage: setup-triton.sh [--check] [--skip-deps] [--jobs=N]"
      echo "Activate boscame first, or set TRITON_PYTHON=/path/to/python."
      echo "Build paths: LLVM_SYSPATH and BUDDY_MLIR_BINARY_DIR."
      exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be positive" >&2; exit 2; }
source "${script_dir}/triton-env.sh"
"${TRITON_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "This lock was validated with Python 3.12; activate boscame or set TRITON_PYTHON"'

for required in "${LLVM_SYSPATH:-}/lib/cmake/llvm/LLVMConfig.cmake" \
                "${LLVM_SYSPATH:-}/lib/cmake/mlir/MLIRConfig.cmake" \
                "${BUDDY_MLIR_BINARY_DIR:-}/buddy-opt" \
                "${BUDDY_MLIR_BINARY_DIR:-}/buddy-translate"; do
  [[ -f "${required}" ]] || { echo "Missing ${required}; build Buddy/LLVM first or override their paths." >&2; exit 1; }
done

lock_value() {
  "${TRITON_PYTHON}" -c 'import json,sys; obj=json.load(open(sys.argv[1])); print(obj[sys.argv[2]][sys.argv[3]])' \
    "${script_dir}/toolchain-lock.json" "$1" "$2"
}
checkout_locked() {
  local directory="$1" url="$2" revision="$3"
  if [[ ! -e "${directory}/.git" ]]; then
    [[ "${check_only}" == 0 ]] || { echo "Missing checkout: ${directory}" >&2; return 1; }
    [[ ! -e "${directory}" ]] || { echo "Existing non-checkout directory: ${directory}" >&2; return 1; }
    mkdir -p "$(dirname "${directory}")"
    git clone "${url}" "${directory}"
    git -C "${directory}" checkout --detach "${revision}"
  fi
  [[ "$(git -C "${directory}" rev-parse HEAD)" == "${revision}" ]] || {
    echo "${directory} is not at pinned commit ${revision}; existing source was preserved." >&2
    return 1
  }
}
checkout_locked "${TRITON_RISCV_DIR}" "$(lock_value triton_riscv url)" "$(lock_value triton_riscv commit)"
checkout_locked "${TRITON_DIR}" "$(lock_value triton url)" "$(lock_value triton commit)"
[[ "$(git -C "${BUDDY_SOURCE_DIR}/llvm" rev-parse HEAD)" == "$(lock_value llvm commit)" ]] || {
  echo "The LLVM source revision differs from toolchain-lock.json; this compatibility patch has not been verified for that revision." >&2
  exit 1
}

# Check the series as a whole: later upstream patches overlap earlier patches.
patch_args=()
[[ "${check_only}" == 0 ]] || patch_args+=(--check)
"${TRITON_PYTHON}" "${script_dir}/apply-patches.py" \
  "${TRITON_RISCV_DIR}" "${TRITON_DIR}" "${patch_args[@]}"

if [[ "${check_only}" == 0 ]]; then
  if [[ "${skip_deps}" == 0 ]]; then
    "${TRITON_PYTHON}" -m pip install -r "${script_dir}/requirements-build.txt"
  fi
  # The shared sanitizer is unrelated to the NR frontend and uses APIs from
  # another LLVM revision. All TTIR/linalg frontend passes remain enabled.
  export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DTRITON_SHARED_BUILD_TRITON_SAN=OFF"
  export MAX_JOBS="${jobs}"
  export PATH="$("${TRITON_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"
  (
    cd "${TRITON_DIR}"
    "${TRITON_PYTHON}" -m pip install --no-build-isolation -v -e .
  )
  source "${script_dir}/triton-env.sh"
fi

[[ -x "${TRITON_SHARED_OPT_PATH:-}" ]] || { echo "triton-shared-opt is not built; run setup without --check." >&2; exit 1; }
"${TRITON_PYTHON}" "${script_dir}/frontend_smoke.py"
echo "Triton frontend ready: ${TRITON_SHARED_OPT_PATH}"
