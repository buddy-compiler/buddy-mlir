#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLVM_SRC="${1:-${LLVM_SRC:-${REPO_ROOT}/llvm}}"
PATCH_DIR="${LLVM_PATCH_DIR:-${REPO_ROOT}/patches/llvm}"

if [[ ! -d "${LLVM_SRC}/llvm" ]]; then
  echo "error: LLVM source tree not found: ${LLVM_SRC}" >&2
  exit 1
fi

if [[ ! -d "${PATCH_DIR}" ]]; then
  exit 0
fi

shopt -s nullglob
patches=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if (( ${#patches[@]} == 0 )); then
  exit 0
fi

for patch in "${patches[@]}"; do
  echo "Applying LLVM patch: ${patch}"
  if git -C "${LLVM_SRC}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "LLVM patch already applied: ${patch}"
    continue
  fi
  git -C "${LLVM_SRC}" apply --check "${patch}"
  git -C "${LLVM_SRC}" apply "${patch}"
done
