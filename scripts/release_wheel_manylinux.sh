#!/usr/bin/env bash
# Build a manylinux_x86_64 wheel inside the official manylinux container.
# This script must be run on a host with Docker available.
#
# Usage:
#   ./scripts/build_manylinux.sh [cp_tag]
# cp_tag defaults to cp310-cp310. Other valid tags are the Python versions
# present under /opt/python in the manylinux image (e.g., cp311-cp311).

set -euo pipefail

IMAGE="quay.io/pypa/manylinux_2_28_x86_64"

PY_TAG="${1:-cp310-cp310}"
TORCH_VERSION="${TORCH_VERSION:-2.8}"
# MLIR version is calculated in setup.py

# Host dir
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Docker dir (default mount)
WORKSPACE=/workspace/buddy-mlir

# Note: outputs are placed under build.docker/ to avoid clashing with host builds.
BUDDY_BUILD_DIR="${WORKSPACE}/build.docker"
LLVM_BUILD_DIR="${WORKSPACE}/llvm/build.docker"

docker run --rm -i \
  -e WORKSPACE="${WORKSPACE}" \
  -e BUDDY_BUILD_DIR="${BUDDY_BUILD_DIR}" \
  -e LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  -e LLVM_CACHE_HIT="${LLVM_CACHE_HIT:-false}" \
  -e CLEAN_BUILD="${CLEAN_BUILD:-0}" \
  -e PY_TAG="${PY_TAG}" \
  -e TORCH_VERSION="${TORCH_VERSION}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e HOME=/workspace \
  -v "${REPO_ROOT}:${WORKSPACE}" \
  -w "${WORKSPACE}" \
  "${IMAGE}" \
  /bin/bash -s <<'BASH'
    set -euo pipefail
    set -x

    # manylinux stores multiple Python versions under /opt/python; PATH does not
    # select a version by default, so we choose explicitly.
    # Docs: https://github.com/pypa/manylinux#docker-images
    PYBIN=/opt/python/${PY_TAG}/bin/python
    if [ ! -x "$PYBIN" ]; then
      echo "Python tag ${PY_TAG} not found under /opt/python" >&2
      ls /opt/python >&2
      exit 1
    fi
    export PATH="/opt/python/${PY_TAG}/bin:$PATH"

    # manylinux images ship newer GCC via gcc-toolset; it is not enabled by default.
    # Docs: https://github.com/pypa/manylinux#manylinux2014-2_28-and-2_34-images
    if [ -f /opt/rh/gcc-toolset-14/enable ]; then
      source /opt/rh/gcc-toolset-14/enable
    fi
    export CC=gcc
    export CXX=g++

    "$PYBIN" -m pip install --upgrade pip build auditwheel ninja cmake numpy pybind11==2.10.* nanobind==2.4.* PyYAML >/dev/null

    # Optional clean rebuild (set CLEAN_BUILD=1 to force)
    if [ "${CLEAN_BUILD:-0}" = "1" ]; then
      rm -rf "${LLVM_BUILD_DIR}" "${BUDDY_BUILD_DIR}"
    fi

    if [ "${LLVM_CACHE_HIT:-false}" = "true" ]; then
      echo "LLVM build cache hit; skipping LLVM build."
    else
      # Build LLVM/MLIR first
      cmake -G Ninja -S "${WORKSPACE}/llvm/llvm" -B "${LLVM_BUILD_DIR}" \
        -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
        -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$PYBIN"
      ninja -C "${LLVM_BUILD_DIR}" check-clang check-mlir omp || true
    fi
    ${LLVM_BUILD_DIR}/bin/mlir-opt --version

    # Build buddy-mlir with Python packages enabled
    cmake -G Ninja -S "${WORKSPACE}" -B "${BUDDY_BUILD_DIR}" \
      -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
      -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
      -DPython3_EXECUTABLE="$PYBIN"
    ninja -C "${BUDDY_BUILD_DIR}"
    ninja -C "${BUDDY_BUILD_DIR}" python-package-buddy python-package-buddy-mlir || true
    ${BUDDY_BUILD_DIR}/bin/buddy-opt --version

    # Optional build tag (must start with a digit). Example: 1pytorch2_2mlir19
    "$PYBIN" -m build --wheel --outdir "${BUDDY_BUILD_DIR}/dist"
    auditwheel repair "${BUDDY_BUILD_DIR}/dist"/buddy-*.whl -w "${BUDDY_BUILD_DIR}/dist"

    # Fix ownership for host user
    chown -R "$HOST_UID":"$HOST_GID" "${BUDDY_BUILD_DIR}" "${LLVM_BUILD_DIR}" || true
BASH

echo "Wheels are in ${REPO_ROOT}/build.docker/dist"
