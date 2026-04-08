#!/usr/bin/env bash
# Build a manylinux wheel inside the official manylinux container.
# This script must be run on a host with Docker available.
#
# Usage:
#   ./scripts/release_wheel_manylinux.sh [cp_tag] [version] [target_arch]

set -euo pipefail

PY_TAG="${1:?Error: Python ABI (parameter 1, format \"cp310-cp310\") is required but not set.}"

HOST_ARCH_RAW="$(uname -m)"
case "${HOST_ARCH_RAW}" in
  x86_64|amd64)
    HOST_ARCH="x86_64"
    ;;
  riscv64)
    HOST_ARCH="riscv64"
    ;;
  *)
    HOST_ARCH="${HOST_ARCH_RAW}"
    ;;
esac

VERSION="${2:?Error: VERSION (parameter 3, format \"0.0.0\") is required but not set.}"
TARGET_ARCH="${3:-${TARGET_ARCH:-${HOST_ARCH}}}"

case "${TARGET_ARCH}" in
  x86_64)
    DEFAULT_MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
    DEFAULT_DOCKER_PLATFORM="linux/amd64"
    DEFAULT_MANYLINUX_TAG="manylinux_2_28_x86_64"
    ;;
  riscv64)
    DEFAULT_MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_39_riscv64"
    DEFAULT_DOCKER_PLATFORM="linux/riscv64"
    DEFAULT_MANYLINUX_TAG="manylinux_2_39_riscv64"
    ;;
  *)
    echo "Unsupported target arch: ${TARGET_ARCH}" >&2
    echo "Supported: x86_64, riscv64" >&2
    exit 1
    ;;
esac

MANYLINUX_IMAGE="${MANYLINUX_IMAGE:-${DEFAULT_MANYLINUX_IMAGE}}"
# Native same-arch host does not need --platform and some daemons reject it.
if [ "${HOST_ARCH}" = "${TARGET_ARCH}" ] && [ -z "${DOCKER_PLATFORM+x}" ]; then
  DOCKER_PLATFORM=""
else
  DOCKER_PLATFORM="${DOCKER_PLATFORM:-${DEFAULT_DOCKER_PLATFORM}}"
fi
MANYLINUX_TAG="${MANYLINUX_TAG:-${DEFAULT_MANYLINUX_TAG}}"
# MLIR version is calculated in setup.py

# Host dir
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Docker dir (default mount)
WORKSPACE=/workspace/buddy-mlir

# Note: build trees are split by arch + python tag to avoid cross-version CMake cache pollution.
BUDDY_BUILD_ROOT="${WORKSPACE}/build-docker/${TARGET_ARCH}"
LLVM_BUILD_ROOT="${WORKSPACE}/llvm/build-docker/${TARGET_ARCH}"
BUDDY_BUILD_DIR="${BUDDY_BUILD_ROOT}/${PY_TAG}"
LLVM_BUILD_DIR="${LLVM_BUILD_ROOT}/${PY_TAG}"

DOCKER_RUN_ARGS=(run --rm -i)
if [ -n "${DOCKER_PLATFORM}" ]; then
  DOCKER_RUN_ARGS+=(--platform "${DOCKER_PLATFORM}")
fi

docker "${DOCKER_RUN_ARGS[@]}" \
  -e WORKSPACE="${WORKSPACE}" \
  -e TARGET_ARCH="${TARGET_ARCH}" \
  -e BUDDY_PACKAGE_VERSION="${VERSION}" \
  -e BUDDY_BUILD_ROOT="${BUDDY_BUILD_ROOT}" \
  -e LLVM_BUILD_ROOT="${LLVM_BUILD_ROOT}" \
  -e BUDDY_BUILD_DIR="${BUDDY_BUILD_DIR}" \
  -e LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  -e LLVM_CACHE_HIT="${LLVM_CACHE_HIT:-false}" \
  -e CLEAN_BUILD="${CLEAN_BUILD:-0}" \
  -e PY_TAG="${PY_TAG}" \
  -e MANYLINUX_TAG="${MANYLINUX_TAG}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e HOME=/workspace \
  -v "${REPO_ROOT}:${WORKSPACE}" \
  -w "${WORKSPACE}" \
  "${MANYLINUX_IMAGE}" \
  /bin/bash -s <<'BASH'
    ARTIFACT_SUFFIX="${PY_TAG}-${MANYLINUX_TAG}"
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

    # Install image deps via dnf.
    dnf install -y libpng-devel libjpeg-turbo-devel zlib-devel

    FLATBUFFERS_VERSION="25.12.19"
    git clone --depth 1 --branch "v${FLATBUFFERS_VERSION}" https://github.com/google/flatbuffers.git /tmp/flatbuffers
    pushd /tmp/flatbuffers
      mkdir build && cd build
      cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_INSTALL=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      make -j$(nproc)
      make install
    popd

    "$PYBIN" -m pip install --upgrade pip build auditwheel ninja cmake numpy pybind11==2.10.* nanobind==2.4.* PyYAML
    if [ "${TARGET_ARCH}" = "riscv64" ]; then
      dnf install -y rust cargo
      "$PYBIN" -m pip install -i https://ruyirepo.ruyicommunity.cn/pypi/simple/ torch
    fi
    "$PYBIN" -m pip install transformers==4.56.2

    # Optional clean rebuild (set CLEAN_BUILD=1 to force)
    if [ "${CLEAN_BUILD:-0}" = "1" ]; then
      rm -rf "${LLVM_BUILD_DIR}" "${BUDDY_BUILD_DIR}"
    fi

    LLVM_STAMP_FILE="${LLVM_BUILD_DIR}/.manylinux-llvm-ready"
    LLVM_INSTALL_DIR="${LLVM_BUILD_DIR}/dist"
    BUDDY_INSTALL_DIR="${BUDDY_BUILD_DIR}/dist"
    ARTIFACT_DIR="${BUDDY_BUILD_DIR}/target"
    mkdir -p "${ARTIFACT_DIR}"

    LLVM_CACHE_READY="${LLVM_CACHE_HIT:-false}"
    if [ "${LLVM_CACHE_READY}" = "true" ] && [ ! -f "${LLVM_STAMP_FILE}" ]; then
      LLVM_CACHE_READY="false"
    fi

    if [ "${LLVM_CACHE_READY}" != "true" ]; then
      cmake -G Ninja -S "${WORKSPACE}/llvm/llvm" -B "${LLVM_BUILD_DIR}" \
        -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
        -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$PYBIN" \
        -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
      # Keep the release flow moving even if upstream check targets are flaky.
      ninja -C "${LLVM_BUILD_DIR}"
      cmake --build "${LLVM_BUILD_DIR}" --target install -j
      LLVM_VERSION_FILE="$(find "${LLVM_INSTALL_DIR}" -path '*/cmake/llvm/LLVMConfigVersion.cmake' | head -n 1)"
      LLVM_PACKAGE_VERSION="$(sed -nE 's/^[[:space:]]*set\(PACKAGE_VERSION[[:space:]]+"([^"]+)".*/\1/p' "${LLVM_VERSION_FILE}" | head -n 1)"
      LLVM_PACKAGE_VERSION="${LLVM_PACKAGE_VERSION:-unknown}"
      LLVM_TAR_NAME="llvm-${LLVM_PACKAGE_VERSION}-${ARTIFACT_SUFFIX}.tar.gz"
      LLVM_TAR_TMP="${BUDDY_BUILD_DIR}/${LLVM_TAR_NAME}"
      tar -C "${LLVM_INSTALL_DIR}" -czf "${LLVM_TAR_TMP}" .
      mv -f "${LLVM_TAR_TMP}" "${ARTIFACT_DIR}/${LLVM_TAR_NAME}"
      printf 'ready\n' > "${LLVM_STAMP_FILE}"
    fi
    # Build buddy-mlir with Python packages enabled
    cmake -G Ninja -S "${WORKSPACE}" -B "${BUDDY_BUILD_DIR}" \
      -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
      -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
      -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
      -DPython3_EXECUTABLE="${PYBIN}" \
      -DBUDDY_PACKAGE_VERSION="${BUDDY_PACKAGE_VERSION}" \
      -DCMAKE_INSTALL_PREFIX="${BUDDY_INSTALL_DIR}"
    ninja -C "${BUDDY_BUILD_DIR}"
    cmake --build "${BUDDY_BUILD_DIR}" --target install -j

    # Make the buddy bundle self-contained, so that user won't have to include llvm bundle
    cp -a "${LLVM_BUILD_DIR}/lib"/libmlir*.so* "${BUDDY_INSTALL_DIR}/lib/"
    cp -a "${LLVM_BUILD_DIR}/lib/libomp.so"* "${BUDDY_INSTALL_DIR}/lib/"

    ${BUDDY_BUILD_DIR}/bin/buddy-opt --version

    # Optional build tag (must start with a digit). Example: 1mlir22
    "$PYBIN" -m build --wheel --outdir "${ARTIFACT_DIR}"
    auditwheel repair "${ARTIFACT_DIR}"/buddy-*-linux_*.whl -w "${ARTIFACT_DIR}"

    BUDDY_TAR_NAME="buddy-${BUDDY_PACKAGE_VERSION}-${ARTIFACT_SUFFIX}.tar.gz"
    BUDDY_TAR_TMP="${BUDDY_BUILD_DIR}/${BUDDY_TAR_NAME}"
    tar -C "${BUDDY_INSTALL_DIR}" -czf "${BUDDY_TAR_TMP}" .
    mv -f "${BUDDY_TAR_TMP}" "${ARTIFACT_DIR}/${BUDDY_TAR_NAME}"

    echo "Artifacts are in ${ARTIFACT_DIR}"
    echo "Python build dirs:"
    echo "  ${BUDDY_BUILD_DIR}"
    echo "  ${LLVM_BUILD_DIR}"

    # Fix ownership for host user
    chown -R "$HOST_UID":"$HOST_GID" "${BUDDY_BUILD_ROOT}" "${LLVM_BUILD_ROOT}" || true
BASH
