#!/usr/bin/env bash
# Build a manylinux wheel inside the official manylinux container.
# This script must be run on a host with Docker available.
#
# Usage:
#   ./scripts/release_wheel_manylinux.sh [cp_tag] [version] [target_arch]

set -euo pipefail

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

PY_TAG="${1:?Error: Python ABI (parameter 1, format \"cp310-cp310\") is required but not set.}"
VERSION="${2:?Error: VERSION (parameter 2, format \"0.0.0\") is required but not set.}"
TARGET_ARCH="${3:?Error: TARGET_ARCH (parameter 3, format \"x86_64|riscv64\") is required but not set.}"

# -----------------------------------------------------------------------------
# Host paths and cache keys
# -----------------------------------------------------------------------------

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOST_BUILD_DIR="${HOST_BUILD_DIR:-$REPO_ROOT/build-docker}"
HOST_LLVM_SRC="${HOST_LLVM_SRC:-$REPO_ROOT/llvm}"
if [ -z "${LLVM_HASH:-}" ]; then
  LLVM_COMMIT="$(git -C "${REPO_ROOT}" ls-tree HEAD llvm | awk '{print $3}')"
  PATCH_HASH="$(sha256sum "${REPO_ROOT}/riscv-jitlink.patch" | awk '{print $1}')"
  LLVM_HASH="${LLVM_COMMIT}-${PATCH_HASH:0:12}"
fi

# -----------------------------------------------------------------------------
# Container paths
# -----------------------------------------------------------------------------

WORKSPACE=/workspace/buddy-mlir
WORKSPACE_LLVM_SRC=/workspace/llvm
WORKSPACE_BUILD_DIR=/workspace/build-docker

# -----------------------------------------------------------------------------
# Target-specific defaults
# -----------------------------------------------------------------------------

case "${TARGET_ARCH}" in
  x86_64)
    DEFAULT_MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
    DEFAULT_DOCKER_PLATFORM="linux/amd64"
    DEFAULT_MANYLINUX_TAG="manylinux_2_28_x86_64"
    LLVM_RUNTIME_LIBDIR="x86_64-unknown-linux-gnu"
    ;;
  riscv64)
    DEFAULT_MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_39_riscv64"
    DEFAULT_DOCKER_PLATFORM="linux/riscv64"
    DEFAULT_MANYLINUX_TAG="manylinux_2_39_riscv64"
    LLVM_RUNTIME_LIBDIR="riscv64-unknown-linux-gnu"
    ;;
  *)
    echo "Unsupported target arch: ${TARGET_ARCH}" >&2
    echo "Supported: x86_64, riscv64" >&2
    exit 1
    ;;
esac

MANYLINUX_IMAGE="${MANYLINUX_IMAGE:-${DEFAULT_MANYLINUX_IMAGE}}"

# Native same-arch host does not need --platform and some daemons reject it.
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

MANYLINUX_TAG="${MANYLINUX_TAG:-${DEFAULT_MANYLINUX_TAG}}"

is_in_container() {
    [ -f /.dockerenv ] || grep -q 'docker' /proc/1/cgroup 2>/dev/null
}

if ! is_in_container; then
  # ---------------------------------------------------------------------------
  # Host side: validate mounts and re-enter inside the manylinux container
  # ---------------------------------------------------------------------------

  BUDDY_HASH="${BUDDY_HASH:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
  DOCKER_RUN_ARGS=(run --rm -i)
  DOCKER_ENV_ARGS=()
  for proxy_var in http_proxy https_proxy ftp_proxy no_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY ALL_PROXY all_proxy; do
    if [ -n "${!proxy_var:-}" ]; then
      DOCKER_ENV_ARGS+=(-e "${proxy_var}=${!proxy_var}")
    fi
  done
  if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_RUN_ARGS+=(-t)
  fi
  if [ "${HOST_ARCH}" = "${TARGET_ARCH}" ] && [ -z "${DOCKER_PLATFORM+x}" ]; then
    DOCKER_PLATFORM=""
  else
    DOCKER_PLATFORM="${DOCKER_PLATFORM:-${DEFAULT_DOCKER_PLATFORM}}"
  fi
  if [ -n "${DOCKER_PLATFORM}" ]; then
    DOCKER_RUN_ARGS+=(--platform "${DOCKER_PLATFORM}")
  fi
  if [ ! -f "${HOST_LLVM_SRC}/llvm/CMakeLists.txt" ]; then
    echo "HOST_LLVM_SRC is invalid: ${HOST_LLVM_SRC}" >&2
    ls -ld "${HOST_LLVM_SRC}" "${HOST_LLVM_SRC}/llvm" 2>/dev/null || true
    exit 1
  fi
  docker "${DOCKER_RUN_ARGS[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    -e WORKSPACE="${WORKSPACE}" \
    -e TARGET_ARCH="${TARGET_ARCH}" \
    -e PY_TAG="${PY_TAG}" \
    -e BUDDY_PACKAGE_VERSION="${VERSION}" \
    -e BUDDY_HASH="${BUDDY_HASH}" \
    -e LLVM_CACHE_HIT="${LLVM_CACHE_HIT:-false}" \
    -e LLVM_HASH="${LLVM_HASH}" \
    -e MANYLINUX_TAG="${MANYLINUX_TAG}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e HOME=/workspace \
    -v "${REPO_ROOT}:${WORKSPACE}" \
    -w "${WORKSPACE}" \
    -v "${HOST_LLVM_SRC}:${WORKSPACE_LLVM_SRC}:ro" \
    -v "${HOST_BUILD_DIR}:${WORKSPACE_BUILD_DIR}" \
    "${MANYLINUX_IMAGE}" \
    /bin/bash $0 $@
else
  # ---------------------------------------------------------------------------
  # Container side: build layout and exit cleanup
  # ---------------------------------------------------------------------------

  ARTIFACT_SUFFIX="${PY_TAG}-${MANYLINUX_TAG}"

  cleanup_workspace_staging() {
    [ -e "${WORKSPACE}" ] && chown -R "$HOST_UID":"$HOST_GID" "${WORKSPACE}"
    [ -e "${BUDDY_BUILD_ROOT}" ] && chown -R "$HOST_UID":"$HOST_GID" "${BUDDY_BUILD_ROOT}"
    [ -e "${LLVM_BUILD_ROOT}" ] && chown -R "$HOST_UID":"$HOST_GID" "${LLVM_BUILD_ROOT}"
  }
  trap cleanup_workspace_staging EXIT

  # Note: build trees are split by arch + python tag + source hash to avoid stale CMake cache reuse.
  export BUDDY_BUILD_ROOT="${WORKSPACE_BUILD_DIR}/${TARGET_ARCH}"
  export LLVM_BUILD_ROOT="${WORKSPACE_BUILD_DIR}/${TARGET_ARCH}"
  export BUDDY_BUILD_DIR="${BUDDY_BUILD_ROOT}/buddy/${PY_TAG}/${BUDDY_HASH}"
  export LLVM_BUILD_DIR="${LLVM_BUILD_ROOT}/llvm/${PY_TAG}/${LLVM_HASH}"
  export PYTHONPYCACHEPREFIX="${LLVM_BUILD_DIR}/.pycache"
  mkdir -p "${PYTHONPYCACHEPREFIX}"

  set -euo pipefail
  set -x

  # ---------------------------------------------------------------------------
  # Python toolchain and compiler setup
  # ---------------------------------------------------------------------------

  # manylinux stores multiple Python versions under /opt/python; PATH does not
  # select a version by default, so we choose explicitly.
  # Docs: https://github.com/pypa/manylinux#docker-images
  PYBIN=/opt/python/${PY_TAG}/bin/python
  if [ ! -x "$PYBIN" ]; then
    echo "Python tag ${PY_TAG} not found under /opt/python" >&2
    ls /opt/python >&2
    exit 1
  fi
  if [ ! -f "${WORKSPACE_LLVM_SRC}/llvm/CMakeLists.txt" ]; then
    echo "Container LLVM source is invalid: ${WORKSPACE_LLVM_SRC}" >&2
    ls -ld "${WORKSPACE_LLVM_SRC}" "${WORKSPACE_LLVM_SRC}/llvm" 2>/dev/null || true
    exit 1
  fi
  export PATH="/opt/python/${PY_TAG}/bin:$PATH"

  # x86_64 image uses gcc-toolset, riscv64 image uses system GCC under /usr.
  GCC_TOOLCHAIN_ROOT=""
  GCC_INSTALL_DIR=""
  LLVM_RUNTIMES_CMAKE_ARGS_VALUE=""
  case "${TARGET_ARCH}" in
    x86_64)
      source /opt/rh/gcc-toolset-14/enable
      GCC_TOOLCHAIN_ROOT="/opt/rh/gcc-toolset-14/root/usr"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="-DCMAKE_C_FLAGS=--gcc-toolchain=${GCC_TOOLCHAIN_ROOT};-DCMAKE_CXX_FLAGS=--gcc-toolchain=${GCC_TOOLCHAIN_ROOT}"
      ;;
    riscv64)
      GCC_INSTALL_DIR="$(dirname "$(g++ -print-libgcc-file-name)")"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="-DCMAKE_C_FLAGS=--gcc-install-dir=${GCC_INSTALL_DIR}"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="${LLVM_RUNTIMES_CMAKE_ARGS_VALUE};-DCMAKE_CXX_FLAGS=--gcc-install-dir=${GCC_INSTALL_DIR}"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="${LLVM_RUNTIMES_CMAKE_ARGS_VALUE};-DCMAKE_HAVE_LIBC_PTHREAD=TRUE"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="${LLVM_RUNTIMES_CMAKE_ARGS_VALUE};-DCMAKE_USE_PTHREADS_INIT=TRUE"
      LLVM_RUNTIMES_CMAKE_ARGS_VALUE="${LLVM_RUNTIMES_CMAKE_ARGS_VALUE};-DCMAKE_THREAD_LIBS_INIT="
      ;;
  esac

  # ---------------------------------------------------------------------------
  # System and Python dependencies inside the manylinux image
  # ---------------------------------------------------------------------------

  # Install image deps via dnf.
  dnf install -y cmake libpng-devel libjpeg-turbo-devel zlib-devel

  # Prepare ccache if support
  if dnf install -y ccache; then
    export CCACHE_BIN="/usr/bin/ccache"
    CCACHE_LINK_DIR="/usr/lib64/ccache"
    mkdir -p "$CCACHE_LINK_DIR"
    CCACHE_COMPILERS=(gcc g++ cc c++)
    case "${TARGET_ARCH}" in
      x86_64)
        CCACHE_COMPILERS+=(x86_64-redhat-linux-gcc x86_64-redhat-linux-g++)
        ;;
      riscv64)
        CCACHE_COMPILERS+=(riscv64-redhat-linux-gcc riscv64-redhat-linux-g++)
        ;;
    esac
    for cmd in "${CCACHE_COMPILERS[@]}"; do
      ln -sf "$CCACHE_BIN" "$CCACHE_LINK_DIR/$cmd"
    done
    export PATH="$CCACHE_LINK_DIR:$PATH"
    export CCACHE_DIR="${WORKSPACE_BUILD_DIR}/.ccache"
    ccache -M 50G
  fi

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

  "$PYBIN" -m pip install --upgrade pip build auditwheel ninja numpy pybind11==2.10.* nanobind==2.4.* PyYAML
  if [ "${TARGET_ARCH}" = "riscv64" ]; then
    dnf install -y rust cargo
    "$PYBIN" -m pip install -i https://ruyirepo.ruyicommunity.cn/pypi/simple/ torch
  fi
  "$PYBIN" -m pip install transformers==4.56.2

  LLVM_STAMP_FILE="${LLVM_BUILD_DIR}/.manylinux-llvm-ready"
  LLVM_INSTALL_DIR="${LLVM_BUILD_DIR}/dist"
  BUDDY_INSTALL_DIR="${BUDDY_BUILD_DIR}/dist"
  ARTIFACT_DIR="${BUDDY_BUILD_DIR}/target"
  mkdir -p "${ARTIFACT_DIR}"

  # ---------------------------------------------------------------------------
  # LLVM configure/build/install
  # ---------------------------------------------------------------------------

  LLVM_CACHE_READY="${LLVM_CACHE_HIT:-false}"
  if [ "${LLVM_CACHE_READY}" = "true" ] && [ ! -f "${LLVM_STAMP_FILE}" ]; then
    LLVM_CACHE_READY="false"
  fi

  if [ "${LLVM_CACHE_READY}" != "true" ]; then
    LLVM_RUNTIMES_CMAKE_ARGS=()
    if [ -n "${LLVM_RUNTIMES_CMAKE_ARGS_VALUE}" ]; then
      LLVM_RUNTIMES_CMAKE_ARGS+=("-DRUNTIMES_CMAKE_ARGS=${LLVM_RUNTIMES_CMAKE_ARGS_VALUE}")
    fi
    rm -rf "${LLVM_BUILD_DIR}"
    cmake -G Ninja -S "${WORKSPACE_LLVM_SRC}/llvm" -B "${LLVM_BUILD_DIR}" \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DLLVM_ENABLE_RUNTIMES="openmp" \
      "${LLVM_RUNTIMES_CMAKE_ARGS[@]}" \
      -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE="$PYBIN" \
      -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
    ccache -z || true
    ninja -C "${LLVM_BUILD_DIR}" check-clang check-mlir check-openmp || true
    ccache -s || true
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

  KEEP=3
  LLVM_CACHE_DIR="${LLVM_BUILD_ROOT}/llvm/${PY_TAG}"
  if [ -d "${LLVM_CACHE_DIR}" ]; then
    (
      cd "${LLVM_CACHE_DIR}"
      ls -1dt */ | tail -n +$((KEEP+1)) | xargs -r rm -rf
    )
  fi

  # ---------------------------------------------------------------------------
  # buddy-mlir configure/build/install/package
  # ---------------------------------------------------------------------------

  # Build buddy-mlir with Python packages enabled.
  # DeepSeek R1 stays OFF: gen_config.py needs HF config/transformers; wheel images do not provide them.
  cmake -G Ninja -S "${WORKSPACE}" -B "${BUDDY_BUILD_DIR}" \
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
    -DLLVM_MAIN_SRC_DIR="${WORKSPACE_LLVM_SRC}/llvm" \
    -DMLIR_MAIN_SRC_DIR="${WORKSPACE_LLVM_SRC}/mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUDDY_ENABLE_TESTS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
    -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=OFF \
    -DPython3_EXECUTABLE="${PYBIN}" \
    -DBUDDY_PACKAGE_VERSION="${BUDDY_PACKAGE_VERSION}" \
    -DCMAKE_INSTALL_PREFIX="${BUDDY_INSTALL_DIR}"
  ccache -z || true
  ninja -C "${BUDDY_BUILD_DIR}"
  ccache -s || true
  cmake --build "${BUDDY_BUILD_DIR}" --target install -j

  # Make the buddy bundle self-contained, so that user won't have to include llvm bundle.
  cp -a "${LLVM_INSTALL_DIR}/lib"/libmlir*.so* "${BUDDY_INSTALL_DIR}/lib/"
  cp -a "${LLVM_INSTALL_DIR}/lib/${LLVM_RUNTIME_LIBDIR}"/libomp*.so* "${BUDDY_INSTALL_DIR}/lib/"

  ${BUDDY_BUILD_DIR}/bin/buddy-opt --version

  export PYTHONPATH="${BUDDY_BUILD_DIR}/python_packages${PYTHONPATH:+:${PYTHONPATH}}"
  "$PYBIN" -m build --wheel --outdir "${ARTIFACT_DIR}" $PWD
  auditwheel repair "${ARTIFACT_DIR}"/buddy-${BUDDY_PACKAGE_VERSION}-${PY_TAG}-linux_*.whl -w "${ARTIFACT_DIR}"

  BUDDY_TAR_NAME="buddy-${BUDDY_PACKAGE_VERSION}-${ARTIFACT_SUFFIX}.tar.gz"
  BUDDY_TAR_TMP="${BUDDY_BUILD_DIR}/${BUDDY_TAR_NAME}"
  tar -C "${BUDDY_INSTALL_DIR}" -czf "${BUDDY_TAR_TMP}" .
  mv -f "${BUDDY_TAR_TMP}" "${ARTIFACT_DIR}/${BUDDY_TAR_NAME}"

  # ---------------------------------------------------------------------------
  # Cache pruning and final summary
  # ---------------------------------------------------------------------------

  BUDDY_CACHE_DIR="${BUDDY_BUILD_ROOT}/buddy/${PY_TAG}"
  if [ -d "${BUDDY_CACHE_DIR}" ]; then
    (
      cd "${BUDDY_CACHE_DIR}"
      ls -1dt */ | tail -n +$((KEEP+1)) | xargs -r rm -rf
    )
  fi

  echo "Artifacts are in ${ARTIFACT_DIR}"
  echo "Python build dirs:"
  echo "  ${BUDDY_BUILD_DIR}"
  echo "  ${LLVM_BUILD_DIR}"
fi
