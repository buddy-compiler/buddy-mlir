#!/usr/bin/env bash

set -euo pipefail

REPO="${BUDDY_CLI_REPO:-buddy-compiler/buddy-mlir}"
INSTALL_DIR="${BUDDY_CLI_INSTALL_DIR:-$HOME/.local/bin}"
REQUESTED_VERSION="${1:-latest}"

case "$(uname -s)" in
  Linux)
    ;;
  *)
    echo "Only Linux is supported by this installer." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64)
    TARGET_ARCH="x86_64"
    ;;
  riscv64)
    TARGET_ARCH="riscv64"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    echo "Supported: x86_64, riscv64" >&2
    exit 1
    ;;
esac

resolve_tag() {
  if [ "${REQUESTED_VERSION}" = "latest" ]; then
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
      | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' \
      | head -n 1
    return
  fi

  case "${REQUESTED_VERSION}" in
    release/*)
      printf '%s\n' "${REQUESTED_VERSION}"
      ;;
    v*)
      printf 'release/%s\n' "${REQUESTED_VERSION}"
      ;;
    *)
      printf 'release/v%s\n' "${REQUESTED_VERSION}"
      ;;
  esac
}

TAG="$(resolve_tag)"
if [ -z "${TAG}" ]; then
  echo "Failed to resolve release tag for ${REQUESTED_VERSION}." >&2
  exit 1
fi

VERSION="${TAG#release/v}"
ASSET_NAME="buddy-cli-${VERSION}.${TARGET_ARCH}"
ASSET_URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET_NAME}"

mkdir -p "${INSTALL_DIR}"

echo "Downloading ${ASSET_NAME} from ${TAG}..."
curl -fL "${ASSET_URL}" -o "${INSTALL_DIR}/buddy-cli"
chmod 0755 "${INSTALL_DIR}/buddy-cli"

echo "Installed buddy-cli to ${INSTALL_DIR}/buddy-cli"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*)
    ;;
  *)
    echo "Add ${INSTALL_DIR} to PATH to use buddy-cli directly."
    ;;
esac
