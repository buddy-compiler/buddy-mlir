#!/usr/bin/env bash
# ===- run_llama31_p150_chat.sh -----------------------------------------------
# Full Llama-3.1-8B-Instruct chat on P150A with 1024-token static cache.
# Pipeline:
#   1) lower prefill TTIR (--static-cache --max-cache-len 1024) -> TTNN -> FB
#   2) lower decode  TTIR (same)                                -> TTNN -> FB
#   3) prepare chat artifacts (weights + slot roles) for both phases
#   4) package TTNN artifacts into a llama31_tt .rax manifest
#   5) launch buddy-cli --model llama31_tt.rax by default
#
# Use MAX_CACHE_LEN (default 1024) to control cache length.
# Set SKIP_LOWER=1 / SKIP_PREPARE=1 / SKIP_PACKAGE=1 to rerun later stages
# with existing artifacts. Set RUN_WITH_BUDDY_CLI=0 to call the Python
# ttrt runner directly. Set MAX_NEW_TOKENS=N to cap generation length.
#
# Memory note: Llama-3.1-8B's bf16 weights are ~16 GB; the chat artifacts
# are loaded twice (prefill + decode contexts), so the script raises the
# virtual-memory soft limit to 95 GB before running anything.
# ===---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

ulimit -v 95000000 || true

cd "${SCRIPT_DIR}"
export LLAMA31_MODEL_PATH="${LLAMA31_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-1024}"
SKIP_LOWER="${SKIP_LOWER:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
RUN_WITH_BUDDY_CLI="${RUN_WITH_BUDDY_CLI:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
# Set DEVICE_ARGMAX=1 to emit i64 token-id directly from the device graph
# (saves ~256.5 KB / step on logits.to_host; see TTIR_E2E_OPTIMIZATION_ROADMAP P4).
DEVICE_ARGMAX="${DEVICE_ARGMAX:-0}"
# Set FULL_ARG_ATTRS=0 to use the older unannotated full TTIR path. The
# default emits official-style input/parameter/constant metadata so TTIR->TTNN
# can hoist/cache parameter layout work like the layer/block alignment gates.
FULL_ARG_ATTRS="${FULL_ARG_ATTRS:-1}"
LOWER_EXTRA=()
PREPARE_EXTRA=()
GEN_MANIFEST_EXTRA=()
RUNNER_EXTRA=()
ARGMAX_SUFFIX=""
if [[ "${DEVICE_ARGMAX}" == "1" ]]; then
  LOWER_EXTRA+=(--device-argmax)
  PREPARE_EXTRA+=(--device-argmax)
  GEN_MANIFEST_EXTRA+=(--device-token-loop --ignore-eos)
  RUNNER_EXTRA+=(--device-token-loop --ignore-eos)
  ARGMAX_SUFFIX="_argmax"
fi
ARG_ATTR_SUFFIX=""
if [[ "${FULL_ARG_ATTRS}" == "1" ]]; then
  LOWER_EXTRA+=(--annotate-official-arg-attrs)
  ARG_ATTR_SUFFIX="_argattrs"
fi
ARTIFACT_SUFFIX="${ARG_ATTR_SUFFIX}${ARGMAX_SUFFIX}"
TTIR_OUT="${SCRIPT_DIR}/ttir_out_static"
BASE_CHAT_ART="${BASE_CHAT_ART:-${SCRIPT_DIR}/chat_artifacts}"
if [[ "${DEVICE_ARGMAX}" == "1" ]]; then
  CHAT_ART="${CHAT_ART:-${SCRIPT_DIR}/chat_artifacts_argmax}"
else
  CHAT_ART="${CHAT_ART:-${BASE_CHAT_ART}}"
fi
RAX_PACKAGE_DIR="${RAX_PACKAGE_DIR:-${SCRIPT_DIR}/tt_package${ARGMAX_SUFFIX}}"
mkdir -p "${TTIR_OUT}" "${CHAT_ART}" "${RAX_PACKAGE_DIR}"

have_artifact() {
  [[ -s "$1" ]]
}

reuse_static_payloads() {
  for phase in prefill decode; do
    local dst_dir="${CHAT_ART}/${phase}"
    local src_dir="${BASE_CHAT_ART}/${phase}"
    mkdir -p "${dst_dir}"
    for name in weights.npz inv_freq.npy; do
      local src="${src_dir}/${name}"
      local dst="${dst_dir}/${name}"
      if [[ -e "${dst}" ]]; then
        continue
      fi
      if [[ ! -e "${src}" ]]; then
        if [[ "${name}" == "weights.npz" ]]; then
          echo "error: missing reusable payload ${src}" >&2
          exit 1
        fi
        continue
      fi
      ln "${src}" "${dst}" 2>/dev/null || ln -s "${src}" "${dst}"
    done
  done
}

SYS_DIR="/tmp/ttrt_sys_llama31"
if [[ ! -f "${SYS_DIR}/system_desc.ttsys" ]]; then
  echo "=== query system descriptor ==="
  python -m ttrt query --save-artifacts --artifact-dir "${SYS_DIR}" 2>&1 | tail -3
fi
SYS="${SYS_DIR}/system_desc.ttsys"

# ttmlir-opt fusion flags that work without TTMLIR_ENABLE_OPMODEL=ON.
# Confirmed +15 % steady wall t/s/u and -50 % submit_wait_ms on Llama-3.1-8B
# decode (2026-04-29). Set BUDDY_TTNN_PIPELINE_EXTRA="" to disable.
BUDDY_TTNN_PIPELINE_EXTRA="${BUDDY_TTNN_PIPELINE_EXTRA:- enable-fusing-pass=true enable-permute-matmul-fusion=true enable-erase-inverse-ops-pass=true enable-dram-space-saving-optimization-pass=true enable-implicit-broadcast-folding-pass=true enable-optimization-passes=true}"

PREFILL_TTNN="${TTIR_OUT}/llama31_prefill_static${ARTIFACT_SUFFIX}.ttnn"
DECODE_TTNN="${TTIR_OUT}/llama31_decode_static${ARTIFACT_SUFFIX}.ttnn"
RAX_MANIFEST="${RAX_PACKAGE_DIR}/llama31_tt.rhal.mlir"
RAX_FILE="${RAX_PACKAGE_DIR}/llama31_tt.rax"

if [[ "${SKIP_LOWER}" != "1" ]]; then
  if ! have_artifact "${PREFILL_TTNN}"; then
    echo "=== [1/4] Lower prefill TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-llama31-lower-ttir.py \
      --mode prefill --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --element-dtype bf16 \
      "${LOWER_EXTRA[@]}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [1b/4] prefill TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/llama31_ttir_prefill.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}${BUDDY_TTNN_PIPELINE_EXTRA}" \
      -o "${TTIR_OUT}/llama31_prefill_static${ARTIFACT_SUFFIX}_ttnn.mlir"
    echo "=== [1c/4] prefill TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/llama31_prefill_static${ARTIFACT_SUFFIX}_ttnn.mlir" \
      -o "${PREFILL_TTNN}"
  else
    echo "=== [1/4] prefill flatbuffer already exists: ${PREFILL_TTNN} ==="
  fi

  if ! have_artifact "${DECODE_TTNN}"; then
    echo "=== [2/4] Lower decode TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-llama31-lower-ttir.py \
      --mode decode --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --element-dtype bf16 \
      "${LOWER_EXTRA[@]}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [2b/4] decode TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/llama31_ttir_decode.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}${BUDDY_TTNN_PIPELINE_EXTRA}" \
      -o "${TTIR_OUT}/llama31_decode_static${ARTIFACT_SUFFIX}_ttnn.mlir"
    echo "=== [2c/4] decode TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/llama31_decode_static${ARTIFACT_SUFFIX}_ttnn.mlir" \
      -o "${DECODE_TTNN}"
  else
    echo "=== [2/4] decode flatbuffer already exists: ${DECODE_TTNN} ==="
  fi
else
  echo "=== skipping lower/ttnn/flatbuffer (SKIP_LOWER=1) ==="
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if [[ ! -f "${CHAT_ART}/prefill/slot_roles.json" \
        || ! -f "${CHAT_ART}/decode/slot_roles.json" ]]; then
    echo "=== [3/4] Prepare chat artifacts (weights + roles) ==="
    if [[ "${DEVICE_ARGMAX}" == "1" && "${BASE_CHAT_ART}" != "${CHAT_ART}" \
          && -f "${BASE_CHAT_ART}/prefill/weights.npz" \
          && -f "${BASE_CHAT_ART}/decode/weights.npz" ]]; then
      PREPARE_EXTRA+=(--metadata-only)
    fi
    python llama31_chat_prepare.py \
      --max-cache-len "${MAX_CACHE_LEN}" \
      "${PREPARE_EXTRA[@]}" \
      -o "${CHAT_ART}"
    if [[ "${DEVICE_ARGMAX}" == "1" && "${BASE_CHAT_ART}" != "${CHAT_ART}" ]]; then
      reuse_static_payloads
    fi
  else
    echo "=== [3/4] chat artifacts already exist: ${CHAT_ART} ==="
    if [[ "${DEVICE_ARGMAX}" == "1" && "${BASE_CHAT_ART}" != "${CHAT_ART}" ]]; then
      reuse_static_payloads
    fi
  fi
else
  echo "=== skipping chat prepare (SKIP_PREPARE=1) ==="
  if [[ "${DEVICE_ARGMAX}" == "1" && "${BASE_CHAT_ART}" != "${CHAT_ART}" ]]; then
    reuse_static_payloads
  fi
fi

if [[ "${SKIP_PACKAGE}" != "1" ]]; then
  echo "=== [4/5] Package TTNN artifacts for buddy-cli ==="
  if ! have_artifact "${PREFILL_TTNN}" || ! have_artifact "${DECODE_TTNN}"; then
    echo "error: missing non-empty TTNN flatbuffers; rerun without SKIP_LOWER=1" >&2
    exit 1
  fi
  python "${REPO_ROOT}/tools/buddy-codegen/gen_tenstorrent_manifest.py" \
    --prefill-ttnn "${PREFILL_TTNN}" \
    --decode-ttnn "${DECODE_TTNN}" \
    --artifacts "${CHAT_ART}" \
    --runner "${SCRIPT_DIR}/llama31_chat_run.py" \
    --tokenizer "${LLAMA31_MODEL_PATH}" \
    --max-cache-len "${MAX_CACHE_LEN}" \
    "${GEN_MANIFEST_EXTRA[@]}" \
    -o "${RAX_MANIFEST}"

  RAX_PACK_ARGS=()
  if [[ "${EMBED_TTNN_IN_RAX:-0}" == "1" ]]; then
    RAX_PACK_ARGS+=(--embed-payload)
  fi
  "${BUDDY_BUILD}/bin/rax-pack" "${RAX_MANIFEST}" -o "${RAX_FILE}" "${RAX_PACK_ARGS[@]}"
else
  echo "=== skipping RAX package (SKIP_PACKAGE=1): ${RAX_FILE} ==="
fi

echo "=== [5/5] Interactive chat on P150A ==="
EXTRA_ARGS=()
if [[ "${MAX_NEW_TOKENS}" != "0" ]]; then
  EXTRA_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${RUN_WITH_BUDDY_CLI}" == "1" ]]; then
  CLI_ARGS=(--model "${RAX_FILE}" --max-tokens "${MAX_NEW_TOKENS}")
  BUDDY_TT_PYTHON="${BUDDY_TT_PYTHON:-$(command -v python)}" \
    "${BUDDY_BUILD}/bin/buddy-cli" "${CLI_ARGS[@]}"
else
  python llama31_chat_run.py \
    --prefill-ttnn "${PREFILL_TTNN}" \
    --decode-ttnn "${DECODE_TTNN}" \
    --artifacts "${CHAT_ART}" \
    --max-cache-len "${MAX_CACHE_LEN}" \
    --ignore-system-desc \
    "${RUNNER_EXTRA[@]}" \
    "${EXTRA_ARGS[@]}"
fi
