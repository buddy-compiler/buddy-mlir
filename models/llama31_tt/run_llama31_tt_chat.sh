#!/usr/bin/env bash
# ===- run_llama31_tt_chat.sh --------------------------------------------------
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
# ===---------------------------------------------------------------------------
#
# Full Llama-3.1-8B-Instruct chat on Tenstorrent with 1024-token static cache.
# Pipeline:
#   1) lower prefill TTIR (--static-cache --max-cache-len 1024) -> TTNN -> FB
#   2) lower decode  TTIR (same)                                -> TTNN -> FB
#   3) prepare chat artifacts (weights + slot roles) for both phases
#   4) package TTNN artifacts + chat artifacts into a self-contained .rax
#   5) launch buddy-cli --model llama31_tt.rax by default, unless
#      PACKAGE_ONLY=1 is set.
#
# Use MAX_CACHE_LEN (default 1024) to control cache length.
# Set SKIP_LOWER=1 / SKIP_PREPARE=1 / SKIP_PACKAGE=1 to rerun later stages
# with existing artifacts. Set RUN_WITH_BUDDY_CLI=0 to call the Python
# ttrt runner directly. Set MAX_NEW_TOKENS=N to cap generation length.
# Set PACKAGE_ONLY=1 to stop after .rax generation. The default package embeds
# the TTNN flatbuffers and Llama weights into the .rax payload.
#
# Memory note: Llama-3.1-8B's bf16 weights are ~16 GB; the chat artifacts
# are loaded twice (prefill + decode contexts), so the script raises the
# virtual-memory soft limit to 95 GB before running anything.
#
# ===---------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

ulimit -v 95000000 || true

cd "${SCRIPT_DIR}"
LLAMA_DEFAULT_MODEL="${LLAMA_DEFAULT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LLAMA_DISPLAY_NAME="${LLAMA_DISPLAY_NAME:-Llama-3.1-8B-Instruct}"
if [[ -n "${LLAMA_MODEL_PATH:-}" ]]; then
  export LLAMA31_MODEL_PATH="${LLAMA_MODEL_PATH}"
elif [[ "${LLAMA_USE_LLAMA32_ENV:-0}" == "1" ]]; then
  export LLAMA31_MODEL_PATH="${LLAMA32_MODEL_PATH:-${LLAMA_DEFAULT_MODEL}}"
else
  export LLAMA31_MODEL_PATH="${LLAMA31_MODEL_PATH:-${LLAMA_DEFAULT_MODEL}}"
fi
MAX_CACHE_LEN="${MAX_CACHE_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SKIP_LOWER="${SKIP_LOWER:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
RUN_WITH_BUDDY_CLI="${RUN_WITH_BUDDY_CLI:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
PACKAGE_ONLY="${PACKAGE_ONLY:-0}"
# Generated artifacts default to ${BUDDY_BUILD}/models/llama31_tt. Override
# BUDDY_LLAMA31_ARTIFACT_ROOT, TTIR_OUT, CHAT_ART, RAX_PACKAGE_DIR, or
# TTRT_SYS_DIR to place individual outputs elsewhere.
ARTIFACT_ROOT="${BUDDY_LLAMA31_ARTIFACT_ROOT:-${BUDDY_BUILD}/models/llama31_tt}"
if [[ "${ARTIFACT_ROOT}" == "${BUDDY_BUILD}/llama31_tt" ]]; then
  echo "error: BUDDY_LLAMA31_ARTIFACT_ROOT points to the old layout:" >&2
  echo "  ${ARTIFACT_ROOT}" >&2
  echo "Use ${BUDDY_BUILD}/models/llama31_tt or leave BUDDY_LLAMA31_ARTIFACT_ROOT unset." >&2
  exit 1
fi
# The canonical package emits i64 token ids directly from the device graph.
# This removes the host logits readback from the normal decode loop.
# Set FULL_ARG_ATTRS=0 to use the older unannotated full TTIR path. The
# default emits official-style input/parameter/constant metadata so TTIR->TTNN
# can hoist/cache parameter layout work like the layer/block alignment gates.
FULL_ARG_ATTRS="${FULL_ARG_ATTRS:-1}"
DEVICE_ARGMAX="${DEVICE_ARGMAX:-1}"
DEVICE_ARGMAX_PREFILL="${DEVICE_ARGMAX_PREFILL:-${DEVICE_ARGMAX}}"
DEVICE_ARGMAX_DECODE="${DEVICE_ARGMAX_DECODE:-${DEVICE_ARGMAX}}"
RUNTIME_ATTENTION_MASK="${RUNTIME_ATTENTION_MASK:-0}"
IGNORE_EOS="${IGNORE_EOS:-${BUDDY_LLAMA31_IGNORE_EOS:-0}}"
BUDDY_TT_DECODE_RMSNORM_FUSION="${BUDDY_TT_DECODE_RMSNORM_FUSION:-0}"
BUDDY_TT_DECODE_PRECOMPUTE_SDPA_MASK="${BUDDY_TT_DECODE_PRECOMPUTE_SDPA_MASK:-0}"
BUDDY_TT_DECODE_PRECOMPUTE_ROPE="${BUDDY_TT_DECODE_PRECOMPUTE_ROPE:-0}"
BUDDY_TT_DECODE_MERGE_CACHE_POSITION_INPUTS="${BUDDY_TT_DECODE_MERGE_CACHE_POSITION_INPUTS:-0}"
BUDDY_TT_DECODE_KEEP_CACHE_POSITION_INPUTS="${BUDDY_TT_DECODE_KEEP_CACHE_POSITION_INPUTS:-0}"
BUDDY_TT_DECODE_FOLD_IDENTITY_MUL="${BUDDY_TT_DECODE_FOLD_IDENTITY_MUL:-0}"
BUDDY_TT_DECODE_PACK_QKV="${BUDDY_TT_DECODE_PACK_QKV:-0}"
BUDDY_TT_DECODE_PACK_MLP_GATE_UP="${BUDDY_TT_DECODE_PACK_MLP_GATE_UP:-0}"
BUDDY_TT_DECODE_ARGMAX_TILE="${BUDDY_TT_DECODE_ARGMAX_TILE:-0}"
BUDDY_TT_DECODE_LMHEAD_HIFI2="${BUDDY_TT_DECODE_LMHEAD_HIFI2:-0}"
BUDDY_TT_DECODE_SPLIT_LM_HEAD="${BUDDY_TT_DECODE_SPLIT_LM_HEAD:-0}"
BUDDY_TT_DECODE_LM_HEAD_SPLITS="${BUDDY_TT_DECODE_LM_HEAD_SPLITS:-8}"
BUDDY_TT_DECODE_LM_HEAD_DRAM_PC="${BUDDY_TT_DECODE_LM_HEAD_DRAM_PC:-0}"
BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC="${BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC:-0}"
BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES="${BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES:-48}"
BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT="${BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT:-0}"
BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS="${BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS:-0}"
BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO="${BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO:-0}"
BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS="${BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS:-0}"
BUDDY_TT_DECODE_FUSE_CONCAT_HEADS="${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS:-0}"
BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT="${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT:-0}"
LOWER_COMMON_EXTRA=()
LOWER_PREFILL_EXTRA=()
LOWER_DECODE_EXTRA=()
PREPARE_PREFILL_EXTRA=()
PREPARE_DECODE_EXTRA=()
GEN_MANIFEST_EXTRA=()
DECODE_LOWER_ENV=()
if [[ "${DEVICE_ARGMAX_PREFILL}" == "1" ]]; then
  LOWER_PREFILL_EXTRA+=(--device-argmax)
  PREPARE_PREFILL_EXTRA+=(--device-argmax)
fi
if [[ "${DEVICE_ARGMAX_DECODE}" == "1" ]]; then
  LOWER_DECODE_EXTRA+=(--device-argmax)
  PREPARE_DECODE_EXTRA+=(--device-argmax)
fi
GEN_MANIFEST_EXTRA+=(--device-token-loop)
if [[ "${IGNORE_EOS}" == "1" ]]; then
  GEN_MANIFEST_EXTRA+=(--ignore-eos)
fi
if [[ "${BUDDY_LLAMA31_DISABLE_STATIC_REUSE:-0}" == "1" ]]; then
  GEN_MANIFEST_EXTRA+=(--disable-static-reuse)
fi
ARG_ATTR_SUFFIX=""
if [[ "${FULL_ARG_ATTRS}" == "1" ]]; then
  LOWER_COMMON_EXTRA+=(--annotate-official-arg-attrs)
  ARG_ATTR_SUFFIX="_argattrs"
fi
ARTIFACT_SUFFIX="${ARG_ATTR_SUFFIX}_cachefp32"
if [[ "${RUNTIME_ATTENTION_MASK}" == "1" ]]; then
  LOWER_COMMON_EXTRA+=(--runtime-attention-mask)
  PREPARE_PREFILL_EXTRA+=(--runtime-attention-mask)
  PREPARE_DECODE_EXTRA+=(--runtime-attention-mask)
  ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX}_attnmask"
  export BUDDY_TTIR_MASK_AS_WHERE="${BUDDY_TTIR_MASK_AS_WHERE:-1}"
  export BUDDY_TTIR_BYPASS_BOOL_IDENTITY_INDEX="${BUDDY_TTIR_BYPASS_BOOL_IDENTITY_INDEX:-1}"
fi
if [[ "${DEVICE_ARGMAX_PREFILL}" != "${DEVICE_ARGMAX_DECODE}" ]]; then
  if [[ "${DEVICE_ARGMAX_PREFILL}" == "0" && "${DEVICE_ARGMAX_DECODE}" == "1" ]]; then
    ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX}_decodeargmax"
  elif [[ "${DEVICE_ARGMAX_PREFILL}" == "1" && "${DEVICE_ARGMAX_DECODE}" == "0" ]]; then
    ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX}_prefillargmax"
  fi
fi
if [[ "${BUDDY_TT_DECODE_RMSNORM_FUSION}" == "1" ]]; then
  DECODE_LOWER_ENV+=(
    "BUDDY_TTIR_SKIP_RMSNORM_BF16_SCALAR_CAST=1"
    "BUDDY_TTIR_PRESERVE_F32_ADD=1"
  )
fi
TTIR_OUT="${TTIR_OUT:-${ARTIFACT_ROOT}/ttir_out_static}"
BASE_CHAT_ART="${BASE_CHAT_ART:-${ARTIFACT_ROOT}/chat_artifacts}"
CHAT_ART_SUFFIX=""
if [[ "${BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT}" == "1" ]]; then
  CHAT_ART_SUFFIX="${CHAT_ART_SUFFIX}_splitembed"
fi
CHAT_ART="${CHAT_ART:-${BASE_CHAT_ART}${CHAT_ART_SUFFIX}}"
if [[ -n "${CHAT_ART_SUFFIX}" && "${CHAT_ART}" == "${BASE_CHAT_ART}" ]]; then
  CHAT_ART="${CHAT_ART}${CHAT_ART_SUFFIX}"
fi
RAX_PACKAGE_DIR="${RAX_PACKAGE_DIR:-${ARTIFACT_ROOT}}"
RAX_STEM="${RAX_STEM:-llama31_tt}"
RAX_MODEL_NAME="${RAX_MODEL_NAME:-${RAX_STEM}}"
TTIR_STEM="${TTIR_STEM:-llama31}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
PREFILL_KV_OUTPUT_ORDER="${PREFILL_KV_OUTPUT_ORDER:-key_value}"
mkdir -p "${TTIR_OUT}" "${CHAT_ART}" "${RAX_PACKAGE_DIR}"

have_artifact() {
  [[ -s "$1" ]]
}

reuse_static_payloads() {
  for phase in prefill decode; do
    local dst_dir="${CHAT_ART}/${phase}"
    local src_dir="${BASE_CHAT_ART}/${phase}"
    mkdir -p "${dst_dir}"
    for name in weights.bin inv_freq.npy; do
      local src="${src_dir}/${name}"
      local dst="${dst_dir}/${name}"
      if [[ -e "${dst}" ]]; then
        continue
      fi
      if [[ ! -e "${src}" ]]; then
        if [[ "${name}" == "weights.bin" ]]; then
          echo "error: missing reusable payload ${src}" >&2
          exit 1
        fi
        continue
      fi
      ln "${src}" "${dst}" 2>/dev/null || ln -s "${src}" "${dst}"
    done
  done
}

have_phase_metadata() {
  local phase_dir="${CHAT_ART}/$1"
  local roles="${phase_dir}/slot_roles.json"
  local summary="${phase_dir}/summary.json"
  [[ -f "${roles}" \
    && -f "${phase_dir}/shapes.json" \
    && -f "${phase_dir}/dtypes.json" \
    && -f "${summary}" ]] || return 1
  grep -q "\"max_cache_len\": ${MAX_CACHE_LEN}" "${summary}" || return 1
  grep -q "\"batch\": ${BATCH_SIZE}" "${summary}" || return 1
  if [[ "${RUNTIME_ATTENTION_MASK}" == "1" ]]; then
    grep -q '"role": "attention_mask"' "${roles}" || return 1
  else
    ! grep -q '"role": "attention_mask"' "${roles}" || return 1
  fi
}

have_phase_weights() {
  [[ -s "${CHAT_ART}/$1/weights.bin" ]]
}

share_phase_weights() {
  local src="${CHAT_ART}/prefill/weights.bin"
  local dst="${CHAT_ART}/decode/weights.bin"
  if [[ ! -s "${src}" ]]; then
    echo "error: missing shared Llama weights ${src}" >&2
    exit 1
  fi
  if [[ -e "${dst}" ]]; then
    if [[ "${src}" -ef "${dst}" ]]; then
      return
    fi
    rm -f "${dst}"
  fi
  ln -s "../prefill/weights.bin" "${dst}" 2>/dev/null || ln "${src}" "${dst}"
}

kb_to_gib() {
  awk -v kb="$1" 'BEGIN { printf "%.1f", kb / 1024 / 1024 }'
}

check_rax_payload_space() {
  local payload_kb
  payload_kb="$(du -sk "${PREFILL_TTNN}" "${DECODE_TTNN}" "${CHAT_ART}" \
    | awk '{sum += $1} END {print sum + 0}')"
  local margin_kb=$((2 * 1024 * 1024))
  local required_kb=$((payload_kb + margin_kb))
  local available_kb
  available_kb="$(df -Pk "${RAX_PACKAGE_DIR}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < required_kb )); then
    echo "error: not enough free space to embed the Llama payload into ${RAX_FILE}" >&2
    echo "  required: $(kb_to_gib "${required_kb}") GiB" >&2
    echo "  available: $(kb_to_gib "${available_kb}") GiB" >&2
    echo "Free space on this filesystem or set EMBED_RAX_PAYLOAD=0 for a manifest-only debug package." >&2
    exit 1
  fi
}

SYS_DIR="${TTRT_SYS_DIR:-${ARTIFACT_ROOT}/ttrt_sys}"
SYS="${SYS_DIR}/system_desc.ttsys"
SYS_CACHE_DIR="${BUDDY_BUILD}/ttrt_sys"
SYS_CACHE="${SYS_CACHE_DIR}/system_desc.ttsys"
if [[ ! -f "${SYS}" ]]; then
  if [[ -f "${SYS_CACHE}" ]]; then
    echo "=== reuse cached system descriptor: ${SYS_CACHE} ==="
    mkdir -p "${SYS_DIR}"
    cp -f "${SYS_CACHE}" "${SYS}"
  else
    echo "=== query system descriptor ==="
    python -m ttrt query --save-artifacts --artifact-dir "${SYS_DIR}" 2>&1 | tail -3
  fi
fi
if [[ -f "${SYS}" ]]; then
  mkdir -p "${SYS_CACHE_DIR}"
  cp -f "${SYS}" "${SYS_CACHE}" 2>/dev/null || true
fi

# ttmlir-opt fusion flags that work without TTMLIR_ENABLE_OPMODEL=ON.
# Confirmed +15 % steady wall t/s/u and -50 % submit_wait_ms on Llama-3.1-8B
# decode (2026-04-29). Set BUDDY_TTNN_PIPELINE_EXTRA="" to disable.
BUDDY_TTNN_PIPELINE_EXTRA="${BUDDY_TTNN_PIPELINE_EXTRA:- enable-fusing-pass=true enable-permute-matmul-fusion=true enable-erase-inverse-ops-pass=true enable-dram-space-saving-optimization-pass=true enable-implicit-broadcast-folding-pass=true enable-optimization-passes=true}"

PREFILL_ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX}"
DECODE_ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX}"
if [[ "${BUDDY_TT_DECODE_RMSNORM_FUSION}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_rmsnorm"
fi
DECODE_POSTPROCESS_ARGS=()
if [[ "${BUDDY_TT_DECODE_PRECOMPUTE_SDPA_MASK}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_sdpmask"
  DECODE_POSTPROCESS_ARGS+=(--precompute-sdpa-mask)
fi
if [[ "${BUDDY_TT_DECODE_PRECOMPUTE_ROPE}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_ropecache"
  DECODE_POSTPROCESS_ARGS+=(--precompute-rope)
fi
if [[ "${BUDDY_TT_DECODE_MERGE_CACHE_POSITION_INPUTS}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_mergepos"
  DECODE_POSTPROCESS_ARGS+=(--merge-cache-position-inputs)
fi
if [[ "${BUDDY_TT_DECODE_KEEP_CACHE_POSITION_INPUTS}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_keeppos"
  DECODE_POSTPROCESS_ARGS+=(--keep-cache-position-inputs)
fi
if [[ "${BUDDY_TT_DECODE_FOLD_IDENTITY_MUL}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_fold1"
  DECODE_POSTPROCESS_ARGS+=(--fold-identity-mul)
fi
if [[ "${BUDDY_TT_DECODE_PACK_QKV}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_packedqkv"
  DECODE_POSTPROCESS_ARGS+=(--pack-qkv)
fi
if [[ "${BUDDY_TT_DECODE_PACK_MLP_GATE_UP}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_packedmlp"
  DECODE_POSTPROCESS_ARGS+=(--pack-mlp-gate-up)
fi
if [[ "${BUDDY_TT_DECODE_ARGMAX_TILE}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_argmaxtile"
  DECODE_POSTPROCESS_ARGS+=(--argmax-tile)
fi
if [[ "${BUDDY_TT_DECODE_LMHEAD_HIFI2}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_lmheadhifi2"
  DECODE_POSTPROCESS_ARGS+=(--lm-head-hifi2)
fi
if [[ "${BUDDY_TT_DECODE_SPLIT_LM_HEAD}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_splitlm${BUDDY_TT_DECODE_LM_HEAD_SPLITS}"
  DECODE_POSTPROCESS_ARGS+=(--split-lm-head --lm-head-splits "${BUDDY_TT_DECODE_LM_HEAD_SPLITS}")
fi
if [[ "${BUDDY_TT_DECODE_LM_HEAD_DRAM_PC}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_lmpc${BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES}"
  DECODE_POSTPROCESS_ARGS+=(--lm-head-dram-sharded-program-config --lm-head-program-cores "${BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES}")
fi
if [[ "${BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_lm1dpc${BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES}"
  DECODE_POSTPROCESS_ARGS+=(--lm-head-mcast1d-program-config --lm-head-program-cores "${BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES}")
fi
if [[ "${BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_splitembed"
  DECODE_POSTPROCESS_ARGS+=(--split-embedding-weight)
fi
if [[ "${BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_keepweights"
  DECODE_POSTPROCESS_ARGS+=(--keep-static-weight-inputs)
fi
if [[ "${BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_u32tok"
  DECODE_POSTPROCESS_ARGS+=(--native-u32-token-io)
fi
if [[ "${BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_createqkv"
  DECODE_POSTPROCESS_ARGS+=(--fuse-create-qkv-heads-decode)
fi
if [[ "${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_concatheads"
  DECODE_POSTPROCESS_ARGS+=(--fuse-concat-heads-decode)
fi
if [[ "${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT}" == "1" ]]; then
  DECODE_ARTIFACT_SUFFIX="${DECODE_ARTIFACT_SUFFIX}_concatheadssdpa"
  DECODE_POSTPROCESS_ARGS+=(--fuse-concat-heads-decode-sdpa-output)
fi

if [[ ${#DECODE_POSTPROCESS_ARGS[@]} -gt 0 \
      && "${CHAT_ART}" == "${BASE_CHAT_ART}"* ]]; then
  CHAT_ART="${BASE_CHAT_ART}${DECODE_ARTIFACT_SUFFIX}"
  mkdir -p "${CHAT_ART}"
fi

PREFILL_TTNN="${TTIR_OUT}/${TTIR_STEM}_prefill_static${PREFILL_ARTIFACT_SUFFIX}.ttnn"
DECODE_TTNN="${TTIR_OUT}/${TTIR_STEM}_decode_static${DECODE_ARTIFACT_SUFFIX}.ttnn"
RAX_MANIFEST="${RAX_PACKAGE_DIR}/${RAX_STEM}.rhal.mlir"
RAX_FILE="${RAX_PACKAGE_DIR}/${RAX_STEM}.rax"

if [[ "${SKIP_LOWER}" != "1" ]]; then
  if ! have_artifact "${PREFILL_TTNN}"; then
    echo "=== [1/4] Lower prefill TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    python buddy-llama31-lower-ttir.py \
      --mode prefill --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --batch "${BATCH_SIZE}" \
      --element-dtype bf16 \
      --output-stem-prefix "${TTIR_STEM}_ttir" \
      "${LOWER_COMMON_EXTRA[@]}" \
      "${LOWER_PREFILL_EXTRA[@]}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    echo "=== [1b/4] prefill TTIR -> TTNN ==="
    ttmlir-opt "${TTIR_OUT}/${TTIR_STEM}_ttir_prefill.mlir" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}${BUDDY_TTNN_PIPELINE_EXTRA}" \
      -o "${TTIR_OUT}/${TTIR_STEM}_prefill_static${PREFILL_ARTIFACT_SUFFIX}_ttnn.mlir"
    echo "=== [1c/4] prefill TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${TTIR_OUT}/${TTIR_STEM}_prefill_static${PREFILL_ARTIFACT_SUFFIX}_ttnn.mlir" \
      -o "${PREFILL_TTNN}"
  else
    echo "=== [1/4] prefill flatbuffer already exists: ${PREFILL_TTNN} ==="
  fi

  DECODE_NEEDS_BUILD=0
  if ! have_artifact "${DECODE_TTNN}"; then
    DECODE_NEEDS_BUILD=1
  elif [[ ${#DECODE_POSTPROCESS_ARGS[@]} -gt 0 ]] && ! have_phase_metadata decode; then
    echo "=== [2/4] decode postprocess metadata missing; regenerating ${DECODE_TTNN} ==="
    DECODE_NEEDS_BUILD=1
  fi
  if [[ "${DECODE_NEEDS_BUILD}" == "1" ]]; then
    echo "=== [2/4] Lower decode TTIR (static-cache ${MAX_CACHE_LEN}) ==="
    env "${DECODE_LOWER_ENV[@]}" python buddy-llama31-lower-ttir.py \
      --mode decode --static-cache --max-cache-len "${MAX_CACHE_LEN}" \
      --batch "${BATCH_SIZE}" \
      --element-dtype bf16 \
      --output-stem-prefix "${TTIR_STEM}_ttir" \
      "${LOWER_COMMON_EXTRA[@]}" \
      "${LOWER_DECODE_EXTRA[@]}" \
      --ttmlir-opt "$(command -v ttmlir-opt)" -o "${TTIR_OUT}"
    DECODE_TTIR_FOR_TTNN="${TTIR_OUT}/${TTIR_STEM}_ttir_decode.mlir"
    if [[ "${BUDDY_TT_DECODE_RMSNORM_FUSION}" == "1" ]]; then
      DECODE_TTIR_FOR_TTNN="${TTIR_OUT}/${TTIR_STEM}_ttir_decode_rmsnorm.mlir"
      echo "=== [2a/4] decode TTIR RMSNorm fusion ==="
      ttmlir-opt "${TTIR_OUT}/${TTIR_STEM}_ttir_decode.mlir" \
        --canonicalize --ttir-fusing \
        -o "${DECODE_TTIR_FOR_TTNN}"
    fi
    echo "=== [2b/4] decode TTIR -> TTNN ==="
    DECODE_TTNN_MLIR="${TTIR_OUT}/${TTIR_STEM}_decode_static${DECODE_ARTIFACT_SUFFIX}_ttnn.mlir"
    DECODE_TTNN_PREPOST_MLIR="${DECODE_TTNN_MLIR}"
    if [[ ${#DECODE_POSTPROCESS_ARGS[@]} -gt 0 ]]; then
      DECODE_TTNN_PREPOST_MLIR="${TTIR_OUT}/${TTIR_STEM}_decode_static${DECODE_ARTIFACT_SUFFIX}_prepost_ttnn.mlir"
    fi
    ttmlir-opt "${DECODE_TTIR_FOR_TTNN}" \
      --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYS}${BUDDY_TTNN_PIPELINE_EXTRA}" \
      -o "${DECODE_TTNN_PREPOST_MLIR}"
    if [[ ${#DECODE_POSTPROCESS_ARGS[@]} -gt 0 ]]; then
      if [[ ( "${BUDDY_TT_DECODE_PACK_QKV}" == "1" \
              || "${BUDDY_TT_DECODE_PACK_MLP_GATE_UP}" == "1" \
              || "${BUDDY_TT_DECODE_PRECOMPUTE_SDPA_MASK}" == "1" \
              || "${BUDDY_TT_DECODE_PRECOMPUTE_ROPE}" == "1" \
              || "${BUDDY_TT_DECODE_MERGE_CACHE_POSITION_INPUTS}" == "1" \
              || "${BUDDY_TT_DECODE_KEEP_CACHE_POSITION_INPUTS}" == "1" \
              || "${BUDDY_TT_DECODE_FOLD_IDENTITY_MUL}" == "1" \
              || "${BUDDY_TT_DECODE_SPLIT_LM_HEAD}" == "1" \
              || "${BUDDY_TT_DECODE_LM_HEAD_DRAM_PC}" == "1" \
              || "${BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC}" == "1" \
              || "${BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT}" == "1" \
              || "${BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS}" == "1" \
              || "${BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO}" == "1" \
              || "${BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS}" == "1" \
              || "${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS}" == "1" \
              || "${BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT}" == "1" ) ]] && \
          ! have_phase_metadata decode; then
        echo "=== [2b.meta/4] prepare decode metadata for TTNN postprocess ==="
        python llama31_chat_prepare.py \
          --phases decode \
          --max-cache-len "${MAX_CACHE_LEN}" \
          --batch "${BATCH_SIZE}" \
          --metadata-only \
          "${PREPARE_DECODE_EXTRA[@]}" \
          -o "${CHAT_ART}"
      fi
      echo "=== [2b.post/4] decode TTNN experimental postprocess: ${DECODE_POSTPROCESS_ARGS[*]} ==="
      python pack_decode_qkv_ttnn.py \
        --input-ttnn-mlir "${DECODE_TTNN_PREPOST_MLIR}" \
        --output-ttnn-mlir "${DECODE_TTNN_MLIR}" \
        --input-artifacts "${CHAT_ART}" \
        --output-artifacts "${CHAT_ART}" \
        "${DECODE_POSTPROCESS_ARGS[@]}"
    fi
    echo "=== [2c/4] decode TTNN -> flatbuffer ==="
    ttmlir-translate --ttnn-to-flatbuffer \
      "${DECODE_TTNN_MLIR}" \
      -o "${DECODE_TTNN}"
  else
    echo "=== [2/4] decode flatbuffer already exists: ${DECODE_TTNN} ==="
  fi
else
  echo "=== skipping lower/ttnn/flatbuffer (SKIP_LOWER=1) ==="
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if ! have_phase_metadata prefill || ! have_phase_metadata decode \
      || ! have_phase_weights prefill || ! have_phase_weights decode; then
    echo "=== [3/4] Prepare chat artifacts (weights + roles) ==="
    if [[ "${BASE_CHAT_ART}" != "${CHAT_ART}" \
          && -f "${BASE_CHAT_ART}/prefill/weights.bin" \
          && -f "${BASE_CHAT_ART}/decode/weights.bin" ]]; then
      if ! have_phase_metadata prefill; then
        python llama31_chat_prepare.py \
          --phases prefill \
          --max-cache-len "${MAX_CACHE_LEN}" \
          --batch "${BATCH_SIZE}" \
          --metadata-only \
          "${PREPARE_PREFILL_EXTRA[@]}" \
          -o "${CHAT_ART}"
      fi
      if ! have_phase_metadata decode; then
        python llama31_chat_prepare.py \
          --phases decode \
          --max-cache-len "${MAX_CACHE_LEN}" \
          --batch "${BATCH_SIZE}" \
          --metadata-only \
          "${PREPARE_DECODE_EXTRA[@]}" \
          -o "${CHAT_ART}"
      fi
      reuse_static_payloads
    else
      if ! have_phase_metadata prefill || ! have_phase_weights prefill; then
        python llama31_chat_prepare.py \
          --phases prefill \
          --max-cache-len "${MAX_CACHE_LEN}" \
          --batch "${BATCH_SIZE}" \
          "${PREPARE_PREFILL_EXTRA[@]}" \
          -o "${CHAT_ART}"
      fi
      if ! have_phase_metadata decode; then
        python llama31_chat_prepare.py \
          --phases decode \
          --max-cache-len "${MAX_CACHE_LEN}" \
          --batch "${BATCH_SIZE}" \
          --metadata-only \
          "${PREPARE_DECODE_EXTRA[@]}" \
          -o "${CHAT_ART}"
      fi
      share_phase_weights
    fi
  else
    echo "=== [3/4] chat artifacts already exist: ${CHAT_ART} ==="
    if [[ "${BASE_CHAT_ART}" != "${CHAT_ART}" ]]; then
      reuse_static_payloads
    else
      share_phase_weights
    fi
  fi
else
  echo "=== skipping chat prepare (SKIP_PREPARE=1) ==="
  if [[ "${BASE_CHAT_ART}" != "${CHAT_ART}" ]]; then
    reuse_static_payloads
  elif [[ -s "${CHAT_ART}/prefill/weights.bin" ]]; then
    share_phase_weights
  fi
fi

if [[ "${SKIP_PACKAGE}" != "1" ]]; then
  echo "=== [4/5] Package TTNN artifacts for buddy-cli ==="
  if ! have_artifact "${PREFILL_TTNN}" || ! have_artifact "${DECODE_TTNN}"; then
    echo "error: missing non-empty TTNN flatbuffers; rerun without SKIP_LOWER=1" >&2
    exit 1
  fi
  python "${REPO_ROOT}/tools/buddy-codegen/gen_tenstorrent_manifest.py" \
    --model-name "${RAX_MODEL_NAME}" \
    --prefill-ttnn "${PREFILL_TTNN}" \
    --decode-ttnn "${DECODE_TTNN}" \
    --artifacts "${CHAT_ART}" \
    --tokenizer "${LLAMA31_MODEL_PATH}" \
    --max-cache-len "${MAX_CACHE_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --prompt-format "${PROMPT_FORMAT}" \
    --prefill-kv-output-order "${PREFILL_KV_OUTPUT_ORDER}" \
    "${GEN_MANIFEST_EXTRA[@]}" \
    -o "${RAX_MANIFEST}"

  RAX_PACK_ARGS=()
  EMBED_RAX_PAYLOAD="${EMBED_RAX_PAYLOAD:-${EMBED_TTNN_IN_RAX:-1}}"
  if [[ "${EMBED_RAX_PAYLOAD}" == "1" ]]; then
    RAX_PACK_ARGS+=(--embed-payload)
    rm -f "${RAX_FILE}"
    check_rax_payload_space
  fi
  "${BUDDY_BUILD}/bin/rax-pack" "${RAX_MANIFEST}" -o "${RAX_FILE}" "${RAX_PACK_ARGS[@]}"
else
  echo "=== skipping RAX package (SKIP_PACKAGE=1): ${RAX_FILE} ==="
fi

if [[ "${PACKAGE_ONLY}" == "1" ]]; then
  echo "=== package ready: ${RAX_FILE} ==="
  exit 0
fi

echo "=== [5/5] Interactive chat on Tenstorrent ==="
EXTRA_ARGS=()
if [[ "${MAX_NEW_TOKENS}" != "0" ]]; then
  EXTRA_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
CLI_ARGS=(--model "${RAX_FILE}" --max-tokens "${MAX_NEW_TOKENS}")
"${BUDDY_BUILD}/bin/buddy-cli" "${CLI_ARGS[@]}"
