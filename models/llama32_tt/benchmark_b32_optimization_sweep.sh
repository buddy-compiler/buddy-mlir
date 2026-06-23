#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  models/llama32_tt/benchmark_b32_optimization_sweep.sh [variant]

Environment:
  BUDDY_REPO_ROOT          Buddy repository root. Defaults to current directory.
  BUDDY_BUILD             Buddy build dir. Defaults to $BUDDY_REPO_ROOT/build-tenstorrent.
  TTMLIR_BUILD            tt-mlir build dir. Defaults to $BUDDY_REPO_ROOT/build-ttmlir.
  TTMLIR_TOOLCHAIN_DIR    tt-mlir toolchain dir. Defaults to $BUDDY_REPO_ROOT/build-ttmlir-toolchain.
  LLAMA32_MODEL_PATH      Local Llama-3.2-3B model directory, required for rebuilds.
  LOG_DIR                 Output log dir. Defaults to $BUDDY_REPO_ROOT/benchmark_logs/llama32_b32_opt.
  REPEAT_RUNS             Runtime repetitions after one rebuild. Defaults to 1.
  BUDDY_BENCH_TIMEOUT_SEC Optional per-runtime timeout in seconds. Defaults to disabled.

Variants:
  baseline_defer
  rmsnorm_defer
  rmsnorm_splitembed_defer
  rmsnorm_splitembed_keepweights_defer
  rmsnorm_packqkv_splitembed_defer
  rmsnorm_packqkv_splitembed_fold1_defer
  rmsnorm_packqkv_splitembed_lmhead_hifi2_defer
  rmsnorm_packqkv_splitembed_argmax_tile_defer
  rmsnorm_packqkv_splitembed_pack_mlp_defer
  rmsnorm_packqkv_splitembed_u32tok_defer
  rmsnorm_packqkv_splitembed_precompute_rope_defer
  rmsnorm_packqkv_splitembed_rope_splitlm_defer
  rmsnorm_packqkv_splitembed_rope_lmhead_mcast1d_pc_defer
  rmsnorm_packqkv_splitembed_splitlm_defer
  rmsnorm_packqkv_splitembed_lmhead_dram_pc_defer
  rmsnorm_packqkv_splitembed_lmhead_mcast1d_pc_defer
  rmsnorm_packqkv_splitembed_fuse_create_qkv_heads_defer
  rmsnorm_packqkv_splitembed_fuse_concat_heads_defer
  rmsnorm_packqkv_splitembed_fuse_concat_heads_sdpa_output_defer
  rmsnorm_packqkv_splitembed_keepweights_defer
  rmsnorm_packqkv_splitembed_keepweights_fold1_defer
  rmsnorm_packqkv_splitembed_keepweights_lmhead_hifi2_defer
  rmsnorm_packqkv_splitembed_keepweights_argmax_tile_defer
  rmsnorm_packqkv_splitembed_keepweights_pack_mlp_defer
  rmsnorm_packqkv_splitembed_keepweights_u32tok_defer
EOF
  exit 0
fi

variant="${1:-baseline_defer}"

BUDDY_REPO_ROOT="${BUDDY_REPO_ROOT:-$(pwd)}"
BUDDY_BUILD="${BUDDY_BUILD:-$BUDDY_REPO_ROOT/build-tenstorrent}"
TTMLIR_BUILD="${TTMLIR_BUILD:-$BUDDY_REPO_ROOT/build-ttmlir}"
TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-$BUDDY_REPO_ROOT/build-ttmlir-toolchain}"
LOG_DIR="${LOG_DIR:-$BUDDY_REPO_ROOT/benchmark_logs/llama32_b32_opt}"
MODEL_PATH="${LLAMA32_MODEL_PATH:-${BUDDY_LLAMA32_MODEL_PATH:-}}"
REPEAT_RUNS="${REPEAT_RUNS:-1}"
BUDDY_BENCH_TIMEOUT_SEC="${BUDDY_BENCH_TIMEOUT_SEC:-0}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "LLAMA32_MODEL_PATH or BUDDY_LLAMA32_MODEL_PATH must point to a local model directory" >&2
  exit 2
fi
if ! [[ "$REPEAT_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEAT_RUNS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$BUDDY_BENCH_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "BUDDY_BENCH_TIMEOUT_SEC must be a non-negative integer" >&2
  exit 2
fi

if [[ -e /proc/driver/tenstorrent/0/pids ]]; then
  active_pids="$(tr '\n' ' ' </proc/driver/tenstorrent/0/pids | xargs || true)"
  if [[ -n "$active_pids" && "${BUDDY_ALLOW_BUSY_DEVICE:-0}" != "1" ]]; then
    echo "Tenstorrent device is busy: $active_pids" >&2
    echo "Set BUDDY_ALLOW_BUSY_DEVICE=1 to run anyway." >&2
    exit 3
  fi
fi

mkdir -p "$LOG_DIR"

flags=(
  -DBUDDY_BUILD_LLAMA32_TT_MODEL=ON
  -DBUDDY_LLAMA32_MODEL_PATH="$MODEL_PATH"
  -DBUDDY_LLAMA32_FIXED_BATCH_SIZES=32
  -DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=OFF
  -DBUDDY_LLAMA32_DECODE_ARGMAX_TILE=OFF
  -DBUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL=OFF
  -DBUDDY_LLAMA32_DECODE_PACK_QKV=OFF
  -DBUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP=OFF
  -DBUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=OFF
  -DBUDDY_LLAMA32_DECODE_LMHEAD_HIFI2=OFF
  -DBUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD=OFF
  -DBUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC=OFF
  -DBUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC=OFF
  -DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=OFF
  -DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=OFF
  -DBUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO=OFF
  -DBUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS=OFF
  -DBUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS=OFF
  -DBUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT=OFF
)

case "$variant" in
  baseline_defer)
    ;;
  rmsnorm_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    ;;
  rmsnorm_splitembed_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    ;;
  rmsnorm_splitembed_keepweights_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    ;;
  rmsnorm_packqkv_splitembed_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    ;;
  rmsnorm_packqkv_splitembed_fold1_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL=ON)
    ;;
  rmsnorm_packqkv_splitembed_lmhead_hifi2_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_LMHEAD_HIFI2=ON)
    ;;
  rmsnorm_packqkv_splitembed_argmax_tile_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_ARGMAX_TILE=ON)
    ;;
  rmsnorm_packqkv_splitembed_pack_mlp_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP=ON)
    ;;
  rmsnorm_packqkv_splitembed_u32tok_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO=ON)
    ;;
  rmsnorm_packqkv_splitembed_precompute_rope_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON)
    ;;
  rmsnorm_packqkv_splitembed_rope_splitlm_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD=ON)
    ;;
  rmsnorm_packqkv_splitembed_rope_lmhead_mcast1d_pc_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC=ON)
    ;;
  rmsnorm_packqkv_splitembed_splitlm_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD=ON)
    ;;
  rmsnorm_packqkv_splitembed_lmhead_dram_pc_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC=ON)
    ;;
  rmsnorm_packqkv_splitembed_lmhead_mcast1d_pc_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC=ON)
    ;;
  rmsnorm_packqkv_splitembed_fuse_create_qkv_heads_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS=ON)
    ;;
  rmsnorm_packqkv_splitembed_fuse_concat_heads_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS=ON)
    ;;
  rmsnorm_packqkv_splitembed_fuse_concat_heads_sdpa_output_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_fold1_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_lmhead_hifi2_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_LMHEAD_HIFI2=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_argmax_tile_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_ARGMAX_TILE=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_pack_mlp_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP=ON)
    ;;
  rmsnorm_packqkv_splitembed_keepweights_u32tok_defer)
    flags+=(-DBUDDY_LLAMA32_DECODE_RMSNORM_FUSION=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_PACK_QKV=ON)
    flags+=(-DBUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO=ON)
    ;;
  *)
    echo "unknown variant: $variant" >&2
    "$0" --help >&2
    exit 2
    ;;
esac

cd "$BUDDY_REPO_ROOT"

if [[ -f "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env/activate" ]]; then
  export _ACTIVATE_ECHO_TOOLCHAIN_DIR_AND_EXIT="${_ACTIVATE_ECHO_TOOLCHAIN_DIR_AND_EXIT:-}"
  export _ACTIVATE_SUPPRESS_INIT_WARNING="${_ACTIVATE_SUPPRESS_INIT_WARNING:-1}"
  export TTMLIR_VENV_DIR="${TTMLIR_VENV_DIR:-$TTMLIR_TOOLCHAIN_DIR/venv}"
  # shellcheck disable=SC1091
  source "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env/activate"
fi

export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$BUDDY_REPO_ROOT/thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal}"
export TT_METAL_HOME="${TT_METAL_HOME:-$TT_METAL_RUNTIME_ROOT}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"
export TT_LOGGER_LEVEL="${TT_LOGGER_LEVEL:-FATAL}"
export TT_METAL_LOGGER_LEVEL="${TT_METAL_LOGGER_LEVEL:-FATAL}"
export TTMLIR_RUNTIME_LOGGER_LEVEL="${TTMLIR_RUNTIME_LOGGER_LEVEL:-FATAL}"

artifact_root="$BUDDY_BUILD/models/llama32_tt_b32"
rm -rf \
  "$artifact_root/ttir_out_static" \
  "$artifact_root/chat_artifacts" \
  "$artifact_root"/chat_artifacts_* \
  "$artifact_root/llama32_tt_b32.rhal.mlir" \
  "$artifact_root/llama32_tt_b32.rax"

cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" "${flags[@]}"
BUDDY_LLAMA31_IGNORE_EOS=1 cmake --build "$BUDDY_BUILD" --target buddy-cli llama32_tt_b32_rax

payload_base="${BUDDY_RAX_PAYLOAD_BASE:-/wafer/$USER/buddy_rax_payload_bench}"
export BUDDY_RAX_PAYLOAD_DIR="${payload_base}/${variant}"
mkdir -p "$BUDDY_RAX_PAYLOAD_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
traces=()
for run_idx in $(seq 1 "$REPEAT_RUNS"); do
  log="$LOG_DIR/${variant}_${ts}_run${run_idx}.log"
  trace="$LOG_DIR/${variant}_${ts}_run${run_idx}.json"
  traces+=("$trace")
  export BUDDY_LLAMA31_BATCH_TRACE_OUT="$trace"

  echo "=== runtime run ${run_idx}/${REPEAT_RUNS}: ${variant} ==="
  runtime_cmd=(
    "$BUDDY_BUILD/bin/buddy-cli"
    --model "$artifact_root/llama32_tt_b32.rax" \
    --prompt-file "$BUDDY_REPO_ROOT/models/llama32_tt/llama32_3b_default_prompts.txt" \
    --prompt-length 32 \
    --batch-size 32 \
    --max-tokens 96 \
    --temperature 0 \
    --defer-decode-token-readback
  )
  if [[ "$BUDDY_BENCH_TIMEOUT_SEC" == "0" ]]; then
    "${runtime_cmd[@]}" 2>&1 | tee "$log"
  else
    timeout "$BUDDY_BENCH_TIMEOUT_SEC" "${runtime_cmd[@]}" 2>&1 | tee "$log"
  fi
done

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" - "$variant" "${traces[@]}" <<'PY'
import json
import sys

variant = sys.argv[1]
traces = sys.argv[2:]
rows = []
for trace in traces:
    with open(trace, "r", encoding="utf-8") as f:
        data = json.load(f)
    batch = int(data.get("batch_size", 32))
    count = int(data.get("decode_count", 0))
    full_s = data.get("decode_wall_plus_deferred_readback_seconds") or data.get(
        "decode_wall_seconds"
    )
    steady_s = data.get("decode_steady_wall_plus_deferred_readback_seconds") or data.get(
        "decode_steady_wall_seconds"
    )
    full = (count / full_s) if count and full_s else 0.0
    steady = ((count - 1) / steady_s) if count > 1 and steady_s else 0.0
    rows.append((trace, batch, count, full, steady))
    print(
        f"summary trace={trace} batch={batch} decode_count={count} "
        f"full_tok_s_user={full:.4f} steady_tok_s_user={steady:.4f}"
    )

avg_full = sum(row[3] for row in rows) / len(rows)
avg_steady = sum(row[4] for row in rows) / len(rows)
min_steady = min(row[4] for row in rows)
max_steady = max(row[4] for row in rows)
print(
    f"average variant={variant} runs={len(rows)} full_tok_s_user={avg_full:.4f} "
    f"steady_tok_s_user={avg_steady:.4f} steady_min={min_steady:.4f} "
    f"steady_max={max_steady:.4f}"
)
PY
