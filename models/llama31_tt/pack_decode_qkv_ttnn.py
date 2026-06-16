#!/usr/bin/env python3
# ===- pack_decode_qkv_ttnn.py ---------------------------------------------
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
# ===------------------------------------------------------------------------
#
# Experimental post-processing for generated decode TTNN MLIR.
#
# Supported rewrites:
#   - fold generated decode-time QKV weight concat inputs into one packed weight
#     slot per layer;
#   - let argmax consume tiled LM-head logits directly, skipping a full-logits
#     row-major conversion;
#   - optionally force that tiled-input argmax path to single-core execution,
#     because the current multicore TTNN argmax requires row-major input.
#   - replace the full-logits row-major argmax path with tiled-input
#     ttnn.topk(k=1) as an LM-head sampling experiment.
#   - fold generated multiply-by-one ops in the decode graph.
#   - merge repeated cache_position input slots into one graph input.
#   - keep cached cache_position inputs alive by removing graph-side dealloc ops.
#   - precompute the decode SDPA attention mask and position inputs on the
#     host/runtime side instead of building them in the TTNN graph.
#   - precompute the per-position RoPE cos/sin tensors on the host/runtime side.
#   - change only the LM-head matmul compute config to HiFi2/no-fp32-accum for
#     narrow precision experiments.
#   - change only the LM-head matmul fp32 destination accumulation flag while
#     leaving math fidelity unchanged.
#   - split the final LM-head vocab projection into fixed vocab-row shards,
#     preserving BF16 math and concatenating logits before argmax.
#   - optionally attach an official-style DRAM-sharded matmul program config to
#     the LM-head projection matmul(s).
#   - add a second static embedding-weight slot in the row-major layout used by
#     ttnn.embedding, avoiding a large per-token tied-weight layout conversion.
#   - keep static weight inputs alive in the decode graph by removing generated
#     ttnn.deallocate ops for weight/inv_freq function arguments.
#   - replace the generated decode RoPE arithmetic subgraphs with
#     ttnn.rotary_embedding.
#
# ===------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import NamedTuple


class QKVPack(NamedTuple):
    q_arg: int
    k_arg: int
    v_arg: int
    result: str
    packed_type: str
    packed_shape: list[int]


class MLPGateUpPack(NamedTuple):
    gate_arg: int
    up_arg: int
    gate_result: str
    up_result: str
    input_value: str
    input_type: str
    gate_weight_type: str
    packed_weight_type: str
    packed_output_type: str
    half_output_type: str
    packed_weight_shape: list[int]


class SplitEmbeddingWeight(NamedTuple):
    source_arg: int
    split_arg: int
    split_type: str


class LmHeadSplit(NamedTuple):
    source_arg: int
    first_split_arg: int
    splits: int
    rows_per_split: int
    hidden: int
    dtype: str
    split_weight_type: str
    dropped_source_arg: bool


class LmHeadProgramConfig(NamedTuple):
    rewrites: int
    num_cores: int
    in0_block_w: int
    per_core_m: int
    per_core_n_values: list[int]


class PrecomputedSdpaInputs(NamedTuple):
    mask_arg: int
    mask_type: str
    position_arg: int
    position_type: str


class PrecomputedRopeInputs(NamedTuple):
    cos_arg: int
    cos_type: str
    sin_arg: int
    sin_type: str


_CONCAT_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "ttnn\.concat"'
    r'\(%arg(?P<q>\d+), %arg(?P<k>\d+), %arg(?P<v>\d+)\) '
    r'<\{dim = 0 : si32\}> : \((?P<input_types>.*)\) -> '
    r'(?P<output_type>tensor<(?P<rows>\d+)x(?P<cols>\d+)x'
    r'(?P<dtype>[a-z0-9]+), #[^>]+>)$'
)
_TENSOR_TYPE_RE = re.compile(
    r'tensor<(?P<rows>\d+)x(?P<cols>\d+)x(?P<dtype>[a-z0-9]+), #[^>]+>'
)
_RANK2_TENSOR_TYPE_WITH_LAYOUT_RE = re.compile(
    r"^tensor<(?P<rows>\d+)x(?P<cols>\d+)x(?P<dtype>[a-z0-9]+), "
    r"(?P<layout>#[^>]+)>$"
)
_MATMUL_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "ttnn\.matmul"'
    r'\(%(?P<input>\d+), %arg(?P<weight_arg>\d+)\) '
    r'<\{(?P<attrs>.*)\}> : '
    r'\((?P<input_type>tensor<[^>]+>), (?P<weight_type>tensor<[^>]+>)\) '
    r'-> (?P<output_type>tensor<[^>]+>)$'
)
_ARG_RE = re.compile(r"%arg(\d+)\b")
_ASSIGN_RE = re.compile(r'^\s*%(?P<result>\d+) = "(?P<op>[^"]+)"')
_LM_HEAD_MATMUL_RE = re.compile(
    r'(?P<prefix>^\s*%\d+ = "ttnn\.matmul"\([^)]*\) <\{)'
    r'(?P<attrs>.*compute_config = #ttnn\.device_compute_kernel_config<'
    r'(?P<config>[^>]*)>.*)'
    r'(?P<suffix>\}> : \(tensor<[^>]+>, '
    r'tensor<(?P<vocab>\d+)x(?P<hidden>\d+)x(?P<dtype>[a-z0-9]+), #[^>]+>\) '
    r'-> tensor<[^x>]+x(?P=vocab)x(?P=dtype), #[^>]+>$)'
)
_LM_HEAD_MATMUL_FULL_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "ttnn\.matmul"'
    r'\(%(?P<input>\d+), %arg(?P<weight_arg>\d+)\) '
    r'<\{(?P<attrs>.*)\}> : '
    r'\((?P<input_type>tensor<(?P<batch>\d+)x(?P<hidden>\d+)x'
    r'(?P<dtype>[a-z0-9]+), (?P<input_layout>#[^>]+)>), '
    r'(?P<weight_type>tensor<(?P<vocab>\d+)x(?P=hidden)x(?P=dtype), '
    r'(?P<weight_layout>#[^>]+)>)\) -> '
    r'(?P<output_type>tensor<(?P=batch)x(?P=vocab)x(?P=dtype), '
    r'(?P<output_layout>#[^>]+)>)$'
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process generated decode TTNN MLIR for experiments."
    )
    parser.add_argument("--input-ttnn-mlir", type=Path, required=True)
    parser.add_argument("--output-ttnn-mlir", type=Path, required=True)
    parser.add_argument(
        "--input-artifacts",
        type=Path,
        required=True,
        help="Source chat_artifacts directory.",
    )
    parser.add_argument(
        "--output-artifacts",
        type=Path,
        required=True,
        help="Destination chat_artifacts directory to rewrite.",
    )
    parser.add_argument(
        "--pack-qkv",
        action="store_true",
        help=(
            "Pack per-layer Q/K/V weight slots and remove generated static "
            "QKV concat ops. This is the default when no rewrite is selected."
        ),
    )
    parser.add_argument(
        "--pack-mlp-gate-up",
        action="store_true",
        help=(
            "Pack per-layer MLP gate/up weight slots into one matmul, then "
            "slice the packed output and apply SiLU to the gate half."
        ),
    )
    parser.add_argument(
        "--argmax-tile",
        action="store_true",
        help=(
            "Remove the full-logits row-major to_layout immediately before "
            "device-side argmax when the argmax can consume tiled logits."
        ),
    )
    parser.add_argument(
        "--argmax-tile-singlecore",
        action="store_true",
        help=(
            "Remove the full-logits row-major to_layout before argmax and "
            "set the rewritten argmax to use_multicore=false."
        ),
    )
    parser.add_argument(
        "--lm-head-topk",
        action="store_true",
        help=(
            "Replace the final full-logits row-major argmax path with "
            "ttnn.topk(k=1) on tiled logits. Experimental."
        ),
    )
    parser.add_argument(
        "--fold-identity-mul",
        action="store_true",
        help=(
            "Fold generated ttnn.multiply ops whose right-hand side is a "
            "cached all-ones scalar tensor and whose result type matches the "
            "non-scalar input."
        ),
    )
    parser.add_argument(
        "--merge-cache-position-inputs",
        action="store_true",
        help=(
            "Merge duplicate decode cache_position input slots into a single "
            "slot and remove their graph-side dealloc ops."
        ),
    )
    parser.add_argument(
        "--keep-cache-position-inputs",
        action="store_true",
        help=(
            "Remove generated ttnn.deallocate ops for decode cache_position "
            "input slots without changing the graph signature."
        ),
    )
    parser.add_argument(
        "--precompute-sdpa-mask",
        action="store_true",
        help=(
            "Replace the generated decode SDPA mask/position construction "
            "with runtime-provided sdpa_mask and sdpa_position inputs."
        ),
    )
    parser.add_argument(
        "--precompute-rope",
        action="store_true",
        help=(
            "Replace generated per-step RoPE cos/sin construction with "
            "runtime-provided rope_cos and rope_sin inputs."
        ),
    )
    parser.add_argument(
        "--lm-head-hifi2",
        action="store_true",
        help=(
            "Change only the final vocab-projection matmul to "
            "math_fidelity=hifi2 and fp32_dest_acc_en=false."
        ),
    )
    parser.add_argument(
        "--lm-head-no-fp32-accum",
        action="store_true",
        help=(
            "Change only the final vocab-projection matmul to "
            "fp32_dest_acc_en=false while preserving its math fidelity."
        ),
    )
    parser.add_argument(
        "--split-lm-head",
        action="store_true",
        help=(
            "Split the final vocab-projection matmul into fixed vocab-row "
            "shards and concatenate logits before argmax. Experimental."
        ),
    )
    parser.add_argument(
        "--lm-head-splits",
        type=int,
        default=8,
        help="Number of vocab-row shards used with --split-lm-head.",
    )
    parser.add_argument(
        "--lm-head-dram-sharded-program-config",
        action="store_true",
        help=(
            "Attach an official-style DRAM-sharded matmul program config to "
            "the final LM-head projection matmul(s). Experimental."
        ),
    )
    parser.add_argument(
        "--lm-head-mcast1d-program-config",
        action="store_true",
        help=(
            "Attach a 1D multicast matmul program config to the final "
            "LM-head projection matmul(s). Experimental."
        ),
    )
    parser.add_argument(
        "--lm-head-program-cores",
        type=int,
        default=48,
        help=(
            "Number of cores used for --lm-head-dram-sharded-program-config. "
            "The Llama3.2 Blackhole single-device LM-head path uses 48."
        ),
    )
    parser.add_argument(
        "--split-embedding-weight",
        action="store_true",
        help=(
            "Add a duplicate static embedding weight slot with the row-major "
            "layout consumed by ttnn.embedding, keeping the original tied "
            "weight slot for the LM-head matmul."
        ),
    )
    parser.add_argument(
        "--keep-static-weight-inputs",
        action="store_true",
        help=(
            "Remove generated ttnn.deallocate ops for static weight/inv_freq "
            "function arguments. Runtime inputs and temporaries are left "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--native-u32-token-io",
        action="store_true",
        help=(
            "Use the row-major u32 token tensor produced by device-side argmax "
            "as the decode output and the next decode input. This removes the "
            "generated si32/tile token-id conversion loop around embedding."
        ),
    )
    parser.add_argument(
        "--fuse-concat-heads-decode",
        action="store_true",
        help=(
            "Replace decode attention output reshape with "
            "ttnn.nlp_concat_heads_decode followed by the original reshape. "
            "This mirrors the specialized tt-metal decode head-concat path."
        ),
    )
    parser.add_argument(
        "--fuse-concat-heads-decode-sdpa-output",
        action="store_true",
        help=(
            "Like --fuse-concat-heads-decode, but asks "
            "ttnn.scaled_dot_product_attention_decode to produce the sharded "
            "layout consumed by ttnn.nlp_concat_heads_decode directly. This "
            "avoids an extra to_memory_config after SDPA."
        ),
    )
    parser.add_argument(
        "--fuse-create-qkv-heads-decode",
        action="store_true",
        help=(
            "Replace the generated packed-QKV slice/reshape head creation "
            "with ttnn.nlp_create_qkv_heads_decode. This is an experimental "
            "step toward the tt-metal attention path."
        ),
    )
    parser.add_argument(
        "--fuse-rope",
        action="store_true",
        help=(
            "Replace generated decode RoPE arithmetic patterns with "
            "ttnn.rotary_embedding. This is experimental and guarded because "
            "the local tt-mlir build does not enable OpModel-backed TTNN "
            "fusing."
        ),
    )
    return parser.parse_args()


def _split_func_args(args_text: str) -> list[str]:
    if not args_text.strip():
        return []
    return re.split(r", (?=%arg\d+:)", args_text)


def _find_subgraph_args(text: str) -> tuple[int, int, list[str]]:
    marker = "func.func @subgraph0("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("missing func.func @subgraph0")
    args_start = start + len(marker)
    args_end = text.find(") ->", args_start)
    if args_end < 0:
        raise RuntimeError("could not locate @subgraph0 argument list")
    return args_start, args_end, _split_func_args(text[args_start:args_end])


def _extract_packs(text: str) -> list[QKVPack]:
    packs: list[QKVPack] = []
    for line in text.splitlines():
        match = _CONCAT_RE.match(line)
        if not match:
            continue
        input_types = list(_TENSOR_TYPE_RE.finditer(match.group("input_types")))
        if len(input_types) != 3:
            continue
        q_rows = int(input_types[0].group("rows"))
        k_rows = int(input_types[1].group("rows"))
        v_rows = int(input_types[2].group("rows"))
        q_cols = int(input_types[0].group("cols"))
        k_cols = int(input_types[1].group("cols"))
        v_cols = int(input_types[2].group("cols"))
        dtype = input_types[0].group("dtype")
        if (
            k_rows != v_rows
            or q_cols != k_cols
            or q_cols != v_cols
            or any(t.group("dtype") != dtype for t in input_types)
        ):
            continue
        rows = int(match.group("rows"))
        cols = int(match.group("cols"))
        if rows != q_rows + k_rows + v_rows or cols != q_cols:
            continue
        packs.append(
            QKVPack(
                q_arg=int(match.group("q")),
                k_arg=int(match.group("k")),
                v_arg=int(match.group("v")),
                result=match.group("result"),
                packed_type=match.group("output_type"),
                packed_shape=[rows, cols],
            )
        )
    if not packs:
        raise RuntimeError("found no decode QKV concat patterns to pack")
    return packs


def _rewrite_signature(text: str, packs: list[QKVPack]) -> tuple[str, dict[int, int]]:
    args_start, args_end, args = _find_subgraph_args(text)
    pack_by_q = {pack.q_arg: pack for pack in packs}
    remove_args = {pack.k_arg for pack in packs} | {pack.v_arg for pack in packs}
    kept: list[str] = []
    old_to_new: dict[int, int] = {}

    for spec in args:
        match = re.match(r"%arg(\d+):", spec)
        if not match:
            raise RuntimeError(f"could not parse function argument: {spec[:80]}")
        old_idx = int(match.group(1))
        if old_idx in remove_args:
            continue
        if old_idx in pack_by_q:
            spec = re.sub(
                r": tensor<[^>]+>",
                f": {pack_by_q[old_idx].packed_type}",
                spec,
                count=1,
            )
        old_to_new[old_idx] = len(kept)
        kept.append(spec)

    rewritten = text[:args_start] + ", ".join(kept) + text[args_end:]
    return rewritten, old_to_new


def _rewrite_body(text: str, packs: list[QKVPack]) -> str:
    pack_by_result = {pack.result: pack for pack in packs}
    pack_q_args = {pack.q_arg for pack in packs}
    remove_args = {pack.q_arg for pack in packs}
    remove_args |= {pack.k_arg for pack in packs}
    remove_args |= {pack.v_arg for pack in packs}

    output_lines: list[str] = []
    removed_concat = 0
    replaced_concat_uses = 0
    for line in text.splitlines():
        match = _CONCAT_RE.match(line)
        if match and match.group("result") in pack_by_result:
            removed_concat += 1
            continue

        dealloc_arg = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
        if dealloc_arg and int(dealloc_arg.group(1)) in remove_args:
            continue

        for result, pack in pack_by_result.items():
            if re.search(rf"%{result}\b", line):
                line = re.sub(rf"%{result}\b", f"%arg{pack.q_arg}", line)
                replaced_concat_uses += 1

        # The concat result deallocate now becomes the packed argument
        # deallocate. Keep it after the matmul, matching the original lifetime.
        for q_arg in pack_q_args:
            if f'"ttnn.deallocate"(%arg{q_arg}) ' in line:
                break
        output_lines.append(line)

    if removed_concat != len(packs):
        raise RuntimeError(
            f"removed {removed_concat} concat ops, expected {len(packs)}"
        )
    if replaced_concat_uses < len(packs) * 2:
        raise RuntimeError(
            "unexpectedly few concat-result uses were rewritten; "
            f"got {replaced_concat_uses}"
        )
    return "\n".join(output_lines) + "\n"


def _parse_rank2_tensor_type(
    type_text: str,
) -> tuple[int, int, str, str] | None:
    match = _RANK2_TENSOR_TYPE_WITH_LAYOUT_RE.match(type_text)
    if not match:
        return None
    return (
        int(match.group("rows")),
        int(match.group("cols")),
        match.group("dtype"),
        match.group("layout"),
    )


def _next_layout_name(text: str) -> str:
    max_id = 0
    for match in re.finditer(r"^#ttnn_layout(?P<num>\d*)\s*=", text, re.MULTILINE):
        num = match.group("num")
        if num:
            max_id = max(max_id, int(num))
    return f"#ttnn_layout{max_id + 1}"


def _replace_tensor_layout(type_text: str, layout: str) -> str:
    return re.sub(r", #[^>]+>$", f", {layout}>", type_text)


def _insert_mlp_pack_layouts(
    text: str, pack: MLPGateUpPack
) -> tuple[str, str, str]:
    weight = _parse_rank2_tensor_type(pack.packed_weight_type)
    output = _parse_rank2_tensor_type(pack.packed_output_type)
    if not weight or not output:
        raise RuntimeError("could not parse packed MLP tensor types")
    weight_rows, weight_cols, dtype, _ = weight
    output_rows, output_cols, output_dtype, _ = output
    if (
        dtype != "bf16"
        or output_dtype != "bf16"
        or weight_rows % 32
        or weight_cols % 32
        or output_rows % 32
        or output_cols % 32
    ):
        raise RuntimeError(
            "MLP gate/up packed layouts expect tile-aligned bf16 rank-2 tensors"
        )

    weight_layout = _next_layout_name(text)
    output_layout = f"#ttnn_layout{int(weight_layout.removeprefix('#ttnn_layout')) + 1}"
    weight_decl = (
        f"{weight_layout} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), "
        f"<1x1>, memref<{weight_rows // 32}x{weight_cols // 32}x"
        f"!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>"
    )
    output_decl = (
        f"{output_layout} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), "
        f"<1x1>, memref<{output_rows // 32}x{output_cols // 32}x"
        f"!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>"
    )
    insert_at = text.find("module {")
    if insert_at < 0:
        raise RuntimeError("could not locate module body for layout insertion")
    text = text[:insert_at] + weight_decl + "\n" + output_decl + "\n" + text[insert_at:]
    return (
        text,
        _replace_tensor_layout(pack.packed_weight_type, weight_layout),
        _replace_tensor_layout(pack.packed_output_type, output_layout),
    )


def _attach_mlp_pack_layouts(
    text: str, packs: list[MLPGateUpPack]
) -> tuple[str, list[MLPGateUpPack]]:
    text, packed_weight_type, packed_output_type = _insert_mlp_pack_layouts(
        text, packs[0]
    )
    return (
        text,
        [
            pack._replace(
                packed_weight_type=packed_weight_type,
                packed_output_type=packed_output_type,
            )
            for pack in packs
        ],
    )


def _extract_mlp_gate_up_packs(text: str) -> list[MLPGateUpPack]:
    lines = text.splitlines()
    packs: list[MLPGateUpPack] = []
    for index in range(len(lines) - 5):
        gate = _MATMUL_RE.match(lines[index])
        if not gate:
            continue
        if 'activation = "silu"' not in gate.group("attrs"):
            continue
        up_index = index + 1
        if (
            up_index < len(lines)
            and f'"ttnn.deallocate"(%arg{gate.group("weight_arg")}) ' in lines[up_index]
        ):
            up_index += 1
        if up_index >= len(lines):
            continue
        up = _MATMUL_RE.match(lines[up_index])
        if not up:
            continue
        if "activation =" in up.group("attrs"):
            continue
        if gate.group("input") != up.group("input"):
            continue
        if gate.group("input_type") != up.group("input_type"):
            continue
        if gate.group("output_type") != up.group("output_type"):
            continue
        gate_weight = _parse_rank2_tensor_type(gate.group("weight_type"))
        up_weight = _parse_rank2_tensor_type(up.group("weight_type"))
        half_output = _parse_rank2_tensor_type(gate.group("output_type"))
        if not gate_weight or not up_weight or not half_output:
            continue
        gate_rows, gate_cols, gate_dtype, _ = gate_weight
        up_rows, up_cols, up_dtype, _ = up_weight
        out_rows, out_cols, out_dtype, _ = half_output
        if (
            gate_rows != up_rows
            or gate_cols != up_cols
            or gate_dtype != up_dtype
            or out_cols != gate_rows
            or out_dtype != gate_dtype
        ):
            continue
        multiply_index = up_index + 2
        if (
            multiply_index < len(lines)
            and f'"ttnn.deallocate"(%arg{up.group("weight_arg")}) '
            in lines[multiply_index]
        ):
            multiply_index += 1
        if multiply_index >= len(lines):
            continue
        multiply = re.match(
            rf'\s*%(?P<result>\d+) = "ttnn\.multiply"\('
            rf'%{gate.group("result")}, %{up.group("result")}\) ',
            lines[multiply_index],
        )
        if not multiply:
            continue
        packed_shape = [gate_rows + up_rows, gate_cols]
        packed_weight_type = re.sub(
            rf"tensor<{gate_rows}x{gate_cols}x",
            f"tensor<{packed_shape[0]}x{packed_shape[1]}x",
            gate.group("weight_type"),
            count=1,
        )
        packed_output_type = re.sub(
            rf"tensor<{out_rows}x{out_cols}x",
            f"tensor<{out_rows}x{out_cols * 2}x",
            gate.group("output_type"),
            count=1,
        )
        packs.append(
            MLPGateUpPack(
                gate_arg=int(gate.group("weight_arg")),
                up_arg=int(up.group("weight_arg")),
                gate_result=gate.group("result"),
                up_result=up.group("result"),
                input_value=gate.group("input"),
                input_type=gate.group("input_type"),
                gate_weight_type=gate.group("weight_type"),
                packed_weight_type=packed_weight_type,
                packed_output_type=packed_output_type,
                half_output_type=gate.group("output_type"),
                packed_weight_shape=packed_shape,
            )
        )
    if not packs:
        raise RuntimeError("found no decode MLP gate/up matmul patterns to pack")
    return packs


def _rewrite_mlp_signature(
    text: str, packs: list[MLPGateUpPack]
) -> tuple[str, dict[int, int]]:
    args_start, args_end, args = _find_subgraph_args(text)
    pack_by_gate = {pack.gate_arg: pack for pack in packs}
    remove_args = {pack.up_arg for pack in packs}
    kept: list[str] = []
    old_to_new: dict[int, int] = {}

    for spec in args:
        match = re.match(r"%arg(\d+):", spec)
        if not match:
            raise RuntimeError(f"could not parse function argument: {spec[:80]}")
        old_idx = int(match.group(1))
        if old_idx in remove_args:
            continue
        if old_idx in pack_by_gate:
            spec = re.sub(
                r": tensor<[^>]+>",
                f": {pack_by_gate[old_idx].packed_weight_type}",
                spec,
                count=1,
            )
        old_to_new[old_idx] = len(kept)
        kept.append(spec)

    return text[:args_start] + ", ".join(kept) + text[args_end:], old_to_new


def _rewrite_mlp_body(text: str, packs: list[MLPGateUpPack]) -> str:
    if not packs:
        return text
    lines = text.splitlines()
    pack_by_gate = {pack.gate_result: pack for pack in packs}
    remove_args = {pack.gate_arg for pack in packs} | {pack.up_arg for pack in packs}
    max_result = -1
    for line in lines:
        match = _ASSIGN_RE.match(line)
        if match:
            max_result = max(max_result, int(match.group("result")))
    next_result_id = max_result + 1

    output: list[str] = []
    index = 0
    rewritten = 0
    while index < len(lines):
        gate = _MATMUL_RE.match(lines[index])
        if gate and gate.group("result") in pack_by_gate:
            pack = pack_by_gate[gate.group("result")]
            up_index = index + 1
            if (
                up_index < len(lines)
                and f'"ttnn.deallocate"(%arg{pack.gate_arg}) ' in lines[up_index]
            ):
                up_index += 1
            up = _MATMUL_RE.match(lines[up_index])
            if not up or up.group("result") != pack.up_result:
                raise RuntimeError("MLP gate/up block shape changed during rewrite")
            dealloc_input = lines[up_index + 1]
            multiply_index = up_index + 2
            if (
                multiply_index < len(lines)
                and f'"ttnn.deallocate"(%arg{pack.up_arg}) ' in lines[multiply_index]
            ):
                multiply_index += 1
            multiply = lines[multiply_index]
            dealloc_up = lines[multiply_index + 1]
            dealloc_gate = lines[multiply_index + 2]
            if (
                f'"ttnn.deallocate"(%{pack.input_value}) ' not in dealloc_input
                or f'"ttnn.deallocate"(%{pack.up_result}) ' not in dealloc_up
                or f'"ttnn.deallocate"(%{pack.gate_result}) ' not in dealloc_gate
            ):
                raise RuntimeError("unexpected MLP gate/up deallocation block")

            gate_slice = str(next_result_id)
            up_slice = str(next_result_id + 1)
            silu_result = str(next_result_id + 2)
            next_result_id += 3
            half = _parse_rank2_tensor_type(pack.half_output_type)
            if not half:
                raise RuntimeError("could not parse MLP half output type")
            out_rows, out_cols, _dtype, _layout = half

            matmul_line = lines[index].replace('activation = "silu", ', "", 1)
            matmul_line = matmul_line.replace(
                pack.gate_weight_type, pack.packed_weight_type
            )
            matmul_line = matmul_line.replace(
                pack.half_output_type, pack.packed_output_type
            )
            gate_slice_line = (
                f'{gate.group("indent")}%{gate_slice} = "ttnn.slice_static"'
                f'(%{pack.gate_result}) <{{begins = [0 : i32, 0 : i32], '
                f'ends = [{out_rows} : i32, {out_cols} : i32], '
                f'step = [1 : i32, 1 : i32]}}> : '
                f'({pack.packed_output_type}) -> {pack.half_output_type}'
            )
            up_slice_line = (
                f'{gate.group("indent")}%{up_slice} = "ttnn.slice_static"'
                f'(%{pack.gate_result}) <{{begins = [0 : i32, {out_cols} : i32], '
                f'ends = [{out_rows} : i32, {out_cols * 2} : i32], '
                f'step = [1 : i32, 1 : i32]}}> : '
                f'({pack.packed_output_type}) -> {pack.half_output_type}'
            )
            silu_line = (
                f'{gate.group("indent")}%{silu_result} = "ttnn.silu"'
                f'(%{gate_slice}) : ({pack.half_output_type}) -> '
                f'{pack.half_output_type}'
            )
            gate_slice_dealloc = (
                f'{gate.group("indent")}"ttnn.deallocate"(%{gate_slice}) '
                f'<{{force = false}}> : ({pack.half_output_type}) -> ()'
            )
            packed_dealloc = (
                f'{gate.group("indent")}"ttnn.deallocate"(%{pack.gate_result}) '
                f'<{{force = false}}> : ({pack.packed_output_type}) -> ()'
            )
            multiply = re.sub(
                rf"%{pack.gate_result}\b", f"%{silu_result}", multiply
            )
            multiply = re.sub(rf"%{pack.up_result}\b", f"%{up_slice}", multiply)
            dealloc_up = re.sub(rf"%{pack.up_result}\b", f"%{up_slice}", dealloc_up)
            dealloc_gate = re.sub(
                rf"%{pack.gate_result}\b", f"%{silu_result}", dealloc_gate
            )
            output.extend(
                [
                    matmul_line,
                    gate_slice_line,
                    up_slice_line,
                    silu_line,
                    gate_slice_dealloc,
                    packed_dealloc,
                    dealloc_input,
                    multiply,
                    dealloc_up,
                    dealloc_gate,
                ]
            )
            index = multiply_index + 3
            rewritten += 1
            continue

        dealloc_arg = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', lines[index])
        if dealloc_arg and int(dealloc_arg.group(1)) in remove_args:
            index += 1
            continue
        output.append(lines[index])
        index += 1

    if rewritten != len(packs):
        raise RuntimeError(f"rewrote {rewritten} MLP packs, expected {len(packs)}")
    return "\n".join(output) + "\n"


def _renumber_args(text: str, old_to_new: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old not in old_to_new:
            raise RuntimeError(f"removed argument %arg{old} still referenced")
        return f"%arg{old_to_new[old]}"

    return _ARG_RE.sub(replace, text)


def _rewrite_lm_head_compute_config(
    text: str,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
) -> tuple[str, int]:
    lines = text.splitlines()
    candidate = None
    for idx, line in enumerate(lines):
        if _LM_HEAD_MATMUL_RE.match(line):
            candidate = idx
    if candidate is None:
        raise RuntimeError("could not find final LM-head vocab matmul")

    line = lines[candidate]
    rewritten = line
    if math_fidelity is not None:
        rewritten = re.sub(
            r"math_fidelity = [a-zA-Z0-9_]+",
            f"math_fidelity = {math_fidelity}",
            rewritten,
            count=1,
        )
    if fp32_dest_acc_en is not None:
        rewritten = re.sub(
            r"fp32_dest_acc_en = (true|false)",
            f"fp32_dest_acc_en = {'true' if fp32_dest_acc_en else 'false'}",
            rewritten,
            count=1,
        )
    if rewritten == line:
        raise RuntimeError("LM-head matmul compute config was not changed")
    lines[candidate] = rewritten
    return "\n".join(lines) + "\n", 1


def _rewrite_lm_head_hifi2(text: str) -> tuple[str, int]:
    return _rewrite_lm_head_compute_config(
        text, math_fidelity="hifi2", fp32_dest_acc_en=False
    )


def _rewrite_lm_head_no_fp32_accum(text: str) -> tuple[str, int]:
    return _rewrite_lm_head_compute_config(text, fp32_dest_acc_en=False)


def _largest_divisor_up_to(value: int, max_divisor: int) -> int:
    for divisor in range(min(value, max_divisor), 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _insert_lm_head_split_layouts(
    text: str,
    *,
    batch: int,
    rows_per_split: int,
    hidden: int,
    dtype: str,
) -> tuple[str, str, str]:
    if dtype != "bf16":
        raise RuntimeError("LM-head split currently expects bf16 weights/logits")
    if rows_per_split % 32 or hidden % 32:
        raise RuntimeError(
            "LM-head split expects tile-aligned vocab rows and hidden size"
        )
    batch_tiles = (batch + 31) // 32
    weight_layout = _next_layout_name(text)
    output_layout = (
        f"#ttnn_layout{int(weight_layout.removeprefix('#ttnn_layout')) + 1}"
    )
    weight_decl = (
        f"{weight_layout} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), "
        f"<1x1>, memref<{rows_per_split // 32}x{hidden // 32}x"
        f"!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>"
    )
    output_decl = (
        f"{output_layout} = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), "
        f"<1x1>, memref<{batch_tiles}x{rows_per_split // 32}x"
        f"!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>"
    )
    insert_at = text.find("module {")
    if insert_at < 0:
        raise RuntimeError("could not locate module body for layout insertion")
    text = text[:insert_at] + weight_decl + "\n" + output_decl + "\n" + text[insert_at:]
    return text, weight_layout, output_layout


def _drop_unused_function_arg(text: str, arg: int) -> tuple[str, bool]:
    args_start, args_end, args = _find_subgraph_args(text)
    body = text[args_end:]
    non_dealloc_uses = [
        line
        for line in body.splitlines()
        if re.search(rf"%arg{arg}\b", line)
        and '"ttnn.deallocate"' not in line
    ]
    if non_dealloc_uses:
        return text, False

    kept_args: list[str] = []
    for spec in args:
        match = re.match(r"%arg(\d+):", spec)
        if not match:
            raise RuntimeError(f"could not parse function argument: {spec[:80]}")
        old = int(match.group(1))
        if old == arg:
            continue
        kept_args.append(spec)

    output: list[str] = []
    for line in text.splitlines():
        dealloc_arg = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
        if dealloc_arg and int(dealloc_arg.group(1)) == arg:
            continue
        output.append(line)
    text = "\n".join(output) + "\n"

    args_start, args_end, _ = _find_subgraph_args(text)
    text = text[:args_start] + ", ".join(kept_args) + text[args_end:]

    def replace_arg(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old == arg:
            raise RuntimeError(f"removed argument %arg{arg} is still referenced")
        if old > arg:
            return f"%arg{old - 1}"
        return f"%arg{old}"

    return _ARG_RE.sub(replace_arg, text), True


def _rewrite_lm_head_split(
    text: str,
    *,
    splits: int,
) -> tuple[str, LmHeadSplit]:
    if splits < 2:
        raise RuntimeError("--lm-head-splits must be at least 2")

    candidate_index = None
    candidate_match: re.Match[str] | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _LM_HEAD_MATMUL_FULL_RE.match(line)
        if match:
            candidate_index = index
            candidate_match = match
    if candidate_index is None or candidate_match is None:
        raise RuntimeError("could not find final LM-head vocab matmul to split")

    vocab = int(candidate_match.group("vocab"))
    hidden = int(candidate_match.group("hidden"))
    batch = int(candidate_match.group("batch"))
    dtype = candidate_match.group("dtype")
    if vocab % splits:
        raise RuntimeError(
            f"LM-head vocab size {vocab} is not divisible by {splits} splits"
        )
    rows_per_split = vocab // splits

    text, weight_layout, output_layout = _insert_lm_head_split_layouts(
        text,
        batch=batch,
        rows_per_split=rows_per_split,
        hidden=hidden,
        dtype=dtype,
    )
    args_start, args_end, args = _find_subgraph_args(text)
    first_split_arg = len(args)
    split_weight_type = (
        f"tensor<{rows_per_split}x{hidden}x{dtype}, {weight_layout}>"
    )
    args.extend(
        f"%arg{first_split_arg + index}: {split_weight_type}"
        for index in range(splits)
    )
    text = text[:args_start] + ", ".join(args) + text[args_end:]

    lines = text.splitlines()
    candidate_index = None
    candidate_match = None
    for index, line in enumerate(lines):
        match = _LM_HEAD_MATMUL_FULL_RE.match(line)
        if match:
            candidate_index = index
            candidate_match = match
    if candidate_index is None or candidate_match is None:
        raise RuntimeError("could not find LM-head matmul after signature rewrite")

    max_ssa = 0
    for match in re.finditer(r"^\s*%(\d+) =", text, re.MULTILINE):
        max_ssa = max(max_ssa, int(match.group(1)))
    next_ssa = max_ssa + 1

    indent = candidate_match.group("indent")
    result = candidate_match.group("result")
    input_value = candidate_match.group("input")
    source_arg = int(candidate_match.group("weight_arg"))
    attrs = candidate_match.group("attrs")
    input_type = candidate_match.group("input_type")
    output_type = candidate_match.group("output_type")
    split_output_type = (
        f"tensor<{batch}x{rows_per_split}x{dtype}, {output_layout}>"
    )

    split_values: list[int] = []
    replacement: list[str] = []
    for split_index in range(splits):
        value = next_ssa
        next_ssa += 1
        split_values.append(value)
        replacement.append(
            f'{indent}%{value} = "ttnn.matmul"(%{input_value}, '
            f"%arg{first_split_arg + split_index}) <{{{attrs}}}> : "
            f"({input_type}, {split_weight_type}) -> {split_output_type}"
        )
    replacement.append(
        f'{indent}"ttnn.deallocate"(%{input_value}) <{{force = false}}> : '
        f"({input_type}) -> ()"
    )
    operands = ", ".join(f"%{value}" for value in split_values)
    input_types = ", ".join(split_output_type for _ in split_values)
    replacement.append(
        f'{indent}%{result} = "ttnn.concat"({operands}) '
        f"<{{dim = 1 : si32}}> : ({input_types}) -> {output_type}"
    )
    for value in split_values:
        replacement.append(
            f'{indent}"ttnn.deallocate"(%{value}) <{{force = false}}> : '
            f"({split_output_type}) -> ()"
        )

    output: list[str] = []
    skipped_input_dealloc = 0
    for index, line in enumerate(lines):
        if index == candidate_index:
            output.extend(replacement)
            continue
        if (
            index == candidate_index + 1
            and f'"ttnn.deallocate"(%{input_value}) ' in line
        ):
            skipped_input_dealloc += 1
            continue
        output.append(line)
    if skipped_input_dealloc != 1:
        raise RuntimeError("expected to replace one LM-head input deallocate")

    text = "\n".join(output) + "\n"
    text, dropped_source_arg = _drop_unused_function_arg(text, source_arg)
    final_first_split_arg = (
        first_split_arg - 1
        if dropped_source_arg and source_arg < first_split_arg
        else first_split_arg
    )
    final_split_weight_type = split_weight_type
    if dropped_source_arg:
        final_split_weight_type = re.sub(
            rf"%arg{first_split_arg}\b", f"%arg{final_first_split_arg}", split_weight_type
        )

    return (
        text,
        LmHeadSplit(
            source_arg=source_arg,
            first_split_arg=final_first_split_arg,
            splits=splits,
            rows_per_split=rows_per_split,
            hidden=hidden,
            dtype=dtype,
            split_weight_type=final_split_weight_type,
            dropped_source_arg=dropped_source_arg,
        ),
    )


def _insert_lm_head_dram_program_config(
    text: str,
    *,
    in0_block_w: int,
    per_core_m: int,
    per_core_n: int,
    name: str,
) -> str:
    if name in text:
        return text
    attr_decl = (
        f"{name} = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<\n"
        f"  in0_block_w = {in0_block_w},\n"
        f"  per_core_m = {per_core_m},\n"
        f"  per_core_n = {per_core_n}\n"
        f">\n"
    )
    insert_at = text.find("module {")
    if insert_at < 0:
        raise RuntimeError("could not locate module body for program config insertion")
    return text[:insert_at] + attr_decl + text[insert_at:]


def _grid_for_num_cores(num_cores: int) -> tuple[int, int]:
    if num_cores % 8 == 0:
        return 8, num_cores // 8
    return num_cores, 1


def _insert_lm_head_mcast1d_program_config(
    text: str,
    *,
    grid_x: int,
    grid_y: int,
    in0_block_w: int,
    per_core_m: int,
    per_core_n: int,
    out_subblock_w: int,
    name: str,
) -> str:
    if name in text:
        return text
    attr_decl = (
        f"{name} = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<\n"
        f"  compute_with_storage_grid_size = #ttnn.core_coord<{grid_x}, {grid_y}>,\n"
        f"  in0_block_w = {in0_block_w},\n"
        f"  out_subblock_h = 1,\n"
        f"  out_subblock_w = {out_subblock_w},\n"
        f"  out_block_h = {per_core_m},\n"
        f"  out_block_w = {per_core_n},\n"
        f"  per_core_m = {per_core_m},\n"
        f"  per_core_n = {per_core_n},\n"
        f"  fuse_batch = true,\n"
        f"  mcast_in0 = true,\n"
        f"  gather_in0 = false,\n"
        f"  hop_cores = #ttnn.core_range_set<>,\n"
        f"  num_global_cb_receivers = 0,\n"
        f"  untilize_out = false\n"
        f">\n"
    )
    insert_at = text.find("module {")
    if insert_at < 0:
        raise RuntimeError("could not locate module body for program config insertion")
    return text[:insert_at] + attr_decl + text[insert_at:]


def _collect_lm_head_matmul_matches(
    lines: list[str],
) -> list[tuple[int, re.Match[str]]]:
    matches: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = _LM_HEAD_MATMUL_FULL_RE.match(line)
        if not match:
            continue
        if "matmul_program_config" in match.group("attrs"):
            continue
        if int(match.group("vocab")) < 16000:
            continue
        matches.append((index, match))
    return matches


def _rewrite_lm_head_mcast1d_program_config(
    text: str,
    *,
    num_cores: int,
) -> tuple[str, LmHeadProgramConfig]:
    if num_cores <= 0:
        raise RuntimeError("--lm-head-program-cores must be positive")

    matches = _collect_lm_head_matmul_matches(text.splitlines())
    if not matches:
        raise RuntimeError("could not find LM-head matmul(s) without program config")

    configs: dict[tuple[int, int, int, int], str] = {}
    per_core_n_values: list[int] = []
    grid_x, grid_y = _grid_for_num_cores(num_cores)
    for _index, match in matches:
        batch = int(match.group("batch"))
        hidden = int(match.group("hidden"))
        vocab = int(match.group("vocab"))
        dtype = match.group("dtype")
        if dtype != "bf16":
            raise RuntimeError("LM-head 1D program config expects bf16")
        if hidden % 32 != 0:
            raise RuntimeError("LM-head hidden size must be tile-aligned")
        k_tiles = hidden // 32
        in0_block_w = 2 if k_tiles % 2 == 0 else 1
        per_core_m = _ceil_div(batch, 32)
        per_core_n = _ceil_div(vocab, 32 * num_cores)
        out_subblock_w = _largest_divisor_up_to(per_core_n, 4)
        key = (in0_block_w, per_core_m, per_core_n, out_subblock_w)
        if key not in configs:
            configs[key] = f"#lm_head_mcast1d_pc{len(configs)}"
        per_core_n_values.append(per_core_n)

    for key, name in configs.items():
        in0_block_w, per_core_m, per_core_n, out_subblock_w = key
        text = _insert_lm_head_mcast1d_program_config(
            text,
            grid_x=grid_x,
            grid_y=grid_y,
            in0_block_w=in0_block_w,
            per_core_m=per_core_m,
            per_core_n=per_core_n,
            out_subblock_w=out_subblock_w,
            name=name,
        )

    lines = text.splitlines()
    rewrites = 0
    for index, line in enumerate(lines):
        match = _LM_HEAD_MATMUL_FULL_RE.match(line)
        if (
            not match
            or "matmul_program_config" in match.group("attrs")
            or int(match.group("vocab")) < 16000
        ):
            continue
        hidden = int(match.group("hidden"))
        vocab = int(match.group("vocab"))
        batch = int(match.group("batch"))
        k_tiles = hidden // 32
        in0_block_w = 2 if k_tiles % 2 == 0 else 1
        per_core_m = _ceil_div(batch, 32)
        per_core_n = _ceil_div(vocab, 32 * num_cores)
        out_subblock_w = _largest_divisor_up_to(per_core_n, 4)
        name = configs[(in0_block_w, per_core_m, per_core_n, out_subblock_w)]
        attrs = f"{match.group('attrs')}, matmul_program_config = {name}"
        lines[index] = (
            f'{match.group("indent")}%{match.group("result")} = '
            f'"ttnn.matmul"(%{match.group("input")}, '
            f"%arg{match.group('weight_arg')}) <{{{attrs}}}> : "
            f"({match.group('input_type')}, {match.group('weight_type')}) -> "
            f"{match.group('output_type')}"
        )
        rewrites += 1

    if rewrites != len(matches):
        raise RuntimeError(
            f"rewrote {rewrites} LM-head matmuls, expected {len(matches)}"
        )

    first_key = next(iter(configs))
    return (
        "\n".join(lines) + "\n",
        LmHeadProgramConfig(
            rewrites=rewrites,
            num_cores=num_cores,
            in0_block_w=first_key[0],
            per_core_m=first_key[1],
            per_core_n_values=sorted(set(per_core_n_values)),
        ),
    )


def _rewrite_lm_head_dram_sharded_program_config(
    text: str,
    *,
    num_cores: int,
) -> tuple[str, LmHeadProgramConfig]:
    if num_cores <= 0:
        raise RuntimeError("--lm-head-program-cores must be positive")

    lines = text.splitlines()
    matches = _collect_lm_head_matmul_matches(lines)
    if not matches:
        raise RuntimeError("could not find LM-head matmul(s) without program config")

    configs: dict[tuple[int, int, int], str] = {}
    per_core_n_values: list[int] = []
    for _index, match in matches:
        batch = int(match.group("batch"))
        hidden = int(match.group("hidden"))
        vocab = int(match.group("vocab"))
        dtype = match.group("dtype")
        if dtype != "bf16":
            raise RuntimeError("LM-head DRAM-sharded program config expects bf16")
        k_tiles_exact = hidden / (32 * num_cores)
        if hidden % (32 * num_cores) != 0:
            raise RuntimeError(
                "LM-head hidden size is not divisible by 32*num_cores: "
                f"{hidden} % {32 * num_cores} != 0"
            )
        k_tiles_per_core = int(k_tiles_exact)
        in0_block_w = _largest_divisor_up_to(k_tiles_per_core, 8)
        per_core_m = _ceil_div(batch, 32)
        per_core_n = _ceil_div(vocab, 32 * num_cores)
        key = (in0_block_w, per_core_m, per_core_n)
        if key not in configs:
            configs[key] = f"#lm_head_dram_pc{len(configs)}"
        per_core_n_values.append(per_core_n)

    for key, name in configs.items():
        in0_block_w, per_core_m, per_core_n = key
        text = _insert_lm_head_dram_program_config(
            text,
            in0_block_w=in0_block_w,
            per_core_m=per_core_m,
            per_core_n=per_core_n,
            name=name,
        )

    lines = text.splitlines()
    rewrites = 0
    for index, line in enumerate(lines):
        match = _LM_HEAD_MATMUL_FULL_RE.match(line)
        if (
            not match
            or "matmul_program_config" in match.group("attrs")
            or int(match.group("vocab")) < 16000
        ):
            continue
        hidden = int(match.group("hidden"))
        vocab = int(match.group("vocab"))
        batch = int(match.group("batch"))
        in0_block_w = _largest_divisor_up_to(hidden // (32 * num_cores), 8)
        per_core_m = _ceil_div(batch, 32)
        per_core_n = _ceil_div(vocab, 32 * num_cores)
        name = configs[(in0_block_w, per_core_m, per_core_n)]
        attrs = match.group("attrs")
        if attrs.strip():
            attrs = f"{attrs}, matmul_program_config = {name}"
        else:
            attrs = f"matmul_program_config = {name}"
        lines[index] = (
            f'{match.group("indent")}%{match.group("result")} = '
            f'"ttnn.matmul"(%{match.group("input")}, '
            f"%arg{match.group('weight_arg')}) <{{{attrs}}}> : "
            f"({match.group('input_type')}, {match.group('weight_type')}) -> "
            f"{match.group('output_type')}"
        )
        rewrites += 1

    if rewrites != len(matches):
        raise RuntimeError(
            f"rewrote {rewrites} LM-head matmuls, expected {len(matches)}"
        )

    first_key = next(iter(configs))
    return (
        "\n".join(lines) + "\n",
        LmHeadProgramConfig(
            rewrites=rewrites,
            num_cores=num_cores,
            in0_block_w=first_key[0],
            per_core_m=first_key[1],
            per_core_n_values=sorted(set(per_core_n_values)),
        ),
    )


def _rewrite_split_embedding_weight(
    text: str,
) -> tuple[str, SplitEmbeddingWeight | None]:
    args_start, args_end, args = _find_subgraph_args(text)
    split_arg = len(args)

    lines = text.splitlines()
    output: list[str] = []
    rewrite: SplitEmbeddingWeight | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 2 < len(lines)
            and '"ttnn.to_layout"' in line
            and "<{layout = #ttnn.layout<row_major>}>" in line
            and '"ttnn.embedding"' in lines[i + 1]
            and '"ttnn.deallocate"' in lines[i + 2]
        ):
            to_layout_result = _ssa_result(line)
            source_match = re.search(r'"ttnn\.to_layout"\(%arg(\d+)\)', line)
            embedding_source = (
                re.search(r'"ttnn\.embedding"\(%(\d+), %(\d+)\)', lines[i + 1])
                if to_layout_result
                else None
            )
            dealloc_source = (
                re.search(r'"ttnn\.deallocate"\(%(\d+)\)', lines[i + 2])
                if to_layout_result
                else None
            )
            type_match = re.search(
                r' : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)$', line
            )
            if (
                to_layout_result
                and source_match
                and embedding_source
                and dealloc_source
                and type_match
                and embedding_source.group(2) == to_layout_result
                and dealloc_source.group(1) == to_layout_result
            ):
                source_arg = int(source_match.group(1))
                split_type = type_match.group(2)
                if rewrite is not None:
                    raise RuntimeError(
                        "found more than one embedding weight row-major conversion"
                    )
                rewrite = SplitEmbeddingWeight(source_arg, split_arg, split_type)
                output.append(
                    re.sub(
                        rf"%{to_layout_result}\b",
                        f"%arg{split_arg}",
                        lines[i + 1],
                    )
                )
                output.append(
                    lines[i + 2]
                    .replace(f"%{to_layout_result}", f"%arg{split_arg}")
                    .replace(split_type, split_type)
                )
                i += 3
                continue
        output.append(line)
        i += 1

    if rewrite is None:
        return text, None

    rewritten = "\n".join(output) + "\n"
    args_start, args_end, args = _find_subgraph_args(rewritten)
    args.append(f"%arg{rewrite.split_arg}: {rewrite.split_type}")
    rewritten = rewritten[:args_start] + ", ".join(args) + rewritten[args_end:]
    return rewritten, rewrite


def _ssa_result(line: str) -> str | None:
    match = _ASSIGN_RE.match(line)
    return match.group("result") if match else None


class OpLine(NamedTuple):
    index: int
    indent: str
    result: str
    op: str
    operands: list[str]
    input_types: str
    result_type: str
    line: str


_OP_LINE_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "(?P<op>ttnn\.[^"]+)"'
    r'\((?P<operands>[^)]*)\)'
    r'(?: <\{(?P<attrs>.*)\}>)? : '
    r'\((?P<input_types>.*)\) -> (?P<result_type>tensor<[^>]+>)$'
)


def _parse_op_line(index: int, line: str) -> OpLine | None:
    match = _OP_LINE_RE.match(line)
    if not match:
        return None
    operands = [
        operand[1:]
        for operand in re.findall(r"%\d+", match.group("operands"))
    ]
    return OpLine(
        index=index,
        indent=match.group("indent"),
        result=match.group("result"),
        op=match.group("op"),
        operands=operands,
        input_types=match.group("input_types"),
        result_type=match.group("result_type"),
        line=line,
    )


def _attrs_contain_half_slice(line: str, *, low_half: bool) -> bool:
    if low_half:
        return (
            "begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]" in line
            and "ends = [32 : i32, 1 : i32," in line
            and "64 : i32]" in line
        )
    return (
        "begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32]" in line
        and "ends = [32 : i32, 1 : i32," in line
        and "128 : i32]" in line
    )


_RANK4_TENSOR_TYPE_RE = re.compile(
    r"^tensor<(?P<d0>\d+)x(?P<d1>\d+)x(?P<d2>\d+)x(?P<d3>\d+)x"
    r"(?P<dtype>[a-z0-9]+), (?P<layout>#[^>]+)>$"
)
_GENERAL_TENSOR_TYPE_RE = re.compile(
    r"^tensor<(?P<body>.+)x(?P<dtype>[a-z0-9]+), #[^>]+>$"
)


def _parse_rank4_tensor_type(type_text: str) -> tuple[list[int], str, str] | None:
    match = _RANK4_TENSOR_TYPE_RE.match(type_text)
    if not match:
        return None
    shape = [int(match.group(f"d{i}")) for i in range(4)]
    return shape, match.group("dtype"), match.group("layout")


def _format_rank4_tensor_type(shape: list[int], dtype: str, layout: str) -> str:
    return (
        f"tensor<{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x"
        f"{dtype}, {layout}>"
    )


def _i32_array(values: list[int]) -> str:
    return "[" + ", ".join(f"{value} : i32" for value in values) + "]"


def _shape_dtype_from_tensor_type(type_text: str) -> tuple[list[int], str]:
    match = _GENERAL_TENSOR_TYPE_RE.match(type_text)
    if not match:
        raise RuntimeError(f"could not parse tensor type: {type_text}")
    shape = [int(dim) for dim in match.group("body").split("x") if dim]
    dtype = match.group("dtype")
    dtype_name = {
        "bf16": "bfloat16",
        "f32": "float32",
        "si32": "int64",
        "ui32": "uint32",
    }.get(dtype, dtype)
    return shape, dtype_name


_CONCAT_HEADS_RESHAPE_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "ttnn\.reshape"\(%(?P<input>\d+)\) '
    r'<\{shape = \[32 : i32, (?P<hidden>\d+) : i32\]\}> : '
    r'\(tensor<1x32x(?P<heads>\d+)x(?P<head_dim>\d+)x'
    r'(?P<dtype>bf16), (?P<input_layout>#[^>]+)>\) -> '
    r'(?P<result_type>tensor<32x(?P=hidden)x(?P=dtype), (?P<result_layout>#[^>]+)>)$',
    re.MULTILINE,
)

_QKV_HEAD_SLICE_RE = re.compile(
    r'^(?P<indent>\s*)%(?P<result>\d+) = "ttnn\.slice_static"'
    r'\(%(?P<input>\d+)\) <\{begins = \[0 : i32, (?P<begin>\d+) : i32\], '
    r'ends = \[32 : i32, (?P<end>\d+) : i32\], '
    r'step = \[1 : i32, 1 : i32\]\}> : '
    r'\((?P<input_type>tensor<32x(?P<input_width>\d+)x(?P<dtype>bf16), '
    r'(?P<input_layout>#[^>]+)>)\) -> '
    r'(?P<result_type>tensor<32x(?P<width>\d+)x(?P=dtype), #[^>]+>)$'
)

_RESHAPE_RESULT_RE = re.compile(
    r'^\s*%\d+ = "ttnn\.reshape"\(%(?P<input>\d+)\) '
    r'<\{shape = \[(?P<shape>[^\]]+)\]\}> : '
    r'\((?P<input_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)$'
)


def _rewrite_concat_heads_decode(
    text: str, *, direct_sdpa_output: bool = False
) -> tuple[str, int]:
    matches = list(_CONCAT_HEADS_RESHAPE_RE.finditer(text))
    if not matches:
        return text, 0

    first = matches[0]
    hidden = int(first.group("hidden"))
    dtype = first.group("dtype")
    if hidden % 32 != 0:
        raise RuntimeError("decode concat-heads output hidden size must be tile aligned")

    layout = _next_layout_name(text)
    layout_decl = (
        f"{layout} = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> "
        f"(d0 * 32 + d1 * 32 + d2, d3), <1x1>, "
        f"memref<1x{hidden // 32}x!ttcore.tile<32x32, {dtype}>, #dram>, "
        "<interleaved>>"
    )
    insert_at = text.find("module {")
    if insert_at < 0:
        raise RuntimeError("could not locate module body for layout insertion")
    text = text[:insert_at] + layout_decl + "\n" + text[insert_at:]

    sharded_memory_config = None
    sharded_layout = None
    for line in text.splitlines():
        if (
            '"ttnn.to_memory_config"' not in line
            or "(tensor<1x32x8x128xbf16," not in line
            or "-> tensor<1x32x8x128xbf16," not in line
        ):
            continue
        memory_match = re.search(r"memory_config = (?P<memory_config>.+?)\}> : ", line)
        layout_match = re.search(
            r"-> tensor<1x32x8x128xbf16, (?P<layout>#[^>]+)>$", line
        )
        if memory_match and layout_match:
            sharded_memory_config = memory_match.group("memory_config")
            sharded_layout = layout_match.group("layout")
            break
    if sharded_memory_config is None or sharded_layout is None:
        raise RuntimeError("could not locate decode L1 height-sharded memory config")

    max_result = max(
        (int(match.group(1)) for match in re.finditer(r"%(\d+)\b", text)),
        default=-1,
    )
    next_result_id = max_result + 1

    def replace(match: re.Match[str]) -> str:
        nonlocal next_result_id
        heads = int(match.group("heads"))
        head_dim = int(match.group("head_dim"))
        hidden = int(match.group("hidden"))
        if heads * head_dim != hidden:
            return match.group(0)
        result = match.group("result")
        input_value = match.group("input")
        indent = match.group("indent")
        input_type = (
            f"tensor<1x32x{heads}x{head_dim}x{match.group('dtype')}, "
            f"{match.group('input_layout')}>"
        )
        concat_type = (
            f"tensor<1x1x32x{hidden}x{match.group('dtype')}, {layout}>"
        )
        sharded_type = (
            f"tensor<1x32x{heads}x{head_dim}x{match.group('dtype')}, "
            f"{sharded_layout}>"
        )
        if direct_sdpa_output:
            concat_result = str(next_result_id)
            next_result_id += 1
            return "\n".join(
                [
                    (
                        f'{indent}%{concat_result} = '
                        f'"ttnn.nlp_concat_heads_decode"'
                        f"(%{input_value}) <{{num_heads = {heads} : ui32}}> : "
                        f"({sharded_type}) -> {concat_type}"
                    ),
                    (
                        f'{indent}%{result} = "ttnn.reshape"(%{concat_result}) '
                        f"<{{shape = [32 : i32, {hidden} : i32]}}> : "
                        f"({concat_type}) -> {match.group('result_type')}"
                    ),
                    (
                        f'{indent}"ttnn.deallocate"(%{concat_result}) '
                        f"<{{force = false}}> : ({concat_type}) -> ()"
                    ),
                ]
            )

        sharded_result = str(next_result_id)
        concat_result = str(next_result_id + 1)
        next_result_id += 2
        return "\n".join(
            [
                (
                    f'{indent}%{sharded_result} = "ttnn.to_memory_config"'
                    f"(%{input_value}) <{{memory_config = "
                    f"{sharded_memory_config}}}> : "
                    f"({input_type}) -> {sharded_type}"
                ),
                (
                    f'{indent}%{concat_result} = "ttnn.nlp_concat_heads_decode"'
                    f"(%{sharded_result}) <{{num_heads = {heads} : ui32}}> : "
                    f"({sharded_type}) -> {concat_type}"
                ),
                (
                    f'{indent}"ttnn.deallocate"(%{sharded_result}) '
                    f"<{{force = false}}> : ({sharded_type}) -> ()"
                ),
                (
                    f'{indent}%{result} = "ttnn.reshape"(%{concat_result}) '
                    f"<{{shape = [32 : i32, {hidden} : i32]}}> : "
                    f"({concat_type}) -> {match.group('result_type')}"
                ),
                (
                    f'{indent}"ttnn.deallocate"(%{concat_result}) '
                    f"<{{force = false}}> : ({concat_type}) -> ()"
                ),
            ]
        )

    rewritten = _CONCAT_HEADS_RESHAPE_RE.sub(replace, text)
    if direct_sdpa_output:
        sdpa_result_types: dict[str, tuple[str, str]] = {}
        for match in matches:
            input_value = match.group("input")
            old_type = (
                f"tensor<1x32x{match.group('heads')}x{match.group('head_dim')}x"
                f"{match.group('dtype')}, {match.group('input_layout')}>"
            )
            new_type = (
                f"tensor<1x32x{match.group('heads')}x{match.group('head_dim')}x"
                f"{match.group('dtype')}, {sharded_layout}>"
            )
            sdpa_result_types[input_value] = (old_type, new_type)

        output_lines: list[str] = []
        rewritten_sdpa = 0
        rewritten_deallocs = 0
        for line in rewritten.splitlines():
            sdpa_match = re.match(
                r'(?P<prefix>\s*%(?P<result>\d+) = '
                r'"ttnn\.scaled_dot_product_attention_decode"\([^)]*\) <\{)'
                r'(?P<attrs>.*)(?P<suffix>\}> : \(.*\) -> )'
                r'(?P<result_type>tensor<[^>]+>)$',
                line,
            )
            if sdpa_match and sdpa_match.group("result") in sdpa_result_types:
                old_type, new_type = sdpa_result_types[sdpa_match.group("result")]
                if sdpa_match.group("result_type") != old_type:
                    raise RuntimeError(
                        "SDPA result type did not match expected concat-heads input"
                    )
                attrs = sdpa_match.group("attrs")
                if "memory_config =" not in attrs:
                    attrs = f"memory_config = {sharded_memory_config}, {attrs}"
                line = (
                    f"{sdpa_match.group('prefix')}{attrs}"
                    f"{sdpa_match.group('suffix')}{new_type}"
                )
                rewritten_sdpa += 1

            for value, (old_type, new_type) in sdpa_result_types.items():
                dealloc_pattern = (
                    f'"ttnn.deallocate"(%{value}) <{{force = false}}> : '
                    f"({old_type}) -> ()"
                )
                if dealloc_pattern in line:
                    line = line.replace(f"({old_type})", f"({new_type})")
                    rewritten_deallocs += 1
            output_lines.append(line)

        if rewritten_sdpa != len(matches):
            raise RuntimeError(
                f"rewrote {rewritten_sdpa} SDPA outputs, expected {len(matches)}"
            )
        if rewritten_deallocs != len(matches):
            raise RuntimeError(
                f"rewrote {rewritten_deallocs} SDPA deallocs, expected {len(matches)}"
            )
        rewritten = "\n".join(output_lines) + "\n"
    count = len(matches)
    return rewritten, count


def _find_reshape_result_type(
    lines: list[str], value: str, shape: list[int]
) -> str:
    shape_text = ", ".join(f"{dim} : i32" for dim in shape)
    for line in lines:
        match = _RESHAPE_RESULT_RE.match(line)
        if (
            match
            and match.group("input") == value
            and match.group("shape") == shape_text
        ):
            return match.group("result_type")
    raise RuntimeError(
        f"could not find reshape of %{value} to [{shape_text}]"
    )


def _rewrite_create_qkv_heads_decode(text: str) -> tuple[str, int]:
    """Use TTNN's decode QKV-head creation op after the packed QKV matmul.

    This intentionally preserves the existing downstream logical layouts. The
    specialized op replaces only the generated rank-2 Q/K/V slices; later
    reshape/rope/cache code is left structurally identical so correctness risk
    stays bounded for the first candidate.
    """

    layout = _next_layout_name(text)
    lines = text.splitlines()
    max_result = max(
        (int(match.group(1)) for match in re.finditer(r"%(\d+)\b", text)),
        default=-1,
    )
    next_result_id = max_result + 1

    replacements: dict[str, tuple[str, str]] = {}
    output_lines: list[str] = []
    rewrites = 0
    inserted_layout = False
    i = 0
    while i < len(lines):
        if i + 3 >= len(lines):
            output_lines.append(lines[i])
            i += 1
            continue

        q = _QKV_HEAD_SLICE_RE.match(lines[i])
        k = _QKV_HEAD_SLICE_RE.match(lines[i + 1])
        v = _QKV_HEAD_SLICE_RE.match(lines[i + 2])
        if not (q and k and v):
            output_lines.append(lines[i])
            i += 1
            continue

        matmul_result = q.group("input")
        dealloc = re.match(
            rf'^(?P<indent>\s*)"ttnn\.deallocate"\(%{matmul_result}\) '
            rf'<\{{force = false\}}> : '
            rf'\({re.escape(q.group("input_type"))}\) -> \(\)$',
            lines[i + 3],
        )
        if not dealloc:
            output_lines.append(lines[i])
            i += 1
            continue

        if (
            k.group("input") != matmul_result
            or v.group("input") != matmul_result
            or q.group("begin") != "0"
            or q.group("end") != "3072"
            or k.group("begin") != "3072"
            or k.group("end") != "4096"
            or v.group("begin") != "4096"
            or v.group("end") != "5120"
            or q.group("input_width") != "5120"
            or k.group("input_type") != q.group("input_type")
            or v.group("input_type") != q.group("input_type")
        ):
            output_lines.append(lines[i])
            i += 1
            continue

        q_value = q.group("result")
        k_value = k.group("result")
        v_value = v.group("result")
        q_type = _find_reshape_result_type(lines, q_value, [1, 32, 24, 128])
        k_type = _find_reshape_result_type(lines, k_value, [1, 32, 8, 128])
        v_type = _find_reshape_result_type(lines, v_value, [1, 32, 8, 128])

        hidden = int(q.group("input_width"))
        dtype = q.group("dtype")
        if hidden % 32 != 0:
            raise RuntimeError("QKV hidden width must be tile aligned")
        qkv_type = (
            f"tensor<1x1x32x{hidden}x{dtype}, {layout}>"
        )
        if not inserted_layout:
            layout_decl = (
                f"{layout} = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> "
                f"(d0 * 32 + d1 * 32 + d2, d3), <1x1>, "
                f"memref<1x{hidden // 32}x!ttcore.tile<32x32, {dtype}>, "
                "#dram>, <interleaved>>"
            )
            module_at = text.find("module {")
            if module_at < 0:
                raise RuntimeError("could not locate module body")
            text = text[:module_at] + layout_decl + "\n" + text[module_at:]
            lines = text.splitlines()
            output_lines = text[:module_at].splitlines()
            i = len(output_lines)
            inserted_layout = True
            continue

        qkv_reshape = str(next_result_id)
        next_result_id += 1
        indent = q.group("indent")
        output_lines.extend(
            [
                (
                    f'{indent}%{qkv_reshape} = "ttnn.reshape"'
                    f"(%{matmul_result}) "
                    f"<{{shape = [1 : i32, 1 : i32, 32 : i32, "
                    f"{hidden} : i32]}}> : ({q.group('input_type')}) -> "
                    f"{qkv_type}"
                ),
                (
                    f'{indent}%{q_value}, %{k_value}, %{v_value} = '
                    f'"ttnn.nlp_create_qkv_heads_decode"'
                    f"(%{qkv_reshape}) <{{num_heads = 24 : ui32, "
                    f"num_kv_heads = 8 : ui32}}> : ({qkv_type}) -> "
                    f"({q_type}, {k_type}, {v_type})"
                ),
                (
                    f'{indent}"ttnn.deallocate"(%{qkv_reshape}) '
                    f"<{{force = false}}> : ({qkv_type}) -> ()"
                ),
                lines[i + 3],
            ]
        )
        replacements[q_value] = (q.group("result_type"), q_type)
        replacements[k_value] = (k.group("result_type"), k_type)
        replacements[v_value] = (v.group("result_type"), v_type)
        rewrites += 1
        i += 4

    if inserted_layout and rewrites == 0:
        raise RuntimeError("inserted layout but rewrote no QKV head patterns")

    rewritten_lines: list[str] = []
    for line in output_lines:
        for value, (old_type, new_type) in replacements.items():
            if re.search(rf"%{value}\b", line):
                line = line.replace(f"({old_type})", f"({new_type})")
        rewritten_lines.append(line)

    if rewrites == 0:
        raise RuntimeError("found no packed-QKV head creation patterns")
    return "\n".join(rewritten_lines) + "\n", rewrites


def _rewrite_rope_fusion(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    ops: dict[str, OpLine] = {}
    for index, line in enumerate(lines):
        op = _parse_op_line(index, line)
        if op:
            ops[op.result] = op

    next_result_id = max((int(result) for result in ops), default=-1) + 1

    def fresh_result_id() -> str:
        nonlocal next_result_id
        result = str(next_result_id)
        next_result_id += 1
        return result

    def get_op(value: str, op_name: str | None = None) -> OpLine | None:
        op = ops.get(value)
        if op is None:
            return None
        if op_name is not None and op.op != op_name:
            return None
        return op

    def match_mul(value: str) -> tuple[str, str] | None:
        mul = get_op(value, "ttnn.multiply")
        if mul is None or len(mul.operands) != 2:
            return None
        lhs, rhs = mul.operands
        # RoPE cos/sin tensors in this graph are broadcastable
        # [1,1,1,head_dim] tensors. The other operand is the activation.
        lhs_type = ops.get(lhs).result_type if lhs in ops else ""
        rhs_type = ops.get(rhs).result_type if rhs in ops else ""
        if re.match(r"tensor<1x1x1x\d+xbf16,", lhs_type):
            return rhs, lhs
        if re.match(r"tensor<1x1x1x\d+xbf16,", rhs_type):
            return lhs, rhs
        return None

    def match_rotated(value: str) -> tuple[str, set[str]] | None:
        reshape = get_op(value, "ttnn.reshape")
        if reshape is None or len(reshape.operands) != 1:
            return None
        concat = get_op(reshape.operands[0], "ttnn.concat")
        if concat is None or len(concat.operands) != 2 or "dim = 3 : si32" not in concat.line:
            return None
        neg = get_op(concat.operands[0], "ttnn.neg")
        low_slice = get_op(concat.operands[1], "ttnn.slice_static")
        if neg is None or low_slice is None or len(neg.operands) != 1:
            return None
        high_slice = get_op(neg.operands[0], "ttnn.slice_static")
        if high_slice is None:
            return None
        if (
            len(low_slice.operands) != 1
            or len(high_slice.operands) != 1
            or low_slice.operands[0] != high_slice.operands[0]
        ):
            return None
        if not _attrs_contain_half_slice(low_slice.line, low_half=True):
            return None
        if not _attrs_contain_half_slice(high_slice.line, low_half=False):
            return None
        base_reshape = get_op(low_slice.operands[0], "ttnn.reshape")
        if base_reshape is None or len(base_reshape.operands) != 1:
            return None
        remove = {
            reshape.result,
            concat.result,
            neg.result,
            low_slice.result,
            high_slice.result,
            base_reshape.result,
        }
        return base_reshape.operands[0], remove

    replacements: dict[int, list[str]] = {}
    remove_values: set[str] = set()
    move_dealloc_values: dict[str, str] = {}

    for add in list(ops.values()):
        if add.op != "ttnn.add" or len(add.operands) != 2:
            continue
        matched = None
        for cos_mul_value, sin_mul_value in (
            (add.operands[0], add.operands[1]),
            (add.operands[1], add.operands[0]),
        ):
            cos_mul = match_mul(cos_mul_value)
            sin_mul = match_mul(sin_mul_value)
            if cos_mul is None or sin_mul is None:
                continue
            x_value, cos_value = cos_mul
            rotated_value, sin_value = sin_mul
            rotated = match_rotated(rotated_value)
            x_op = get_op(x_value, "ttnn.reshape")
            if rotated is None or x_op is None or len(x_op.operands) != 1:
                continue
            source_value, rotated_remove = rotated
            if x_op.operands[0] != source_value:
                continue
            matched = (x_value, cos_value, sin_value, rotated_remove)
            break
        if matched is None:
            continue

        x_value, cos_value, sin_value, rotated_remove = matched
        x_type = ops[x_value].result_type
        cos_type = ops[cos_value].result_type
        sin_type = ops[sin_value].result_type
        replacement_lines = []
        parsed_result_type = _parse_rank4_tensor_type(add.result_type)
        if parsed_result_type is None:
            replacement_lines.append(
                (
                    f'{add.indent}%{add.result} = "ttnn.rotary_embedding"'
                    f"(%{x_value}, %{cos_value}, %{sin_value}) : "
                    f"({x_type}, {cos_type}, {sin_type}) -> {add.result_type}"
                )
            )
        else:
            result_shape, result_dtype, result_layout = parsed_result_type
            padded_shape = list(result_shape)
            tile_height = 32
            original_seq_len = padded_shape[-2]
            padded_seq_len = (
                (original_seq_len + tile_height - 1) // tile_height
            ) * tile_height
            if padded_seq_len == original_seq_len:
                replacement_lines.append(
                    (
                        f'{add.indent}%{add.result} = "ttnn.rotary_embedding"'
                        f"(%{x_value}, %{cos_value}, %{sin_value}) : "
                        f"({x_type}, {cos_type}, {sin_type}) -> "
                        f"{add.result_type}"
                    )
                )
            else:
                padded_shape[-2] = padded_seq_len
                padded_type = _format_rank4_tensor_type(
                    padded_shape, result_dtype, result_layout
                )
                padded_value = fresh_result_id()
                begins = [0] * len(result_shape)
                ends = list(padded_shape)
                ends[-2] = original_seq_len
                steps = [1] * len(result_shape)
                replacement_lines.extend(
                    [
                        (
                            f'{add.indent}%{padded_value} = '
                            f'"ttnn.rotary_embedding"'
                            f"(%{x_value}, %{cos_value}, %{sin_value}) : "
                            f"({x_type}, {cos_type}, {sin_type}) -> "
                            f"{padded_type}"
                        ),
                        (
                            f'{add.indent}%{add.result} = '
                            f'"ttnn.slice_static"(%{padded_value}) '
                            f"<{{begins = {_i32_array(begins)}, "
                            f"ends = {_i32_array(ends)}, "
                            f"step = {_i32_array(steps)}}}> : "
                            f"({padded_type}) -> {add.result_type}"
                        ),
                    ]
                )
        # RotaryEmbedding can expose padded storage through the following
        # slice workaround, so keep the fused intermediates alive for the
        # submit instead of adding eager deallocations here.
        replacements[add.index] = replacement_lines
        remove_values.update(rotated_remove)
        remove_values.add(cos_mul_value)
        remove_values.add(sin_mul_value)
        move_dealloc_values[x_value] = x_type

    if not replacements:
        raise RuntimeError("found no generated decode RoPE patterns to fuse")

    output: list[str] = []
    removed_op_count = 0
    removed_dealloc_count = 0
    for index, line in enumerate(lines):
        if index in replacements:
            output.extend(replacements[index])
            continue
        op = _parse_op_line(index, line)
        if op and op.result in remove_values:
            removed_op_count += 1
            continue
        dealloc = re.match(r'\s*"ttnn\.deallocate"\(%(\d+)\) ', line)
        if dealloc and (
            dealloc.group(1) in remove_values
            or dealloc.group(1) in move_dealloc_values
        ):
            removed_dealloc_count += 1
            continue
        output.append(line)

    rewrite_count = len(replacements)
    expected_removed = rewrite_count * 8
    if removed_op_count != expected_removed:
        raise RuntimeError(
            "unexpected number of RoPE helper ops removed: "
            f"removed={removed_op_count}, expected={expected_removed}"
        )
    if rewrite_count % 2 != 0:
        raise RuntimeError(f"expected Q/K RoPE pairs, got {rewrite_count}")
    return "\n".join(output) + "\n", rewrite_count


def _rewrite_argmax_tile(text: str, *, singlecore: bool = False) -> tuple[str, int]:
    lines = text.splitlines()
    rewrites = 0
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 3 < len(lines)
            and '"ttnn.to_layout"' in line
            and "<{layout = #ttnn.layout<row_major>}>" in line
            and '"ttnn.deallocate"' in lines[i + 1]
            and '"ttnn.argmax"' in lines[i + 2]
        ):
            to_layout_result = _ssa_result(line)
            source_match = re.search(r'"ttnn\.to_layout"\(%(\d+)\)', line)
            if to_layout_result and source_match:
                source = source_match.group(1)
                dealloc_source = re.search(r'"ttnn\.deallocate"\(%(\d+)\)', lines[i + 1])
                argmax_source = re.search(r'"ttnn\.argmax"\(%(\d+)\)', lines[i + 2])
                if (
                    dealloc_source
                    and argmax_source
                    and dealloc_source.group(1) == source
                    and argmax_source.group(1) == to_layout_result
                ):
                    source_type = None
                    result_type = None
                    type_match = re.search(r' : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)$', line)
                    if type_match:
                        source_type = type_match.group(1)
                        result_type = type_match.group(2)
                    if source_type and result_type:
                        argmax_line = lines[i + 2]
                        argmax_line = re.sub(
                            rf'"ttnn\.argmax"\(%{to_layout_result}\)',
                            f'"ttnn.argmax"(%{source})',
                            argmax_line,
                            count=1,
                        )
                        argmax_line = argmax_line.replace(result_type, source_type, 1)
                        if singlecore:
                            argmax_line = argmax_line.replace(
                                "use_multicore = true",
                                "use_multicore = false",
                                1,
                            )
                        output.append(argmax_line)
                        # Reuse the skipped to_layout result deallocate as the
                        # source logits deallocate after argmax.
                        output.append(
                            lines[i + 3]
                            .replace(f"%{to_layout_result}", f"%{source}")
                            .replace(result_type, source_type)
                        )
                        rewrites += 1
                        i += 4
                        continue
        output.append(line)
        i += 1
    if rewrites != 1:
        raise RuntimeError(
            f"expected to rewrite one LM-head row-major argmax path, got {rewrites}"
        )
    return "\n".join(output) + "\n", rewrites


def _rewrite_lm_head_topk(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    used_ids = {
        int(match.group(1))
        for line in lines
        if (match := re.match(r"\s*%(\d+)\b", line))
    }
    next_id = max(used_ids, default=-1) + 1

    def fresh_id() -> str:
        nonlocal next_id
        while next_id in used_ids:
            next_id += 1
        value = str(next_id)
        used_ids.add(next_id)
        next_id += 1
        return value

    rewrites = 0
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 7 < len(lines)
            and '"ttnn.to_layout"' in line
            and "<{layout = #ttnn.layout<row_major>}>" in line
            and '"ttnn.deallocate"' in lines[i + 1]
            and '"ttnn.argmax"' in lines[i + 2]
            and '"ttnn.deallocate"' in lines[i + 3]
            and '"ttnn.to_layout"' in lines[i + 4]
            and "<{layout = #ttnn.layout<tile>}>" in lines[i + 4]
            and '"ttnn.deallocate"' in lines[i + 5]
            and '"ttnn.typecast"' in lines[i + 6]
            and '"ttnn.deallocate"' in lines[i + 7]
        ):
            to_layout_result = _ssa_result(line)
            argmax_result = _ssa_result(lines[i + 2])
            tiled_indices = _ssa_result(lines[i + 4])
            source_match = re.search(r'"ttnn\.to_layout"\(%(\d+)\)', line)
            source_type_match = re.search(
                r' : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)$', line
            )
            typecast_match = re.search(
                rf'"ttnn\.typecast"\(%{tiled_indices}\).*'
                r' : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)$',
                lines[i + 6],
            ) if tiled_indices else None
            if (
                to_layout_result
                and argmax_result
                and tiled_indices
                and source_match
                and source_type_match
                and typecast_match
            ):
                source = source_match.group(1)
                source_type = source_type_match.group(1)
                tiled_index_type = typecast_match.group(1)
                value_type = re.sub(
                    r"xui32, #[^>]+>$",
                    "xbf16, #ttnn_layout5>",
                    tiled_index_type,
                    count=1,
                )
                if value_type == tiled_index_type:
                    raise RuntimeError(
                        "could not derive topk values type from index type"
                    )
                values = fresh_id()
                output.append(
                    f'{line[: len(line) - len(line.lstrip())]}%{values}, '
                    f'%{tiled_indices} = "ttnn.topk"(%{source}) '
                    f"<{{dim = 1 : i32, k = 1 : i32, largest = true, "
                    f"sorted = false}}> : ({source_type}) -> "
                    f"({value_type}, {tiled_index_type})"
                )
                output.append(
                    lines[i + 1]
                    .replace(f"%{source}", f"%{source}")
                    .replace(source_type_match.group(2), source_type)
                )
                output.append(
                    f'{line[: len(line) - len(line.lstrip())]}'
                    f'"ttnn.deallocate"(%{values}) <{{force = false}}> : '
                    f"({value_type}) -> ()"
                )
                output.append(lines[i + 6])
                rewrites += 1
                i += 8
                continue
        output.append(line)
        i += 1

    if rewrites != 1:
        raise RuntimeError(
            f"expected to rewrite one LM-head argmax path to topk, got {rewrites}"
        )
    return "\n".join(output) + "\n", rewrites


def _subgraph0_line_range(lines: list[str]) -> tuple[int, int]:
    start = None
    for index, line in enumerate(lines):
        if "func.func @subgraph0(" in line:
            start = index
            break
    if start is None:
        raise RuntimeError("could not find func.func @subgraph0")
    saw_return = False
    for index in range(start + 1, len(lines)):
        if re.match(r"\s*return\b", lines[index]):
            saw_return = True
        if saw_return and re.match(r"\s*}\s*$", lines[index]):
            return start, index
    raise RuntimeError("could not find end of func.func @subgraph0")


def _rewrite_fold_identity_mul(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    start, end = _subgraph0_line_range(lines)
    one_values: set[str] = set()
    for line in lines[start:end]:
        match = re.match(
            r'\s*%(\d+) = ttcore\.load_cached\(@subgraph0_const_eval_(1|5), '
            r"\[\]\) : \(\) -> tensor<1x1(?:x1x1)?xbf16, #[^>]+>",
            line,
        )
        if match:
            one_values.add(match.group(1))

    if not one_values:
        return text, 0

    multiply_lines: set[int] = set()
    result_to_source: dict[str, tuple[str, str, str]] = {}
    defer_source_dealloc: set[str] = set()
    for index in range(start, end):
        line = lines[index]
        op = _parse_op_line(index, line)
        if op is None or op.op != "ttnn.multiply" or len(op.operands) != 2:
            continue
        lhs, rhs = op.operands
        if lhs in one_values and rhs not in one_values:
            source = rhs
        elif rhs in one_values and lhs not in one_values:
            source = lhs
        else:
            continue

        input_types = re.findall(r"tensor<[^>]+>", op.input_types)
        source_type = ""
        for operand, operand_type in zip(op.operands, input_types):
            if operand == source:
                source_type = operand_type
                break
        if not source_type or source_type != op.result_type:
            continue

        multiply_lines.add(index)
        result_to_source[op.result] = (source, source_type, op.result_type)
        defer_source_dealloc.add(source)

    if not multiply_lines:
        return text, 0

    output: list[str] = []
    removed_loads = 0
    removed_source_deallocs = 0
    replaced_result_deallocs = 0
    removed_one_deallocs = 0
    for index, line in enumerate(lines):
        if start <= index < end:
            load = re.match(r"\s*%(\d+) = ttcore\.load_cached", line)
            if load and load.group(1) in one_values:
                removed_loads += 1
                continue
            if index in multiply_lines:
                continue
            dealloc = re.match(r'(?P<indent>\s*)"ttnn\.deallocate"\(%(\d+)\) ', line)
            if dealloc:
                value = dealloc.group(2)
                if value in defer_source_dealloc:
                    removed_source_deallocs += 1
                    continue
                if value in result_to_source:
                    source, source_type, result_type = result_to_source[value]
                    output.append(
                        line.replace(f"%{value}", f"%{source}", 1).replace(
                            result_type, source_type, 1
                        )
                    )
                    replaced_result_deallocs += 1
                    continue
                if value in one_values:
                    removed_one_deallocs += 1
                    continue

            for result, (source, _source_type, _result_type) in result_to_source.items():
                line = re.sub(rf"%{result}\b", f"%{source}", line)
        output.append(line)

    if replaced_result_deallocs != len(result_to_source):
        raise RuntimeError(
            "did not replace every folded multiply result deallocate: "
            f"replaced={replaced_result_deallocs}, expected={len(result_to_source)}"
        )
    if removed_source_deallocs != len(result_to_source):
        raise RuntimeError(
            "did not remove every early source deallocate for folded multiply: "
            f"removed={removed_source_deallocs}, expected={len(result_to_source)}"
        )
    if removed_loads != len(one_values):
        raise RuntimeError(
            f"removed {removed_loads} ones loads, expected {len(one_values)}"
        )
    if removed_one_deallocs != len(one_values):
        raise RuntimeError(
            f"removed {removed_one_deallocs} ones deallocs, expected {len(one_values)}"
        )
    return "\n".join(output) + "\n", len(result_to_source)


def _copy_artifacts(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    for phase in ("prefill", "decode"):
        src_phase = src / phase
        dst_phase = dst / phase
        dst_phase.mkdir(parents=True, exist_ok=True)
        for path in src_phase.iterdir():
            target = dst_phase / path.name
            if path.name == "weights.bin":
                try:
                    os.link(path, target)
                except OSError:
                    target.symlink_to(path)
            elif path.is_file():
                shutil.copy2(path, target)


def _rewrite_decode_artifacts(root: Path, packs: list[QKVPack]) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    pack_by_q = {pack.q_arg: pack for pack in packs}
    remove_slots = {pack.k_arg for pack in packs} | {pack.v_arg for pack in packs}

    new_roles: list[dict] = []
    for old_slot, entry in enumerate(roles):
        if old_slot in remove_slots:
            continue
        entry = dict(entry)
        if old_slot in pack_by_q:
            pack = pack_by_q[old_slot]
            k_entry = roles[pack.k_arg]
            v_entry = roles[pack.v_arg]
            if (
                entry.get("role") != "weight"
                or k_entry.get("role") != "weight"
                or v_entry.get("role") != "weight"
            ):
                raise RuntimeError(f"QKV slots at {old_slot} are not all weights")
            q_end = entry["weight_offset"] + entry["weight_nbytes"]
            k_end = k_entry["weight_offset"] + k_entry["weight_nbytes"]
            if q_end != k_entry["weight_offset"] or k_end != v_entry["weight_offset"]:
                raise RuntimeError(f"QKV weight bytes are not contiguous at slot {old_slot}")
            if entry["dtype"] != k_entry["dtype"] or entry["dtype"] != v_entry["dtype"]:
                raise RuntimeError(f"QKV dtypes do not match at slot {old_slot}")
            entry["shape"] = pack.packed_shape
            entry["weight_nbytes"] = (
                entry["weight_nbytes"]
                + k_entry["weight_nbytes"]
                + v_entry["weight_nbytes"]
            )
            entry["placeholder"] = f"{entry.get('placeholder', 'qkv')}_packed_qkv"
        entry["slot"] = len(new_roles)
        new_roles.append(entry)

    roles_path.write_text(
        json.dumps(new_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in new_roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in new_roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(new_roles)
    summary["num_weight_slots"] = sum(1 for entry in new_roles if entry["role"] == "weight")
    summary["num_runtime_slots"] = sum(1 for entry in new_roles if entry["role"] != "weight")
    summary["qkv_packed"] = True
    summary["qkv_packed_layers"] = len(packs)
    summary["weight_bytes"] = sum(
        int(entry.get("weight_nbytes", 0))
        for entry in new_roles
        if entry["role"] == "weight"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _rewrite_mlp_gate_up_artifacts(
    root: Path, packs: list[MLPGateUpPack]
) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    pack_by_gate = {pack.gate_arg: pack for pack in packs}
    remove_slots = {pack.up_arg for pack in packs}

    new_roles: list[dict] = []
    for old_slot, entry in enumerate(roles):
        if old_slot in remove_slots:
            continue
        entry = dict(entry)
        if old_slot in pack_by_gate:
            pack = pack_by_gate[old_slot]
            up_entry = roles[pack.up_arg]
            if entry.get("role") != "weight" or up_entry.get("role") != "weight":
                raise RuntimeError(f"MLP gate/up slots at {old_slot} are not weights")
            gate_end = entry["weight_offset"] + entry["weight_nbytes"]
            if gate_end != up_entry["weight_offset"]:
                raise RuntimeError(
                    f"MLP gate/up weight bytes are not contiguous at slot {old_slot}"
                )
            if entry["dtype"] != up_entry["dtype"]:
                raise RuntimeError(f"MLP gate/up dtypes do not match at slot {old_slot}")
            entry["shape"] = pack.packed_weight_shape
            entry["weight_nbytes"] += up_entry["weight_nbytes"]
            entry["placeholder"] = (
                f"{entry.get('placeholder', 'mlp')}_packed_mlp_gate_up"
            )
        entry["slot"] = len(new_roles)
        new_roles.append(entry)

    roles_path.write_text(
        json.dumps(new_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in new_roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in new_roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(new_roles)
    summary["num_weight_slots"] = sum(
        1 for entry in new_roles if entry["role"] == "weight"
    )
    summary["num_runtime_slots"] = sum(
        1 for entry in new_roles if entry["role"] != "weight"
    )
    summary["mlp_gate_up_packed"] = True
    summary["mlp_gate_up_packed_layers"] = len(packs)
    summary["weight_bytes"] = sum(
        int(entry.get("weight_nbytes", 0))
        for entry in new_roles
        if entry["role"] == "weight"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _rewrite_split_embedding_artifacts(
    root: Path, split: SplitEmbeddingWeight
) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if split.source_arg >= len(roles):
        raise RuntimeError(
            f"split embedding source slot {split.source_arg} is out of range"
        )
    if split.split_arg != len(roles):
        raise RuntimeError(
            "split embedding slot must be appended after existing roles; "
            f"got split_arg={split.split_arg}, roles={len(roles)}"
        )
    source = dict(roles[split.source_arg])
    if source.get("role") != "weight":
        raise RuntimeError(
            f"split embedding source slot {split.source_arg} is not a weight"
        )
    source["slot"] = split.split_arg
    source["placeholder"] = (
        f"{source.get('placeholder', 'embedding')}_row_major_embedding"
    )
    source["split_embedding_source_slot"] = split.source_arg
    roles.append(source)

    roles_path.write_text(
        json.dumps(roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(roles)
    summary["num_weight_slots"] = sum(1 for entry in roles if entry["role"] == "weight")
    summary["num_runtime_slots"] = sum(1 for entry in roles if entry["role"] != "weight")
    summary["weight_bytes"] = sum(
        int(entry.get("weight_nbytes", 0))
        for entry in roles
        if entry["role"] == "weight"
    )
    summary["split_embedding_weight"] = True
    summary["split_embedding_source_slot"] = split.source_arg
    summary["split_embedding_slot"] = split.split_arg
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _dtype_nbytes(dtype: str) -> int:
    dtype_nbytes = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "int64": 8,
        "uint32": 4,
        "bf16": 2,
        "f32": 4,
        "ui32": 4,
    }
    if dtype not in dtype_nbytes:
        raise RuntimeError(f"unknown tensor dtype byte width: {dtype}")
    return dtype_nbytes[dtype]


def _rewrite_lm_head_split_artifacts(root: Path, split: LmHeadSplit) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if split.source_arg >= len(roles):
        raise RuntimeError(
            f"LM-head source slot {split.source_arg} is out of range"
        )

    source = dict(roles[split.source_arg])
    if source.get("role") != "weight":
        raise RuntimeError(f"LM-head source slot {split.source_arg} is not a weight")
    source_shape = source.get("shape")
    if source_shape != [split.rows_per_split * split.splits, split.hidden]:
        raise RuntimeError(
            "LM-head source shape does not match split request: "
            f"{source_shape} vs "
            f"{[split.rows_per_split * split.splits, split.hidden]}"
        )

    dtype = source.get("dtype")
    row_nbytes = split.hidden * _dtype_nbytes(dtype)
    split_nbytes = split.rows_per_split * row_nbytes
    source_offset = int(source["weight_offset"])
    source_nbytes = int(source["weight_nbytes"])
    if split_nbytes * split.splits != source_nbytes:
        raise RuntimeError(
            "LM-head source bytes do not match split request: "
            f"{split_nbytes * split.splits} vs {source_nbytes}"
        )

    if split.dropped_source_arg:
        roles = [
            dict(entry)
            for index, entry in enumerate(roles)
            if index != split.source_arg
        ]
        for new_slot, entry in enumerate(roles):
            old_slot = int(entry["slot"])
            entry["slot"] = new_slot
            if "split_embedding_source_slot" in entry:
                if int(entry["split_embedding_source_slot"]) == split.source_arg:
                    entry["split_embedding_source_slot_removed"] = split.source_arg
                    del entry["split_embedding_source_slot"]
                elif int(entry["split_embedding_source_slot"]) > split.source_arg:
                    entry["split_embedding_source_slot"] = (
                        int(entry["split_embedding_source_slot"]) - 1
                    )
            if "lm_head_split_source_slot" in entry:
                if int(entry["lm_head_split_source_slot"]) > split.source_arg:
                    entry["lm_head_split_source_slot"] = (
                        int(entry["lm_head_split_source_slot"]) - 1
                    )
            if old_slot > split.source_arg and "merged_cache_position_slots" in entry:
                entry["merged_cache_position_slots"] = [
                    slot - 1 if slot > split.source_arg else slot
                    for slot in entry["merged_cache_position_slots"]
                ]

    if split.first_split_arg != len(roles):
        raise RuntimeError(
            "LM-head split slots must be appended after existing roles; "
            f"got first_split_arg={split.first_split_arg}, roles={len(roles)}"
        )

    for index in range(split.splits):
        entry = dict(source)
        entry["slot"] = split.first_split_arg + index
        entry["shape"] = [split.rows_per_split, split.hidden]
        entry["weight_offset"] = source_offset + split_nbytes * index
        entry["weight_nbytes"] = split_nbytes
        entry["placeholder"] = (
            f"{source.get('placeholder', 'lm_head')}_lm_head_split_{index}"
        )
        if split.dropped_source_arg:
            entry["lm_head_split_source_slot_removed"] = split.source_arg
        else:
            entry["lm_head_split_source_slot"] = split.source_arg
        entry["lm_head_split_index"] = index
        entry["lm_head_split_count"] = split.splits
        roles.append(entry)

    roles_path.write_text(
        json.dumps(roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(roles)
    summary["num_weight_slots"] = sum(1 for entry in roles if entry["role"] == "weight")
    summary["num_runtime_slots"] = sum(1 for entry in roles if entry["role"] != "weight")
    summary["weight_bytes"] = sum(
        int(entry.get("weight_nbytes", 0))
        for entry in roles
        if entry["role"] == "weight"
    )
    summary["lm_head_split"] = True
    if split.dropped_source_arg:
        summary["lm_head_split_source_slot_removed"] = split.source_arg
    else:
        summary["lm_head_split_source_slot"] = split.source_arg
    summary["lm_head_split_first_slot"] = split.first_split_arg
    summary["lm_head_split_count"] = split.splits
    summary["lm_head_split_rows"] = split.rows_per_split
    if split.dropped_source_arg:
        if "split_embedding_slot" in summary and summary["split_embedding_slot"] > split.source_arg:
            summary["split_embedding_slot"] -= 1
        if (
            "split_embedding_source_slot" in summary
            and summary["split_embedding_source_slot"] == split.source_arg
        ):
            summary["split_embedding_source_slot_removed"] = split.source_arg
            del summary["split_embedding_source_slot"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _remove_static_weight_input_deallocs(text: str, root: Path) -> tuple[str, int]:
    roles_path = root / "decode" / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    static_slots = {
        int(entry["slot"])
        for entry in roles
        if entry.get("role") in ("weight", "inv_freq")
    }
    output: list[str] = []
    removed = 0
    for line in text.splitlines():
        match = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
        if match and int(match.group(1)) in static_slots:
            removed += 1
            continue
        output.append(line)
    if removed == 0:
        raise RuntimeError("removed no static weight/inv_freq input dealloc ops")
    return "\n".join(output) + "\n", removed


def _rewrite_native_u32_token_io(text: str, root: Path) -> tuple[str, int]:
    input_pattern = re.compile(
        r'(?P<indent>\s*)%(?P<cast>\d+) = "ttnn\.typecast"\(%arg1\) '
        r'<\{dtype = #ttcore\.supportedDataTypes<u32>\}> : '
        r'\((?P<input_type>tensor<[^>]+si32[^>]*>)\) -> '
        r'(?P<tile_u32_type>tensor<[^>]+ui32[^>]*>)\n'
        r'(?P=indent)"ttnn\.deallocate"\(%arg1\) <\{force = false\}> : '
        r'\((?P=input_type)\) -> \(\)\n'
        r'(?P=indent)%(?P<row>\d+) = "ttnn\.to_layout"\(%(?P=cast)\) '
        r'<\{layout = #ttnn\.layout<row_major>\}> : '
        r'\((?P=tile_u32_type)\) -> (?P<row_u32_type>tensor<[^>]+ui32[^>]*>)\n'
        r'(?P=indent)"ttnn\.deallocate"\(%(?P=cast)\) <\{force = false\}> : '
        r'\((?P=tile_u32_type)\) -> \(\)\n'
    )
    input_match = input_pattern.search(text)
    if not input_match:
        raise RuntimeError("could not find decode token-id input conversion")

    input_type = input_match.group("input_type")
    row_u32_type = input_match.group("row_u32_type")
    row_value = input_match.group("row")

    text = (
        text[: input_match.start()]
        + text[input_match.end() :]
    )
    text = text.replace(f"%arg1: {input_type}", f"%arg1: {row_u32_type}", 1)
    text = text.replace(f'"ttnn.embedding"(%{row_value},', '"ttnn.embedding"(%arg1,', 1)

    row_dealloc_pattern = re.compile(
        rf'\n\s*"ttnn\.deallocate"\(%{row_value}\) <\{{force = false\}}> : '
        rf'\({re.escape(row_u32_type)}\) -> \(\)'
    )
    text, removed_row_deallocs = row_dealloc_pattern.subn("", text, count=1)
    if removed_row_deallocs != 1:
        raise RuntimeError("could not remove token embedding input deallocate")

    # Update the forward function result type. The input type occurs once in the
    # argument list and once as the final token-id result; the argument has
    # already been changed above, so this remaining occurrence is the result.
    result_suffix = f"{input_type}) attributes {{tt.function_type = \"forward_device\"}}"
    if result_suffix not in text:
        raise RuntimeError("could not find decode token-id result type")
    text = text.replace(
        result_suffix,
        f"{row_u32_type}) attributes {{tt.function_type = \"forward_device\"}}",
        1,
    )

    output_pattern = re.compile(
        r'(?P<argmax_line>(?P<indent>\s*)%(?P<argmax>\d+) = '
        r'"ttnn\.argmax"\([^\n]+\) [^\n]+ -> (?P<argmax_type>tensor<[^>]+ui32[^>]*>)\n)'
        r'(?P<logits_dealloc>(?P=indent)"ttnn\.deallocate"\(%\d+\) '
        r'<\{force = false\}> : \(tensor<[^>]+bf16[^>]*>\) -> \(\)\n)'
        r'(?P=indent)%(?P<tile>\d+) = "ttnn\.to_layout"\(%(?P=argmax)\) '
        r'<\{layout = #ttnn\.layout<tile>\}> : '
        r'\((?P=argmax_type)\) -> (?P<tile_type>tensor<[^>]+ui32[^>]*>)\n'
        r'(?P=indent)"ttnn\.deallocate"\(%(?P=argmax)\) <\{force = false\}> : '
        r'\((?P=argmax_type)\) -> \(\)\n'
        r'(?P=indent)%(?P<cast>\d+) = "ttnn\.typecast"\(%(?P=tile)\) '
        r'<\{dtype = #ttcore\.supportedDataTypes<si32>\}> : '
        r'\((?P=tile_type)\) -> (?P<cast_type>tensor<[^>]+si32[^>]*>)\n'
        r'(?P=indent)"ttnn\.deallocate"\(%(?P=tile)\) <\{force = false\}> : '
        r'\((?P=tile_type)\) -> \(\)\n'
    )
    output_match = output_pattern.search(text)
    if not output_match:
        raise RuntimeError("could not find decode token-id output conversion")
    if output_match.group("argmax_type") != row_u32_type:
        raise RuntimeError(
            "decode argmax token type does not match rewritten input type: "
            f"{output_match.group('argmax_type')} vs {row_u32_type}"
        )

    text = (
        text[: output_match.start()]
        + output_match.group("argmax_line")
        + output_match.group("logits_dealloc")
        + text[output_match.end() :]
    )
    return_rewrites = 0
    rewritten_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("return ") and (
            f"%{output_match.group('cast')} :" in line
        ):
            line = line.replace(
                f"%{output_match.group('cast')} :",
                f"%{output_match.group('argmax')} :",
                1,
            )
            type_pos = line.rfind(output_match.group("cast_type"))
            if type_pos < 0:
                raise RuntimeError("could not find token-id return type")
            line = (
                line[:type_pos]
                + row_u32_type
                + line[type_pos + len(output_match.group("cast_type")) :]
            )
            return_rewrites += 1
        rewritten_lines.append(line)
    if return_rewrites != 1:
        raise RuntimeError(
            f"expected to rewrite one token-id return, got {return_rewrites}"
        )
    text = "\n".join(rewritten_lines) + "\n"

    summary_path = root / "decode" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["native_u32_token_io"] = True
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return text, 6


def _remove_input_deallocs_by_role(
    text: str, root: Path, roles_to_keep: set[str]
) -> tuple[str, int]:
    roles_path = root / "decode" / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    slots = {
        int(entry["slot"])
        for entry in roles
        if entry.get("role") in roles_to_keep
    }
    output: list[str] = []
    removed = 0
    for line in text.splitlines():
        match = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
        if match and int(match.group(1)) in slots:
            removed += 1
            continue
        output.append(line)
    if removed == 0:
        raise RuntimeError(
            "removed no input dealloc ops for roles: "
            + ", ".join(sorted(roles_to_keep))
        )
    return "\n".join(output) + "\n", removed


def _rewrite_precompute_sdpa_mask(
    text: str,
) -> tuple[str, PrecomputedSdpaInputs, int]:
    subgraph_start = text.find("func.func @subgraph0(")
    if subgraph_start < 0:
        raise RuntimeError("missing func.func @subgraph0")
    subgraph_end = text.find("\n      func.func private @cpu_hoisted", subgraph_start)
    if subgraph_end < 0:
        subgraph_end = text.find("\n      llvm.func", subgraph_start)
    if subgraph_end < 0:
        raise RuntimeError("could not locate end of @subgraph0")
    prefix = text[:subgraph_start]
    subgraph = text[subgraph_start:subgraph_end]
    suffix = text[subgraph_end:]

    lines = subgraph.splitlines()
    ops: dict[str, OpLine] = {}
    users: dict[str, set[str]] = {}
    for index, line in enumerate(lines):
        op = _parse_op_line(index, line)
        if not op:
            continue
        ops[op.result] = op
        for operand in op.operands:
            users.setdefault(operand, set()).add(op.result)

    sdpa = next(
        (op for op in ops.values() if op.op == "ttnn.scaled_dot_product_attention_decode"),
        None,
    )
    if sdpa is None:
        raise RuntimeError("could not find decode SDPA op with mask/position")
    sdpa_operands_match = re.search(r'"ttnn\.scaled_dot_product_attention_decode"\(([^)]*)\)', sdpa.line)
    if not sdpa_operands_match:
        raise RuntimeError("could not parse decode SDPA operands")
    sdpa_operands = [
        operand[1:] for operand in re.findall(r"%(?:arg)?\d+", sdpa_operands_match.group(1))
    ]
    if len(sdpa_operands) < 5:
        raise RuntimeError("decode SDPA op has too few operands")
    mask_value = sdpa_operands[3]
    position_value = sdpa_operands[4]
    if mask_value.startswith("arg") or position_value.startswith("arg"):
        raise RuntimeError("SDPA mask/position are already graph inputs")
    mask_op = ops.get(mask_value)
    position_op = ops.get(position_value)
    if mask_op is None or position_op is None:
        raise RuntimeError("SDPA mask/position values are not produced by TTNN ops")

    args_start, args_end, args = _find_subgraph_args(subgraph)
    mask_arg = len(args)
    position_arg = mask_arg + 1
    args.append(f"%arg{mask_arg}: {mask_op.result_type}")
    args.append(f"%arg{position_arg}: {position_op.result_type}")

    remove_values = {mask_value, position_value}
    queue = [mask_value, position_value]
    while queue:
        value = queue.pop()
        op = ops.get(value)
        if op is None:
            continue
        for operand in op.operands:
            operand_op = ops.get(operand)
            if operand_op is None or operand in remove_values:
                continue
            operand_users = users.get(operand, set())
            if operand_users and operand_users.issubset(remove_values):
                remove_values.add(operand)
                queue.append(operand)

    removed_arg_candidates: set[int] = set()
    for value in remove_values:
        for arg in re.findall(r"%arg(\d+)\b", ops[value].line):
            removed_arg_candidates.add(int(arg))

    kept_lines: list[str] = []
    removed_ops = 0
    removed_deallocs = 0
    for line in lines:
        result = _ssa_result(line)
        if result in remove_values:
            removed_ops += 1
            continue
        dealloc_value = re.match(r'\s*"ttnn\.deallocate"\(%(\d+)\) ', line)
        if dealloc_value and dealloc_value.group(1) in remove_values:
            removed_deallocs += 1
            continue
        kept_lines.append(line)

    filtered = "\n".join(kept_lines) + "\n"
    args_start, args_end, _ = _find_subgraph_args(filtered)
    filtered = (
        filtered[:args_start] + ", ".join(args) + filtered[args_end:]
    )
    filtered = re.sub(rf"%{mask_value}\b", f"%arg{mask_arg}", filtered)
    filtered = re.sub(
        rf"%{position_value}\b", f"%arg{position_arg}", filtered
    )

    # Remove deallocs for inputs that are now unused because their only
    # consumers were folded into the runtime-provided mask/position inputs.
    body_start = filtered.find(" attributes {tt.function_type = \"forward_device\"}")
    removable_arg_deallocs: set[int] = set()
    for arg in removed_arg_candidates:
        if arg in (mask_arg, position_arg):
            continue
        body = filtered[body_start:] if body_start >= 0 else filtered
        non_dealloc_uses = [
            line
            for line in body.splitlines()
            if re.search(rf"%arg{arg}\b", line)
            and '"ttnn.deallocate"' not in line
        ]
        if not non_dealloc_uses:
            removable_arg_deallocs.add(arg)

    if removable_arg_deallocs:
        output: list[str] = []
        for line in filtered.splitlines():
            dealloc_arg = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
            if dealloc_arg and int(dealloc_arg.group(1)) in removable_arg_deallocs:
                removed_deallocs += 1
                continue
            output.append(line)
        filtered = "\n".join(output) + "\n"

    # The removed mask subgraph can leave cached scalar/range constants with
    # only a matching deallocate. Drop those orphan loads as well.
    while True:
        candidate_loads: dict[str, str] = {}
        for line in filtered.splitlines():
            match = re.match(r"\s*%(\d+) = ttcore\.load_cached\(", line)
            if match:
                candidate_loads[match.group(1)] = line
        orphan_loads = set()
        for value in candidate_loads:
            body = filtered[body_start:] if body_start >= 0 else filtered
            non_dealloc_uses = [
                line
                for line in body.splitlines()
                if re.search(rf"%{value}\b", line)
                and not re.search(rf"%{value}\s*=", line)
                and '"ttnn.deallocate"' not in line
            ]
            if not non_dealloc_uses:
                orphan_loads.add(value)
        if not orphan_loads:
            break
        output = []
        for line in filtered.splitlines():
            result = re.match(r"\s*%(\d+) = ttcore\.load_cached\(", line)
            dealloc = re.match(r'\s*"ttnn\.deallocate"\(%(\d+)\) ', line)
            if result and result.group(1) in orphan_loads:
                removed_ops += 1
                continue
            if dealloc and dealloc.group(1) in orphan_loads:
                removed_deallocs += 1
                continue
            output.append(line)
        filtered = "\n".join(output) + "\n"

    if removed_ops < 6:
        raise RuntimeError(
            "unexpectedly small SDPA precompute rewrite; "
            f"removed_ops={removed_ops}"
        )
    return (
        prefix + filtered + suffix,
        PrecomputedSdpaInputs(
            mask_arg=mask_arg,
            mask_type=mask_op.result_type,
            position_arg=position_arg,
            position_type=position_op.result_type,
        ),
        removed_ops + removed_deallocs,
    )


def _rewrite_precomputed_sdpa_artifacts(
    root: Path, sdpa: PrecomputedSdpaInputs
) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if sdpa.mask_arg != len(roles):
        raise RuntimeError(
            "precomputed SDPA mask slot must be appended after existing roles; "
            f"got mask_arg={sdpa.mask_arg}, roles={len(roles)}"
        )
    mask_shape, mask_dtype = _shape_dtype_from_tensor_type(sdpa.mask_type)
    pos_shape, pos_dtype = _shape_dtype_from_tensor_type(sdpa.position_type)
    roles.append(
        {
            "slot": sdpa.mask_arg,
            "placeholder": "runtime_sdpa_mask",
            "role": "sdpa_mask",
            "shape": mask_shape,
            "dtype": mask_dtype,
        }
    )
    roles.append(
        {
            "slot": sdpa.position_arg,
            "placeholder": "runtime_sdpa_position",
            "role": "sdpa_position",
            "shape": pos_shape,
            "dtype": pos_dtype,
        }
    )

    roles_path.write_text(
        json.dumps(roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(roles)
    summary["num_weight_slots"] = sum(
        1 for entry in roles if entry["role"] == "weight"
    )
    summary["num_runtime_slots"] = sum(
        1 for entry in roles if entry["role"] != "weight"
    )
    summary["precomputed_sdpa_mask"] = True
    summary["sdpa_mask_slot"] = sdpa.mask_arg
    summary["sdpa_position_slot"] = sdpa.position_arg
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _subgraph_slice(text: str) -> tuple[str, str, str]:
    subgraph_start = text.find("func.func @subgraph0(")
    if subgraph_start < 0:
        raise RuntimeError("missing func.func @subgraph0")
    subgraph_end = text.find("\n      func.func private @cpu_hoisted", subgraph_start)
    if subgraph_end < 0:
        subgraph_end = text.find("\n      llvm.func", subgraph_start)
    if subgraph_end < 0:
        raise RuntimeError("could not locate end of @subgraph0")
    return text[:subgraph_start], text[subgraph_start:subgraph_end], text[subgraph_end:]


def _cleanup_orphan_load_cached(text: str) -> tuple[str, int]:
    body_start = text.find(" attributes {tt.function_type = \"forward_device\"}")
    removed = 0
    while True:
        candidate_loads: set[str] = set()
        for line in text.splitlines():
            match = re.match(r"\s*%(\d+) = ttcore\.load_cached\(", line)
            if match:
                candidate_loads.add(match.group(1))
        orphan_loads = set()
        body = text[body_start:] if body_start >= 0 else text
        for value in candidate_loads:
            non_dealloc_uses = [
                line
                for line in body.splitlines()
                if re.search(rf"%{value}\b", line)
                and not re.search(rf"%{value}\s*=", line)
                and '"ttnn.deallocate"' not in line
            ]
            if not non_dealloc_uses:
                orphan_loads.add(value)
        if not orphan_loads:
            break
        output = []
        for line in text.splitlines():
            result = re.match(r"\s*%(\d+) = ttcore\.load_cached\(", line)
            dealloc = re.match(r'\s*"ttnn\.deallocate"\(%(\d+)\) ', line)
            if result and result.group(1) in orphan_loads:
                removed += 1
                continue
            if dealloc and dealloc.group(1) in orphan_loads:
                removed += 1
                continue
            output.append(line)
        text = "\n".join(output) + "\n"
    return text, removed


def _rewrite_precompute_rope(text: str) -> tuple[str, PrecomputedRopeInputs, int]:
    prefix, subgraph, suffix = _subgraph_slice(text)
    lines = subgraph.splitlines()
    ops: dict[str, OpLine] = {}
    users: dict[str, set[str]] = {}
    for index, line in enumerate(lines):
        op = _parse_op_line(index, line)
        if not op:
            continue
        ops[op.result] = op
        for operand in op.operands:
            users.setdefault(operand, set()).add(op.result)

    cos_value = None
    sin_value = None
    for op in ops.values():
        if op.op != "ttnn.typecast" or len(op.operands) != 1:
            continue
        parsed = _parse_rank4_tensor_type(op.result_type)
        if parsed is None:
            continue
        shape, dtype, _layout = parsed
        if shape[:3] != [1, 1, 1] or dtype != "bf16":
            continue
        mul = ops.get(op.operands[0])
        if mul is None or mul.op != "ttnn.multiply":
            continue
        producer_ops = {ops[o].op for o in mul.operands if o in ops}
        if "ttnn.cos" in producer_ops:
            cos_value = op.result
        if "ttnn.sin" in producer_ops:
            sin_value = op.result

    if cos_value is None or sin_value is None:
        raise RuntimeError("could not find generated RoPE cos/sin typecasts")
    cos_op = ops[cos_value]
    sin_op = ops[sin_value]

    args_start, args_end, args = _find_subgraph_args(subgraph)
    cos_arg = len(args)
    sin_arg = cos_arg + 1
    args.append(f"%arg{cos_arg}: {cos_op.result_type}")
    args.append(f"%arg{sin_arg}: {sin_op.result_type}")

    remove_values = {cos_value, sin_value}
    queue = [cos_value, sin_value]
    while queue:
        value = queue.pop()
        op = ops.get(value)
        if op is None:
            continue
        for operand in op.operands:
            operand_op = ops.get(operand)
            if operand_op is None or operand in remove_values:
                continue
            operand_users = users.get(operand, set())
            if operand_users and operand_users.issubset(remove_values):
                remove_values.add(operand)
                queue.append(operand)

    kept_lines: list[str] = []
    removed = 0
    for line in lines:
        result = _ssa_result(line)
        if result in remove_values:
            removed += 1
            continue
        dealloc_value = re.match(r'\s*"ttnn\.deallocate"\(%(\d+)\) ', line)
        if dealloc_value and dealloc_value.group(1) in remove_values:
            removed += 1
            continue
        kept_lines.append(line)

    filtered = "\n".join(kept_lines) + "\n"
    args_start, args_end, _ = _find_subgraph_args(filtered)
    filtered = filtered[:args_start] + ", ".join(args) + filtered[args_end:]
    filtered = re.sub(rf"%{cos_value}\b", f"%arg{cos_arg}", filtered)
    filtered = re.sub(rf"%{sin_value}\b", f"%arg{sin_arg}", filtered)
    filtered, orphan_removed = _cleanup_orphan_load_cached(filtered)
    removed += orphan_removed

    if removed < 12:
        raise RuntimeError(
            "unexpectedly small RoPE precompute rewrite; "
            f"removed={removed}"
        )

    return (
        prefix + filtered + suffix,
        PrecomputedRopeInputs(
            cos_arg=cos_arg,
            cos_type=cos_op.result_type,
            sin_arg=sin_arg,
            sin_type=sin_op.result_type,
        ),
        removed,
    )


def _rewrite_precomputed_rope_artifacts(
    root: Path, rope: PrecomputedRopeInputs
) -> None:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if rope.cos_arg != len(roles):
        raise RuntimeError(
            "precomputed RoPE cos slot must be appended after existing roles; "
            f"got cos_arg={rope.cos_arg}, roles={len(roles)}"
        )
    cos_shape, cos_dtype = _shape_dtype_from_tensor_type(rope.cos_type)
    sin_shape, sin_dtype = _shape_dtype_from_tensor_type(rope.sin_type)
    roles.append(
        {
            "slot": rope.cos_arg,
            "placeholder": "runtime_rope_cos",
            "role": "rope_cos",
            "shape": cos_shape,
            "dtype": cos_dtype,
        }
    )
    roles.append(
        {
            "slot": rope.sin_arg,
            "placeholder": "runtime_rope_sin",
            "role": "rope_sin",
            "shape": sin_shape,
            "dtype": sin_dtype,
        }
    )

    roles_path.write_text(
        json.dumps(roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(roles)
    summary["num_weight_slots"] = sum(
        1 for entry in roles if entry["role"] == "weight"
    )
    summary["num_runtime_slots"] = sum(
        1 for entry in roles if entry["role"] != "weight"
    )
    summary["precomputed_rope"] = True
    summary["rope_cos_slot"] = rope.cos_arg
    summary["rope_sin_slot"] = rope.sin_arg
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _rewrite_merge_cache_position_inputs(
    text: str, root: Path
) -> tuple[str, int]:
    decode_dir = root / "decode"
    roles_path = decode_dir / "slot_roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    cache_slots = [
        int(entry["slot"])
        for entry in roles
        if entry.get("role") == "cache_position"
    ]
    if len(cache_slots) <= 1:
        raise RuntimeError("not enough cache_position slots to merge")

    keep_slot = cache_slots[0]
    remove_slots = set(cache_slots[1:])

    args_start, args_end, args = _find_subgraph_args(text)
    kept_args: list[str] = []
    old_to_new: dict[int, int] = {}
    for spec in args:
        match = re.match(r"%arg(\d+):", spec)
        if not match:
            raise RuntimeError(f"could not parse function argument: {spec[:80]}")
        old_idx = int(match.group(1))
        if old_idx in remove_slots:
            continue
        old_to_new[old_idx] = len(kept_args)
        kept_args.append(spec)

    if keep_slot not in old_to_new:
        raise RuntimeError("kept cache_position slot is missing from signature")
    keep_arg_after_renumber = old_to_new[keep_slot]

    filtered_lines: list[str] = []
    removed_deallocs = 0
    for line in text.splitlines():
        dealloc_arg = re.match(r'\s*"ttnn\.deallocate"\(%arg(\d+)\) ', line)
        if dealloc_arg and int(dealloc_arg.group(1)) in set(cache_slots):
            removed_deallocs += 1
            continue
        filtered_lines.append(line)
    filtered = "\n".join(filtered_lines) + "\n"
    filtered = filtered[:args_start] + ", ".join(kept_args) + filtered[args_end:]

    def replace_arg(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old in remove_slots:
            return f"%arg{keep_arg_after_renumber}"
        if old not in old_to_new:
            raise RuntimeError(f"removed argument %arg{old} still referenced")
        return f"%arg{old_to_new[old]}"

    text = _ARG_RE.sub(replace_arg, filtered)

    new_roles: list[dict] = []
    for old_slot, entry in enumerate(roles):
        if old_slot in remove_slots:
            continue
        entry = dict(entry)
        entry["slot"] = len(new_roles)
        if old_slot == keep_slot:
            entry["merged_cache_position_slots"] = cache_slots
        new_roles.append(entry)

    roles_path.write_text(
        json.dumps(new_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (decode_dir / "shapes.json").write_text(
        json.dumps([entry["shape"] for entry in new_roles]), encoding="utf-8"
    )
    (decode_dir / "dtypes.json").write_text(
        json.dumps([entry["dtype"] for entry in new_roles]), encoding="utf-8"
    )

    summary_path = decode_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_slots"] = len(new_roles)
    summary["num_weight_slots"] = sum(
        1 for entry in new_roles if entry["role"] == "weight"
    )
    summary["num_runtime_slots"] = sum(
        1 for entry in new_roles if entry["role"] != "weight"
    )
    summary["cache_position_merged_slots"] = cache_slots
    summary["cache_position_removed_slots"] = len(remove_slots)
    summary["cache_position_removed_deallocs"] = removed_deallocs
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return text, len(remove_slots)


def main() -> int:
    args = _parse_args()
    if not (
        args.pack_qkv
        or args.pack_mlp_gate_up
        or args.argmax_tile
        or args.argmax_tile_singlecore
        or args.lm_head_topk
        or args.fold_identity_mul
        or args.merge_cache_position_inputs
        or args.keep_cache_position_inputs
        or args.precompute_sdpa_mask
        or args.precompute_rope
        or args.lm_head_hifi2
        or args.lm_head_no_fp32_accum
        or args.split_lm_head
        or args.lm_head_dram_sharded_program_config
        or args.lm_head_mcast1d_program_config
        or args.split_embedding_weight
        or args.keep_static_weight_inputs
        or args.native_u32_token_io
        or args.fuse_create_qkv_heads_decode
        or args.fuse_concat_heads_decode
        or args.fuse_concat_heads_decode_sdpa_output
        or args.fuse_rope
    ):
        args.pack_qkv = True

    text = args.input_ttnn_mlir.read_text(encoding="utf-8")
    rewrites: list[str] = []

    packs: list[QKVPack] = []
    if args.pack_qkv:
        packs = _extract_packs(text)
        text, old_to_new = _rewrite_signature(text, packs)
        text = _rewrite_body(text, packs)
        text = _renumber_args(text, old_to_new)
        rewrites.append(f"packed_qkv_layers={len(packs)}")

    mlp_packs: list[MLPGateUpPack] = []
    if args.pack_mlp_gate_up:
        mlp_packs = _extract_mlp_gate_up_packs(text)
        text, mlp_packs = _attach_mlp_pack_layouts(text, mlp_packs)
        text, old_to_new = _rewrite_mlp_signature(text, mlp_packs)
        text = _rewrite_mlp_body(text, mlp_packs)
        text = _renumber_args(text, old_to_new)
        rewrites.append(f"packed_mlp_gate_up_layers={len(mlp_packs)}")

    if args.argmax_tile:
        text, count = _rewrite_argmax_tile(text)
        rewrites.append(f"argmax_tile={count}")

    if args.argmax_tile_singlecore:
        text, count = _rewrite_argmax_tile(text, singlecore=True)
        rewrites.append(f"argmax_tile_singlecore={count}")

    if args.lm_head_topk:
        text, count = _rewrite_lm_head_topk(text)
        rewrites.append(f"lm_head_topk={count}")

    if args.fold_identity_mul:
        text, count = _rewrite_fold_identity_mul(text)
        rewrites.append(f"fold_identity_mul={count}")

    if args.lm_head_hifi2:
        text, count = _rewrite_lm_head_hifi2(text)
        rewrites.append(f"lm_head_hifi2={count}")

    if args.lm_head_no_fp32_accum:
        text, count = _rewrite_lm_head_no_fp32_accum(text)
        rewrites.append(f"lm_head_no_fp32_accum={count}")

    if args.fuse_rope:
        text, count = _rewrite_rope_fusion(text)
        rewrites.append(f"fuse_rope={count}")

    split: SplitEmbeddingWeight | None = None
    if args.split_embedding_weight:
        text, split = _rewrite_split_embedding_weight(text)
        if split:
            rewrites.append(
                "split_embedding_weight="
                f"{split.source_arg}->{split.split_arg}"
            )
        else:
            rewrites.append("split_embedding_weight=0")

    lm_head_split: LmHeadSplit | None = None
    if args.split_lm_head:
        text, lm_head_split = _rewrite_lm_head_split(
            text, splits=args.lm_head_splits
        )
        rewrites.append(
            "split_lm_head="
            f"{lm_head_split.splits}x{lm_head_split.rows_per_split}"
        )

    if args.lm_head_dram_sharded_program_config:
        text, config = _rewrite_lm_head_dram_sharded_program_config(
            text, num_cores=args.lm_head_program_cores
        )
        rewrites.append(
            "lm_head_dram_sharded_program_config="
            f"{config.rewrites}xcores{config.num_cores}:"
            f"in0w{config.in0_block_w}:m{config.per_core_m}:"
            f"n{','.join(str(value) for value in config.per_core_n_values)}"
        )

    if args.lm_head_mcast1d_program_config:
        text, config = _rewrite_lm_head_mcast1d_program_config(
            text, num_cores=args.lm_head_program_cores
        )
        rewrites.append(
            "lm_head_mcast1d_program_config="
            f"{config.rewrites}xcores{config.num_cores}:"
            f"in0w{config.in0_block_w}:m{config.per_core_m}:"
            f"n{','.join(str(value) for value in config.per_core_n_values)}"
        )

    sdpa: PrecomputedSdpaInputs | None = None
    if args.precompute_sdpa_mask:
        text, sdpa, count = _rewrite_precompute_sdpa_mask(text)
        rewrites.append(f"precompute_sdpa_mask={count}")

    rope: PrecomputedRopeInputs | None = None
    if args.precompute_rope:
        text, rope, count = _rewrite_precompute_rope(text)
        rewrites.append(f"precompute_rope={count}")

    _copy_artifacts(args.input_artifacts, args.output_artifacts)
    if packs:
        _rewrite_decode_artifacts(args.output_artifacts, packs)
    if mlp_packs:
        _rewrite_mlp_gate_up_artifacts(args.output_artifacts, mlp_packs)
    if split:
        _rewrite_split_embedding_artifacts(args.output_artifacts, split)
    if lm_head_split:
        _rewrite_lm_head_split_artifacts(args.output_artifacts, lm_head_split)
    if sdpa:
        _rewrite_precomputed_sdpa_artifacts(args.output_artifacts, sdpa)
    if rope:
        _rewrite_precomputed_rope_artifacts(args.output_artifacts, rope)
    if args.merge_cache_position_inputs:
        text, count = _rewrite_merge_cache_position_inputs(
            text, args.output_artifacts
        )
        rewrites.append(f"merge_cache_position_inputs={count}")
    if args.keep_cache_position_inputs:
        text, count = _remove_input_deallocs_by_role(
            text, args.output_artifacts, {"cache_position"}
        )
        rewrites.append(f"keep_cache_position_inputs={count}")
    if args.keep_static_weight_inputs:
        text, count = _remove_static_weight_input_deallocs(
            text, args.output_artifacts
        )
        rewrites.append(f"keep_static_weight_inputs={count}")
    if args.native_u32_token_io:
        text, count = _rewrite_native_u32_token_io(text, args.output_artifacts)
        rewrites.append(f"native_u32_token_io={count}")

    if args.fuse_create_qkv_heads_decode:
        text, count = _rewrite_create_qkv_heads_decode(text)
        rewrites.append(f"fuse_create_qkv_heads_decode={count}")

    if args.fuse_concat_heads_decode:
        text, count = _rewrite_concat_heads_decode(text)
        rewrites.append(f"fuse_concat_heads_decode={count}")

    if args.fuse_concat_heads_decode_sdpa_output:
        text, count = _rewrite_concat_heads_decode(text, direct_sdpa_output=True)
        rewrites.append(f"fuse_concat_heads_decode_sdpa_output={count}")

    args.output_ttnn_mlir.parent.mkdir(parents=True, exist_ok=True)
    args.output_ttnn_mlir.write_text(text, encoding="utf-8")

    print(
        "decode TTNN postprocess: "
        f"{', '.join(rewrites)}, output={args.output_ttnn_mlir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
