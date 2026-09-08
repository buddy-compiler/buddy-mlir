#!/usr/bin/env python3
"""Import the complete Qwen3-0.6B prefill/decode graphs with Buddy Frontend.

This follows Buddy's upstream Qwen3 importer, but is fixed to the local model,
uses the FPGA cache length by default, streams the large parameter blob, and
records enough metadata to build a bare-metal runner later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoConfig, AutoModelForCausalLM

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_matmul_transpose_reshape,
    eliminate_transpose,
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from buddy.compiler.graph.transform.quantization import (
    embedding_w8_per_group,
    w8a8_per_group,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.ops import tosa


HERE = Path(__file__).resolve().parent


def _qwen3_0_6b_dir() -> Path:
    """Locate the external qwen3-0.6b checkout (model + platform + tokenizer)."""
    env = os.environ.get("QWEN3_0_6B_DIR")
    if env:
        return Path(env)
    # Fall back to a sibling of the buddy-mlir checkout (kaixinyuan-style
    # layout: <repo>/thirdparty/{buddy-mlir,qwen3-0.6b}).
    return HERE.parents[1].parent / "qwen3-0.6b"


QWEN3_0_6B_DIR = _qwen3_0_6b_dir()
DEFAULT_MODEL_DIR = (
    Path(os.environ["QWEN3_0_6B_MODEL_PATH"])
    if "QWEN3_0_6B_MODEL_PATH" in os.environ
    else QWEN3_0_6B_DIR / "models" / "qwen3" / "model_data"
)
QWEN3_W8A8_GROUP_SIZE_BY_K = {
    1024: 1024,  # QKV, gate/up, LM head
    2048: 512,   # attention output projection
    3072: 512,   # FFN down projection
}
QWEN3_W8A8_EMBED_GROUP_SIZE = 256


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def repeat_kv(value: torch.Tensor, repetitions: int) -> torch.Tensor:
    batch, kv_heads, seq_len, head_dim = value.shape
    if repetitions == 1:
        return value
    value = value[:, :, None, :, :].expand(
        batch, kv_heads, repetitions, seq_len, head_dim
    )
    return value.reshape(batch, kv_heads * repetitions, seq_len, head_dim)


def rotary_embedding(rotary, hidden: torch.Tensor, position_ids: torch.Tensor):
    inv_freq = rotary.inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    )
    frequencies = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(hidden.dtype), embedding.sin().to(hidden.dtype)


def attention(
    layer,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    old_key: torch.Tensor | None = None,
    old_value: torch.Tensor | None = None,
    cache_write_mask: torch.Tensor | None = None,
):
    attn = layer.self_attn
    batch, seq_len, _ = hidden.shape
    head_dim = attn.head_dim
    query = attn.q_norm(attn.q_proj(hidden).view(
        batch, seq_len, -1, head_dim
    )).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden).view(
        batch, seq_len, -1, head_dim
    )).transpose(1, 2)
    value = attn.v_proj(hidden).view(
        batch, seq_len, -1, head_dim
    ).transpose(1, 2)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    key_update = key
    value_update = value

    if old_key is None:
        expanded_key = repeat_kv(key_update, attn.num_key_value_groups)
        expanded_value = repeat_kv(value_update, attn.num_key_value_groups)
        scores = torch.matmul(query, expanded_key.transpose(2, 3))
        probabilities = torch.softmax(
            scores * attn.scaling + attention_mask,
            dim=-1,
            dtype=torch.float32,
        )
        output = torch.matmul(probabilities, expanded_value)
    else:
        assert old_value is not None and cache_write_mask is not None

        # Decode changes exactly one cache position.  Materializing
        # old_cache * (1 - mask) + update * mask copies the complete K/V cache
        # twice per layer.  Apply the same change as a rank-one correction to
        # the attention score/output instead.  K/V temporaries remain one
        # position wide; only the much smaller score tensor spans the cache.
        write_positions = cache_write_mask.transpose(2, 3)
        expanded_old_key = repeat_kv(old_key, attn.num_key_value_groups)
        expanded_key_update = repeat_kv(
            key_update, attn.num_key_value_groups
        )
        old_scores = torch.matmul(query, expanded_old_key.transpose(2, 3))
        new_score = torch.matmul(
            query, expanded_key_update.transpose(2, 3)
        )
        old_score_at_write_position = (
            old_scores * write_positions
        ).sum(dim=-1, keepdim=True)
        scores = old_scores + (
            new_score - old_score_at_write_position
        ) * write_positions
        probabilities = torch.softmax(
            scores * attn.scaling + attention_mask,
            dim=-1,
            dtype=torch.float32,
        )

        expanded_old_value = repeat_kv(old_value, attn.num_key_value_groups)
        old_output = torch.matmul(probabilities, expanded_old_value)
        old_value_at_write_position = torch.matmul(
            write_positions, old_value
        )
        expanded_old_value_at_write_position = repeat_kv(
            old_value_at_write_position, attn.num_key_value_groups
        )
        expanded_value_update = repeat_kv(
            value_update, attn.num_key_value_groups
        )
        probability_at_write_position = (
            probabilities * write_positions
        ).sum(dim=-1, keepdim=True)
        output = old_output + probability_at_write_position * (
            expanded_value_update - expanded_old_value_at_write_position
        )
    output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
    # Decode commits only the newly computed [B,KV,1,H] slice.  The persistent
    # cache is updated in place by the bare-metal scheduler at the op boundary.
    return attn.o_proj(output), key_update, value_update


def decoder_layer(
    layer,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    old_key: torch.Tensor | None = None,
    old_value: torch.Tensor | None = None,
    cache_write_mask: torch.Tensor | None = None,
):
    residual = hidden
    normalized = layer.input_layernorm(hidden)
    attn_output, key, value = attention(
        layer, normalized, cos, sin, attention_mask,
        old_key, old_value, cache_write_mask
    )
    hidden = residual + attn_output
    residual = hidden
    hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, key, value


class Qwen3PrefillGraph(nn.Module):
    """Pure tensor prefill graph with explicit positions and causal mask."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask):
        core = self.model.model
        hidden = core.embed_tokens(input_ids)
        cos, sin = rotary_embedding(core.rotary_emb, hidden, position_ids)
        cache_outputs = []
        for layer in core.layers:
            hidden, key, value = decoder_layer(
                layer, hidden, cos, sin, attention_mask
            )
            cache_outputs.extend((key, value))
        hidden = core.norm(hidden)
        logits = self.model.lm_head(hidden[:, -1:, :])
        return (*cache_outputs, logits)


class Qwen3DecodeGraph(nn.Module):
    """Single-token decode graph with functional fixed-size KV-cache writes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        cache_write_mask,
        *past_key_values,
    ):
        core = self.model.model
        hidden = core.embed_tokens(input_ids)
        cos, sin = rotary_embedding(core.rotary_emb, hidden, position_ids)
        cache_outputs = []
        for index, layer in enumerate(core.layers):
            hidden, key, value = decoder_layer(
                layer,
                hidden,
                cos,
                sin,
                attention_mask,
                past_key_values[index * 2],
                past_key_values[index * 2 + 1],
                cache_write_mask,
            )
            cache_outputs.extend((key, value))
        hidden = core.norm(hidden)
        logits = self.model.lm_head(hidden)
        return (*cache_outputs, logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Qwen3-0.6B Buddy Frontend importer"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "build-model")
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument(
        "--prefill-token-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated fixed token IDs written to the prefill input; "
            "the number of IDs must equal --prefill-len"
        ),
    )
    parser.add_argument(
        "--weights", choices=("checkpoint", "random"), default="checkpoint"
    )
    parser.add_argument(
        "--quantization", choices=("fp32", "w8a8"), default="fp32",
        help="Use the Qwen3 mixed per-group W8A8 Buddy graph rewrite",
    )
    parser.add_argument(
        "--skip-params", action="store_true",
        help="Emit graphs and metadata without writing parameter data files",
    )
    parser.add_argument(
        "--w8a8-scale-method", choices=("max", "mse"), default="mse",
        help="Checkpoint weight scale search used by W8A8 packing",
    )
    parser.add_argument(
        "--w8a8-mse-grid", type=int, default=16,
        help="Number of clipping candidates for W8A8 MSE scales",
    )
    parser.add_argument(
        "--w8a8-smooth-alpha", type=float, default=0.5,
        help="Static SmoothQuant alpha; zero disables smoothing",
    )
    parser.add_argument(
        "--w8a8-smooth-max-scale", type=float, default=8.0,
        help="Maximum SmoothQuant channel scaling factor",
    )
    parser.add_argument("--no-fusion", action="store_true")
    return parser.parse_args()


def import_graph(compiler: DynamoCompiler, model, *args, **kwargs):
    with torch.no_grad():
        graphs = compiler.importer(model, *args, **kwargs)
    if len(graphs) != 1:
        raise RuntimeError(f"expected one graph, got {len(graphs)}")
    return graphs[0]


def transform_graph(
    graph, name: str, patterns: list, quantization: str = "fp32"
) -> None:
    graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
    if quantization == "w8a8":
        w8a8_per_group(graph, QWEN3_W8A8_GROUP_SIZE_BY_K)
        embedding_w8_per_group(graph, QWEN3_W8A8_EMBED_GROUP_SIZE)
    if patterns:
        graph.fuse_ops(patterns)
    if "subgraph0" not in graph.op_groups:
        raise RuntimeError("Buddy graph has no subgraph0 operation group")
    graph.op_groups[name] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[name] = DeviceType.CPU
    graph.group_map_device.pop("subgraph0", None)


def emit_graph(graph, output_dir: Path, stem: str) -> dict[str, list[str]]:
    driver = GraphDriver(graph)
    if len(driver.subgraphs) != 1:
        raise RuntimeError(
            f"expected one {stem} subgraph, got {len(driver.subgraphs)}"
        )
    subgraph_name = f"subgraph0_{stem}"
    driver.subgraphs[0].lower_to_top_level_ir()
    (output_dir / f"subgraph0_{stem}.mlir").write_text(
        str(driver.subgraphs[0]._imported_module) + "\n"
    )
    (output_dir / f"forward_{stem}.mlir").write_text(
        str(driver.construct_main_graph(True)) + "\n"
    )
    # GraphDriver deliberately preserves repeated boundary uses.  Expose the
    # exact order so callers of the generated C wrapper can bind every memref
    # position correctly instead of assuming params + runtime inputs.
    return {
        "arguments": list(driver._subgraphs_inputs[subgraph_name]),
        "outputs": list(driver._subgraphs_outputs[subgraph_name]),
    }


def param_signature(graph) -> list[tuple[int, ...]]:
    return [
        tuple(int(value) for value in node.tensor_meta["shape"])
        for node in graph.params
    ]


def _smooth_linear_inputs(
    norm_weight: torch.Tensor,
    weights: list[torch.Tensor],
    alpha: float,
    max_scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Apply the static SmoothQuant reparameterization used by Qwen3 W8A8."""
    if alpha <= 0.0:
        return norm_weight, weights
    weight_abs = torch.stack([
        weight.detach().float().abs().amax(dim=0) for weight in weights
    ]).amax(dim=0).clamp_min(1e-6)
    activation_abs = norm_weight.detach().float().abs().clamp_min(1e-6)
    scale = activation_abs.pow(alpha) / weight_abs.pow(1.0 - alpha)
    scale = scale / torch.exp(torch.log(scale.clamp_min(1e-6)).mean())
    scale = scale.clamp(1.0 / max_scale, max_scale)
    return (
        norm_weight.detach().float() / scale,
        [weight.detach().float() * scale.unsqueeze(0) for weight in weights],
    )


def apply_static_smoothquant(model, alpha: float, max_scale: float) -> None:
    """Move per-input-channel ranges between RMSNorm and linear weights."""
    if alpha <= 0.0:
        return
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--w8a8-smooth-alpha must be in [0, 1]")
    if max_scale < 1.0:
        raise ValueError("--w8a8-smooth-max-scale must be >= 1")

    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            norm, weights = _smooth_linear_inputs(
                layer.input_layernorm.weight,
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                alpha,
                max_scale,
            )
            layer.input_layernorm.weight.copy_(norm)
            for destination, value in zip(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                weights,
            ):
                destination.copy_(value)

            mlp = layer.mlp
            norm, weights = _smooth_linear_inputs(
                layer.post_attention_layernorm.weight,
                [mlp.gate_proj.weight, mlp.up_proj.weight],
                alpha,
                max_scale,
            )
            layer.post_attention_layernorm.weight.copy_(norm)
            mlp.gate_proj.weight.copy_(weights[0])
            mlp.up_proj.weight.copy_(weights[1])

        norm, weights = _smooth_linear_inputs(
            model.model.norm.weight,
            [model.lm_head.weight],
            alpha,
            max_scale,
        )
        model.model.norm.weight.copy_(norm)
        model.lm_head.weight.copy_(weights[0])


def write_params(params, graph, output_dir: Path) -> tuple[int, int]:
    blob_path = output_dir / "params_f32.data"
    manifest = []
    byte_offset = 0
    elements = 0
    with blob_path.open("wb") as stream:
        for index, (node, tensor) in enumerate(zip(graph.params, params)):
            array = tensor.detach().to(torch.float32).contiguous().numpy()
            array.tofile(stream)
            count = int(array.size)
            manifest.append({
                "index": index,
                "name": node.name,
                "shape": list(array.shape),
                "dtype": "float32",
                "element_offset": elements,
                "byte_offset": byte_offset,
                "num_elements": count,
            })
            elements += count
            byte_offset += int(array.nbytes)
    (output_dir / "params_manifest.json").write_text(
        json.dumps({
            "file": blob_path.name,
            "total_elements": elements,
            "total_bytes": byte_offset,
            "parameters": manifest,
        }, indent=2) + "\n"
    )
    return elements, byte_offset


def _product(shape) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def packed_param_sizes(graph) -> dict[str, int]:
    sizes = {"float32": 0, "int8": 0}
    for node in graph.params:
        count = _product(node.tensor_meta["shape"])
        dtype = node.tensor_meta["dtype"]
        if dtype == TensorDType.Float32:
            sizes["float32"] += count
        elif dtype == TensorDType.Int8:
            sizes["int8"] += count
        else:
            raise ValueError(f"unsupported packed parameter dtype {dtype}")
    return sizes


def _quantize_columns(
    block: torch.Tensor, method: str, mse_grid: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [group_size, output_chunk] block along group_size."""
    absolute_max = block.abs().amax(dim=0).clamp_min(1e-10)
    if method == "max":
        scale = absolute_max / 127.0
    else:
        best_error = torch.full_like(absolute_max, float("inf"))
        best_clip = absolute_max.clone()
        for shrink in torch.linspace(1.0, 0.5, mse_grid):
            clip = (absolute_max * shrink).clamp_min(1e-10)
            candidate_scale = clip / 127.0
            quantized = torch.round(block / candidate_scale).clamp(-127, 127)
            error = ((quantized * candidate_scale - block) ** 2).mean(dim=0)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_clip = torch.where(better, clip, best_clip)
        scale = best_clip / 127.0
    quantized = torch.round(block / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _write_linear_q8(
    weight: torch.Tensor,
    q_stream,
    scale_stream,
    group_size: int,
    method: str,
    mse_grid: int,
    column_chunk: int = 4096,
) -> None:
    """Write Qwen3 AME [D/64,K/group,64,group] qweight and [G,D] scales."""
    k_size, n_size = (int(value) for value in weight.shape)
    if k_size % group_size:
        raise ValueError(
            f"linear K={k_size} must be divisible by group size {group_size}"
        )
    if k_size % 64:
        raise ValueError(f"linear K={k_size} must be divisible by AME K64")
    if n_size % 64:
        raise ValueError(f"linear D={n_size} must be divisible by OUTBLK64")
    if group_size > 1024:
        raise ValueError("linear group size must not exceed 1024")

    groups = k_size // group_size
    output_blocks = n_size // 64
    weight_base = q_stream.tell()
    for group_start in range(0, k_size, group_size):
        group_index = group_start // group_size
        quantized_group = torch.empty(
            (group_size, n_size), dtype=torch.int8
        )
        group_scales = torch.empty((n_size,), dtype=torch.float32)
        for column_start in range(0, n_size, column_chunk):
            column_end = min(column_start + column_chunk, n_size)
            block = weight[
                group_start : group_start + group_size,
                column_start:column_end,
            ].detach().float().contiguous()
            quantized, scale = _quantize_columns(block, method, mse_grid)
            quantized_group[:, column_start:column_end] = quantized
            group_scales[column_start:column_end] = scale
        # The AME blob is output-block major.  Each [64,group_size] tile is
        # N-major so mlbe8 can load it directly without a runtime transpose.
        for output_block in range(output_blocks):
            tile = quantized_group[
                :, output_block * 64 : (output_block + 1) * 64
            ].T.contiguous()
            tile_index = output_block * groups + group_index
            q_stream.seek(weight_base + tile_index * 64 * group_size)
            tile.numpy().tofile(q_stream)
        group_scales.numpy().tofile(scale_stream)
    q_stream.seek(weight_base + k_size * n_size)


def _unpack_linear_q8(
    packed: torch.Tensor, input_dim: int, output_dim: int, group_size: int
) -> torch.Tensor:
    """Unpack OUTBLK64 test data back to logical Buddy [K,D] order."""
    groups = input_dim // group_size
    expected = (output_dim // 64, groups, 64, group_size)
    if tuple(packed.shape) != expected:
        raise ValueError(f"packed shape {tuple(packed.shape)}, expected {expected}")
    logical = torch.empty((input_dim, output_dim), dtype=packed.dtype)
    for output_block in range(output_dim // 64):
        for group in range(groups):
            logical[
                group * group_size : (group + 1) * group_size,
                output_block * 64 : (output_block + 1) * 64,
            ] = packed[output_block, group].T
    return logical


def _write_embedding_q8(
    weight: torch.Tensor,
    q_stream,
    scale_stream,
    group_size: int,
    method: str,
    mse_grid: int,
    row_chunk: int = 256,
) -> None:
    """Write row-major [vocab,hidden] qweight and [vocab,groups] scales."""
    vocab_size, hidden_size = (int(value) for value in weight.shape)
    groups = hidden_size // group_size
    for row_start in range(0, vocab_size, row_chunk):
        row_end = min(row_start + row_chunk, vocab_size)
        block = weight[row_start:row_end].detach().float().contiguous()
        grouped = block.reshape(row_end - row_start, groups, group_size)
        # Reuse the column quantizer by mapping every (row,group) to a column.
        columns = grouped.permute(2, 0, 1).reshape(group_size, -1)
        quantized, scales = _quantize_columns(columns, method, mse_grid)
        quantized = quantized.reshape(
            group_size, row_end - row_start, groups
        ).permute(1, 2, 0).reshape(row_end - row_start, hidden_size)
        scales = scales.reshape(row_end - row_start, groups)
        quantized.contiguous().numpy().tofile(q_stream)
        scales.contiguous().numpy().tofile(scale_stream)


def write_w8a8_params(
    params,
    graph,
    output_dir: Path,
    vocab_size: int,
    hidden_size: int,
    scale_method: str,
    mse_grid: int,
    source_names: list[str] | None = None,
) -> dict:
    """Serialize the dtype-separated packs consumed by Buddy's main graph."""
    if scale_method == "mse" and mse_grid < 2:
        raise ValueError("--w8a8-mse-grid must be >= 2")
    original_nodes = graph.params[: len(params)]
    scaler_nodes = graph.params[len(params) :]
    if source_names is not None and len(source_names) != len(params):
        raise ValueError(
            f"source_names has {len(source_names)} entries for {len(params)} "
            "checkpoint tensors"
        )
    if any(node.name.startswith("scaler_") for node in original_nodes):
        raise RuntimeError("W8A8 original/scaler parameter ordering is invalid")

    f32_path = output_dir / "params_f32.data"
    i8_path = output_dir / "params_i8.data"
    manifest_entries = []
    f32_elements = 0
    i8_elements = 0

    with f32_path.open("wb") as f32_stream, i8_path.open("wb") as i8_stream:
        # Buddy packs by dtype.  All untouched f32 checkpoint parameters come
        # first; dynamically-created weight scalers follow in weight order.
        for index, (node, tensor) in enumerate(zip(original_nodes, params)):
            if node.tensor_meta["dtype"] != TensorDType.Float32:
                continue
            array = tensor.detach().float().contiguous().numpy()
            if list(array.shape) != list(node.tensor_meta["shape"]):
                raise RuntimeError(
                    f"parameter {node.name} data shape {list(array.shape)} "
                    f"does not match Buddy shape {node.tensor_meta['shape']}"
                )
            array.tofile(f32_stream)
            count = int(array.size)
            manifest_entries.append({
                "index": index,
                "name": node.name,
                "model_parameter": (
                    source_names[index] if source_names is not None else None
                ),
                "shape": list(array.shape),
                "dtype": "float32",
                "file": f32_path.name,
                "element_offset": f32_elements,
                "num_elements": count,
                "kind": "checkpoint",
            })
            f32_elements += count

        quantized_nodes = [
            (index, node, params[index])
            for index, node in enumerate(original_nodes)
            if node.tensor_meta["dtype"] == TensorDType.Int8
        ]
        expected_scalers = ["scaler_" + node.name for _, node, _ in quantized_nodes]
        if [node.name for node in scaler_nodes] != expected_scalers:
            raise RuntimeError("W8A8 scaler order does not match weight order")

        for (index, node, tensor), scaler_node in zip(
            quantized_nodes, scaler_nodes
        ):
            physical_shape = tuple(
                int(value) for value in node.tensor_meta["shape"]
            )
            logical_shape = tuple(
                int(value)
                for value in node.tensor_meta.get(
                    "w8a8_logical_shape", tuple(tensor.shape)
                )
            )
            if tuple(tensor.shape) != logical_shape:
                raise RuntimeError(
                    f"quantized parameter {node.name} data shape "
                    f"{tuple(tensor.shape)} does not match logical shape "
                    f"{logical_shape}"
                )
            scaler_shape = tuple(
                int(value) for value in scaler_node.tensor_meta["shape"]
            )
            is_embedding = logical_shape == (vocab_size, hidden_size)
            if is_embedding:
                group_size = QWEN3_W8A8_EMBED_GROUP_SIZE
                expected_scale_shape = (vocab_size, hidden_size // group_size)
                layout = "vocab_hidden_row_major"
                expected_physical_shape = logical_shape
                _write_embedding_q8(
                    tensor, i8_stream, f32_stream, group_size,
                    scale_method, mse_grid,
                )
            else:
                try:
                    group_size = QWEN3_W8A8_GROUP_SIZE_BY_K[logical_shape[0]]
                except KeyError as exc:
                    raise RuntimeError(
                        f"cannot select W8A8 group for {node.name} shape "
                        f"{logical_shape}"
                    ) from exc
                input_dim, output_dim = logical_shape
                expected_scale_shape = (input_dim // group_size, output_dim)
                expected_physical_shape = (
                    output_dim // 64,
                    input_dim // group_size,
                    64,
                    group_size,
                )
                layout = "ame_outblk64"
                _write_linear_q8(
                    tensor, i8_stream, f32_stream, group_size,
                    scale_method, mse_grid,
                )
            if physical_shape != expected_physical_shape:
                raise RuntimeError(
                    f"parameter {node.name} physical shape {physical_shape}, "
                    f"expected {expected_physical_shape}"
                )
            if scaler_shape != expected_scale_shape:
                raise RuntimeError(
                    f"scaler {scaler_node.name} shape {scaler_shape}, expected "
                    f"{expected_scale_shape}"
                )

            weight_count = _product(physical_shape)
            scale_count = _product(scaler_shape)
            manifest_entries.append({
                "index": index,
                "name": node.name,
                "model_parameter": (
                    source_names[index] if source_names is not None else None
                ),
                "shape": list(physical_shape),
                "dtype": "int8",
                "file": i8_path.name,
                "element_offset": i8_elements,
                "num_elements": weight_count,
                "kind": "quantized_weight",
                "group_size": group_size,
                "layout": layout,
                "input_dim": (
                    hidden_size if is_embedding else logical_shape[0]
                ),
                "output_dim": (
                    vocab_size if is_embedding else logical_shape[1]
                ),
            })
            manifest_entries.append({
                "name": scaler_node.name,
                "source_index": index,
                "model_parameter": (
                    source_names[index] if source_names is not None else None
                ),
                "shape": list(scaler_shape),
                "dtype": "float32",
                "file": f32_path.name,
                "element_offset": f32_elements,
                "num_elements": scale_count,
                "kind": "quantization_scale",
                "group_size": group_size,
            })
            i8_elements += weight_count
            f32_elements += scale_count

    manifest = {
        "format": "buddy-dtype-parameter-packs-v1",
        "scale_method": scale_method,
        "mse_grid": mse_grid if scale_method == "mse" else None,
        "files": {
            "float32": {
                "name": f32_path.name,
                "elements": f32_elements,
                "bytes": f32_elements * 4,
            },
            "int8": {
                "name": i8_path.name,
                "elements": i8_elements,
                "bytes": i8_elements,
            },
        },
        "parameters": manifest_entries,
    }
    (output_dir / "params_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    args = parse_args()
    if args.prefill_len <= 0:
        raise ValueError("--prefill-len must be positive")
    if args.max_cache_len < args.prefill_len:
        raise ValueError("--max-cache-len must be >= --prefill-len")
    fixed_prefill_ids = None
    if args.prefill_token_ids is not None:
        try:
            fixed_prefill_ids = [
                int(value.strip())
                for value in args.prefill_token_ids.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--prefill-token-ids must be a comma-separated integer list"
            ) from exc
        if len(fixed_prefill_ids) != args.prefill_len:
            raise ValueError(
                f"--prefill-token-ids contains {len(fixed_prefill_ids)} IDs, "
                f"but --prefill-len is {args.prefill_len}"
            )
    if args.w8a8_mse_grid < 2 and args.w8a8_scale_method == "mse":
        raise ValueError("--w8a8-mse-grid must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Qwen3 config from {args.model_dir}", flush=True)
    if args.weights == "checkpoint":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.float32, attn_implementation="eager"
        ).eval()
    else:
        config = AutoConfig.from_pretrained(args.model_dir)
        config._attn_implementation = "eager"
        torch.manual_seed(20260820)
        model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    model.config.use_cache = False
    if args.quantization == "w8a8" and model.config.tie_word_embeddings:
        # Embedding uses group_size=256 while the tied LM head uses 1024.
        # Give both graph parameters independent storage/layout so the Buddy
        # quantization passes can represent the checkpoint's mixed groups.
        model.lm_head.weight = nn.Parameter(
            model.lm_head.weight.detach().clone(), requires_grad=False
        )
    if args.quantization == "w8a8":
        print(
            "applying static SmoothQuant "
            f"alpha={args.w8a8_smooth_alpha} "
            f"max_scale={args.w8a8_smooth_max_scale}",
            flush=True,
        )
        apply_static_smoothquant(
            model,
            args.w8a8_smooth_alpha,
            args.w8a8_smooth_max_scale,
        )
    prefill_module = Qwen3PrefillGraph(model).eval()
    decode_module = Qwen3DecodeGraph(model).eval()

    prefill_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_prefill",
        verbose=False,
    )
    decode_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_decode",
        verbose=False,
    )

    if fixed_prefill_ids is None:
        prefill_ids = torch.zeros((1, args.prefill_len), dtype=torch.int64)
    else:
        if any(
            token < 0 or token >= model.config.vocab_size
            for token in fixed_prefill_ids
        ):
            raise ValueError(
                "--prefill-token-ids contains an ID outside the model vocabulary"
            )
        prefill_ids = torch.tensor([fixed_prefill_ids], dtype=torch.int64)
    prefill_positions = torch.arange(
        args.prefill_len, dtype=torch.int64
    ).unsqueeze(0)
    prefill_mask = torch.full(
        (1, 1, args.prefill_len, args.prefill_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    ).triu(diagonal=1)
    decode_ids = torch.zeros((1, 1), dtype=torch.int64)
    decode_position_value = min(args.prefill_len, args.max_cache_len - 1)
    decode_positions = torch.tensor(
        [[decode_position_value]], dtype=torch.int64
    )
    decode_mask = torch.full(
        (1, 1, 1, args.max_cache_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    decode_mask[..., : decode_position_value + 1] = 0.0
    cache_write_mask = torch.zeros(
        (1, 1, args.max_cache_len, 1), dtype=torch.float32
    )
    cache_write_mask[..., decode_position_value, :] = 1.0
    cache_shape = (
        1,
        model.config.num_key_value_heads,
        args.max_cache_len,
        model.config.head_dim,
    )
    decode_caches = tuple(
        torch.zeros(cache_shape, dtype=torch.float32)
        for _ in range(model.config.num_hidden_layers * 2)
    )

    print("importing prefill graph", flush=True)
    prefill_graph = import_graph(
        prefill_compiler,
        prefill_module,
        prefill_ids,
        prefill_positions,
        prefill_mask,
    )

    print("importing functional decode graph", flush=True)
    decode_graph = import_graph(
        decode_compiler,
        decode_module,
        decode_ids,
        decode_positions,
        decode_mask,
        cache_write_mask,
        *decode_caches,
    )

    prefill_params = prefill_compiler.imported_params[prefill_graph]
    decode_params = decode_compiler.imported_params[decode_graph]
    prefill_signature = param_signature(prefill_graph)
    decode_signature = param_signature(decode_graph)
    if prefill_signature != decode_signature:
        raise RuntimeError("prefill/decode parameter layouts differ")
    if any(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(prefill_params, decode_params)
    ):
        raise RuntimeError("prefill/decode parameter order differs")

    prefill_patterns = [simply_fuse] if args.no_fusion else [
        simply_fuse, apply_classic_fusion, flash_attention_prefill
    ]
    decode_patterns = [simply_fuse] if args.no_fusion else [
        simply_fuse, apply_classic_fusion, gqa_attention_fusion
    ]
    transform_graph(
        prefill_graph, "subgraph0_prefill", prefill_patterns,
        args.quantization
    )
    transform_graph(
        decode_graph, "subgraph0_decode", decode_patterns,
        args.quantization
    )

    transformed_prefill_signature = [
        (
            tuple(int(value) for value in node.tensor_meta["shape"]),
            node.tensor_meta["dtype"],
        )
        for node in prefill_graph.params
    ]
    transformed_decode_signature = [
        (
            tuple(int(value) for value in node.tensor_meta["shape"]),
            node.tensor_meta["dtype"],
        )
        for node in decode_graph.params
    ]
    if transformed_prefill_signature != transformed_decode_signature:
        raise RuntimeError("transformed prefill/decode parameter layouts differ")
    if any(
        tuple(tensor.shape)
        != tuple(
            prefill_graph.params[index].tensor_meta.get(
                "w8a8_logical_shape",
                prefill_graph.params[index].tensor_meta["shape"],
            )
        )
        for index, tensor in enumerate(prefill_params)
    ):
        raise RuntimeError("transformed checkpoint data does not match Buddy layout")

    print("emitting Buddy MLIR", flush=True)
    emit_graph(prefill_graph, args.output_dir, "prefill")
    emit_graph(decode_graph, args.output_dir, "decode")
    prefill_ids.numpy().tofile(args.output_dir / "prefill_input_ids_i64.data")
    prefill_positions.numpy().tofile(
        args.output_dir / "prefill_position_ids_i64.data"
    )
    prefill_mask.numpy().tofile(args.output_dir / "prefill_mask_f32.data")
    decode_ids.numpy().tofile(args.output_dir / "decode_input_ids_i64.data")
    decode_positions.numpy().tofile(
        args.output_dir / "decode_position_ids_i64.data"
    )
    decode_mask.numpy().tofile(args.output_dir / "decode_mask_f32.data")
    cache_write_mask.numpy().tofile(
        args.output_dir / "cache_write_mask_f32.data"
    )

    raw_param_elements = sum(int(t.numel()) for t in prefill_params)
    packed_sizes = packed_param_sizes(prefill_graph)
    packed_param_bytes = packed_sizes["float32"] * 4 + packed_sizes["int8"]
    param_manifest = None
    if not args.skip_params:
        if args.quantization == "fp32":
            print(
                f"writing {packed_param_bytes} bytes of f32 parameters",
                flush=True,
            )
            write_params(prefill_params, prefill_graph, args.output_dir)
        else:
            print(
                f"writing {packed_param_bytes} bytes of W8A8 parameters",
                flush=True,
            )
            param_manifest = write_w8a8_params(
                prefill_params,
                prefill_graph,
                args.output_dir,
                model.config.vocab_size,
                model.config.hidden_size,
                args.w8a8_scale_method,
                args.w8a8_mse_grid,
            )

    metadata = {
        "model": "Qwen3-0.6B",
        "frontend": "Buddy DynamoCompiler",
        "prefill_len": args.prefill_len,
        "prefill_token_ids": fixed_prefill_ids,
        "decode_len": 1,
        "max_cache_len": args.max_cache_len,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "q_heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "vocab_size": model.config.vocab_size,
        "weights": args.weights,
        "quantization": args.quantization,
        "w8a8_group_sizes_by_k": (
            QWEN3_W8A8_GROUP_SIZE_BY_K
            if args.quantization == "w8a8"
            else None
        ),
        "w8a8_embedding_group_size": (
            QWEN3_W8A8_EMBED_GROUP_SIZE
            if args.quantization == "w8a8"
            else None
        ),
        "w8a8_scale_method": (
            args.w8a8_scale_method if args.quantization == "w8a8" else None
        ),
        "w8a8_mse_grid": (
            args.w8a8_mse_grid
            if args.quantization == "w8a8" and args.w8a8_scale_method == "mse"
            else None
        ),
        "w8a8_smooth_alpha": (
            args.w8a8_smooth_alpha if args.quantization == "w8a8" else None
        ),
        "w8a8_smooth_max_scale": (
            args.w8a8_smooth_max_scale
            if args.quantization == "w8a8"
            else None
        ),
        "parameter_tensors": len(prefill_params),
        "packed_parameter_tensors": len(prefill_graph.params),
        "raw_parameter_elements": raw_param_elements,
        "packed_f32_elements": packed_sizes["float32"],
        "packed_i8_elements": packed_sizes["int8"],
        "packed_parameter_bytes": packed_param_bytes,
        "params_written": not args.skip_params,
        "parameter_pack_format": (
            param_manifest["format"]
            if param_manifest is not None
            else (
                "buddy-dtype-parameter-packs-v1"
                if args.quantization == "w8a8"
                else "f32-flat-v1"
            )
        ),
    }
    (args.output_dir / "model_compile_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
