#!/usr/bin/env python3
"""Import Qwen3-0.6B as reusable layer-wise Buddy FPGA graphs.

The generated ABI mirrors the compact bare-metal Qwen3 runner: embedding and
final-head graphs execute once, while one prefill/decode decoder-layer graph is
called 28 times with different parameter views.  Persistent hidden/KV buffers
are owned by the caller, so temporary allocations can be discarded after each
layer instead of accumulating across a monolithic 28-layer graph.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from buddy.compiler.graph.type import TensorDType
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

from import_qwen3_model import (
    DEFAULT_MODEL_DIR,
    Qwen3PrefillGraph,
    apply_static_smoothquant,
    decoder_layer,
    emit_graph,
    import_graph,
    packed_param_sizes,
    rotary_embedding,
    transform_graph,
    write_w8a8_params,
)
from generate_layerwise_bindings import generate_bindings
from qwen3_prompt import (
    DEFAULT_TOKENIZER_DIR,
    left_pad_token_ids,
    parse_token_ids,
    tokenize_chat_prompt,
)


HERE = Path(__file__).resolve().parent
TRACE_COMPONENTS = (
    "embedding_prefill",
    "embedding_decode",
    "decoder_layer_prefill",
    "decoder_layer_decode",
    "final_head",
)


class EmbeddingGraph(nn.Module):
    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class DecoderLayerPrefillGraph(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        return decoder_layer(self.layer, hidden, cos, sin, attention_mask)


class DecoderLayerDecodeGraph(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_write_mask: torch.Tensor,
        old_key: torch.Tensor,
        old_value: torch.Tensor,
    ):
        return decoder_layer(
            self.layer,
            hidden,
            cos,
            sin,
            attention_mask,
            old_key,
            old_value,
            cache_write_mask,
        )


class FinalHeadGraph(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-wise Buddy FPGA importer for Qwen3-0.6B W8A8"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "build-model-w8a8-layerwise"
    )
    parser.add_argument(
        "--prefill-len", type=int, default=128,
        help="Static padded prefill capacity shared by every prompt",
    )
    parser.add_argument(
        "--max-cache-len", type=int, default=256,
        help="KV-cache length including padded prefill and decode positions",
    )
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prefill-token-ids", type=str, default=None)
    prompt_source.add_argument(
        "--prompt", type=str, default=None,
        help="User text encoded with the local Qwen3 chat template",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--system-prompt", type=str, default="Qwen3")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--reuse-params-from", type=Path, default=None,
        help="Reuse an existing layer-wise params pack instead of repacking weights",
    )
    parser.add_argument("--skip-params", action="store_true")
    parser.add_argument(
        "--w8a8-scale-method", choices=("max", "mse"), default="mse"
    )
    parser.add_argument("--w8a8-mse-grid", type=int, default=16)
    parser.add_argument("--w8a8-smooth-alpha", type=float, default=0.5)
    parser.add_argument("--w8a8-smooth-max-scale", type=float, default=8.0)
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument(
        "--trace-component",
        choices=TRACE_COMPONENTS,
        default=None,
        help="Component whose Buddy graph nodes are instrumented",
    )
    parser.add_argument(
        "--trace-config",
        type=Path,
        default=None,
        help="Trace TOML containing node names for --trace-component",
    )
    parser.add_argument(
        "--trace-spec",
        action="append",
        default=[],
        metavar="COMPONENT=CONFIG.toml",
        help=(
            "Instrument another component; may be repeated. This allows one "
            "image to profile decoder internals and the LM head together"
        ),
    )
    parser.add_argument(
        "--buddy-graph-dir",
        type=Path,
        default=None,
        help="Write verbose Buddy graphs here for choosing trace node names",
    )
    return parser.parse_args()


def resolve_prompt(
    args: argparse.Namespace,
) -> tuple[list[int] | None, list[int], dict]:
    if args.prompt is not None:
        values = tokenize_chat_prompt(
            args.prompt,
            args.tokenizer_dir,
            args.system_prompt,
            args.enable_thinking,
        )
        source = {
            "kind": "qwen3_chat_template",
            "text": args.prompt,
            "system_prompt": args.system_prompt,
            "enable_thinking": args.enable_thinking,
            "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        }
    elif args.prefill_token_ids is not None:
        values = parse_token_ids(args.prefill_token_ids)
        source = {"kind": "token_ids"}
    else:
        values = None
        source = {"kind": "zero_placeholder"}

    actual_tokens = len(values) if values is not None else args.prefill_len
    if actual_tokens > args.prefill_len:
        raise ValueError(
            f"prompt contains {actual_tokens} tokens but the padded prefill "
            f"capacity is {args.prefill_len}"
        )
    padding_tokens = args.prefill_len - actual_tokens
    padded_values = (
        left_pad_token_ids(values, args.prefill_len)
        if values is not None else [0] * args.prefill_len
    )
    source.update({
        "token_count": actual_tokens,
        "padded_length": args.prefill_len,
        "padding_side": "left",
        "padding_tokens": padding_tokens,
        "pad_token_id": 151643,
    })
    return values, padded_values, source


def reuse_parameter_pack(source: Path, output: Path) -> tuple[dict, dict]:
    source = source.resolve()
    manifest_path = source / "params_manifest.json"
    config_path = source / "model_compile_config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(f"invalid --reuse-params-from directory: {source}")
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    for name in ("params_manifest.json", "params_f32.data", "params_i8.data"):
        source_file = source / name
        destination = output / name
        if not source_file.is_file():
            raise ValueError(f"missing reusable parameter file: {source_file}")
        if destination.exists():
            if destination.stat().st_size != source_file.stat().st_size:
                raise ValueError(f"existing parameter file has wrong size: {destination}")
            continue
        try:
            os.link(source_file, destination)
        except OSError:
            shutil.copy2(source_file, destination)
    return manifest, config


def tensor_name_index(model: nn.Module) -> dict[int, list[str]]:
    """Index model tensors by storage address, including rotary buffers."""
    result: dict[int, list[str]] = defaultdict(list)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        result[int(tensor.data_ptr())].append(name)
    return result


def resolve_source_names(
    params: list[torch.Tensor], names_by_pointer: dict[int, list[str]]
) -> list[str]:
    names = []
    for index, tensor in enumerate(params):
        candidates = names_by_pointer.get(int(tensor.data_ptr()), [])
        # Qwen3 exposes original_inv_freq as an alias of inv_freq.  The graph
        # consumes inv_freq, so keep the stable runtime-facing spelling.
        candidates = [
            name for name in candidates if not name.endswith("original_inv_freq")
        ] or candidates
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot resolve checkpoint tensor {index} at "
                f"0x{int(tensor.data_ptr()):x}: candidates={candidates}"
            )
        names.append(candidates[0])
    return names


def compiler(
    name: str,
    trace: TraceConfig | None = None,
    verbose_path: Path | None = None,
) -> DynamoCompiler:
    return DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=f"forward_{name}",
        verbose=verbose_path is not None,
        verbose_path=verbose_path,
        trace=trace,
    )


def import_component(
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    patterns: list,
    output_dir: Path,
    names_by_pointer: dict[int, list[str]],
    trace: TraceConfig | None = None,
    verbose_path: Path | None = None,
):
    graph_compiler = compiler(name, trace, verbose_path)
    graph = import_graph(graph_compiler, module, *inputs)
    params = graph_compiler.imported_params[graph]
    source_names = resolve_source_names(params, names_by_pointer)
    transform_graph(graph, f"subgraph0_{name}", patterns, "w8a8")
    boundary = emit_graph(graph, output_dir, name)
    return graph, params, source_names, boundary


def parameter_records(graph, params, source_names) -> list[dict]:
    """Describe graph parameter order independently of transient arg names."""
    original_nodes = graph.params[: len(params)]
    scaler_nodes = graph.params[len(params) :]
    quantized = [
        (index, node)
        for index, node in enumerate(original_nodes)
        if node.tensor_meta["dtype"] == TensorDType.Int8
    ]
    if [node.name for node in scaler_nodes] != [
        "scaler_" + node.name for _, node in quantized
    ]:
        raise RuntimeError("component W8A8 scaler order is invalid")

    records = []
    for index, node in enumerate(original_nodes):
        kind = (
            "quantized_weight"
            if node.tensor_meta["dtype"] == TensorDType.Int8
            else "checkpoint"
        )
        records.append({
            "graph_parameter_index": index,
            "node": node.name,
            "model_parameter": source_names[index],
            "kind": kind,
            "shape": [int(value) for value in node.tensor_meta["shape"]],
            "dtype": (
                "int8" if node.tensor_meta["dtype"] == TensorDType.Int8
                else "float32"
            ),
        })
    for scaler_offset, ((source_index, _), scaler) in enumerate(
        zip(quantized, scaler_nodes)
    ):
        records.append({
            "graph_parameter_index": len(original_nodes) + scaler_offset,
            "node": scaler.name,
            "model_parameter": source_names[source_index],
            "kind": "quantization_scale",
            "shape": [int(value) for value in scaler.tensor_meta["shape"]],
            "dtype": "float32",
        })
    return records


def pack_lookup(manifest: dict) -> dict[tuple[str, str], int]:
    lookup = {}
    for index, entry in enumerate(manifest["parameters"]):
        key = (entry.get("model_parameter"), entry["kind"])
        if key in lookup:
            raise RuntimeError(f"duplicate packed parameter key {key}")
        lookup[key] = index
    return lookup


def bind_records(records: list[dict], lookup: dict) -> list[dict]:
    result = []
    for record in records:
        bound = dict(record)
        key = (record["model_parameter"], record["kind"])
        if lookup:
            if key not in lookup:
                raise RuntimeError(f"parameter is absent from full pack: {key}")
            bound["pack_manifest_index"] = lookup[key]
        result.append(bound)
    return result


def layer_records(records: list[dict], layer: int, lookup: dict) -> list[dict]:
    prefix = "model.layers.0."
    result = []
    for record in records:
        source = record["model_parameter"]
        if not source.startswith(prefix):
            raise RuntimeError(f"layer template parameter has unexpected name: {source}")
        bound = dict(record)
        bound["model_parameter"] = f"model.layers.{layer}." + source[len(prefix):]
        key = (bound["model_parameter"], bound["kind"])
        if lookup:
            if key not in lookup:
                raise RuntimeError(f"layer {layer} parameter is absent from pack: {key}")
            bound["pack_manifest_index"] = lookup[key]
        result.append(bound)
    return result


def dtype_name(dtype: TensorDType) -> str:
    names = {
        TensorDType.Int8: "int8",
        TensorDType.Int64: "int64",
        TensorDType.Float32: "float32",
    }
    if dtype not in names:
        raise RuntimeError(f"unsupported layer-wise ABI dtype: {dtype}")
    return names[dtype]


def component_call_abi(
    name: str,
    graph,
    boundary: dict[str, list[str]],
    parameter_bindings: list[dict],
    runtime_labels: list[str],
    output_labels: list[str],
) -> dict:
    """Map the exact subgraph C-wrapper positions to semantic operands."""
    if len(graph.inputs) != len(runtime_labels):
        raise RuntimeError(
            f"{name}: graph has {len(graph.inputs)} runtime inputs, "
            f"but {len(runtime_labels)} ABI labels were provided"
        )
    parameter_by_node = {record["node"]: record for record in parameter_bindings}
    runtime_by_node = {
        node.name: (label, node)
        for node, label in zip(graph.inputs, runtime_labels)
    }
    arguments = []
    for position, node_name in enumerate(boundary["arguments"]):
        node = graph.node_table[node_name]
        shape = [int(value) for value in node.tensor_meta["shape"]]
        dtype = dtype_name(node.tensor_meta["dtype"])
        if node_name in parameter_by_node:
            record = parameter_by_node[node_name]
            argument = {
                "position": position,
                "c_wrapper_pointer_index": position + 1,
                "node": node_name,
                "source": "parameter",
                "graph_parameter_index": record["graph_parameter_index"],
                "model_parameter": record["model_parameter"],
                "kind": record["kind"],
                "shape": shape,
                "dtype": dtype,
            }
            if "pack_manifest_index" in record:
                argument["pack_manifest_index"] = record["pack_manifest_index"]
        elif node_name in runtime_by_node:
            label, _ = runtime_by_node[node_name]
            argument = {
                "position": position,
                "c_wrapper_pointer_index": position + 1,
                "node": node_name,
                "source": "runtime",
                "runtime_tensor": label,
                "shape": shape,
                "dtype": dtype,
            }
        else:
            raise RuntimeError(f"{name}: unclassified boundary operand {node_name}")
        arguments.append(argument)

    if len(boundary["outputs"]) != len(output_labels):
        raise RuntimeError(
            f"{name}: graph has {len(boundary['outputs'])} outputs, "
            f"but {len(output_labels)} ABI labels were provided"
        )
    outputs = []
    for position, (node_name, label) in enumerate(
        zip(boundary["outputs"], output_labels)
    ):
        node = graph.node_table[node_name]
        outputs.append({
            "position": position,
            "runtime_tensor": label,
            "shape": [int(value) for value in node.tensor_meta["shape"]],
            "dtype": dtype_name(node.tensor_meta["dtype"]),
        })
    return {
        "c_wrapper": f"_mlir_ciface_subgraph0_{name}",
        "result_pointer_index": 0,
        "arguments": arguments,
        "outputs": outputs,
    }


def main() -> int:
    args = parse_args()
    if (args.trace_component is None) != (args.trace_config is None):
        raise ValueError(
            "--trace-component and --trace-config must be provided together"
        )
    selected_traces: dict[str, tuple[TraceConfig, Path]] = {}
    if args.trace_config is not None:
        selected_traces[args.trace_component] = (
            TraceConfig(load_trace_config(args.trace_config)),
            args.trace_config.resolve(),
        )
    for spec in args.trace_spec:
        if "=" not in spec:
            raise ValueError(
                f"invalid --trace-spec {spec!r}; expected COMPONENT=CONFIG.toml"
            )
        component, config_text = spec.split("=", 1)
        if component not in TRACE_COMPONENTS:
            raise ValueError(
                f"invalid trace component {component!r}; "
                f"choose one of {', '.join(TRACE_COMPONENTS)}"
            )
        if component in selected_traces:
            raise ValueError(f"duplicate trace configuration for {component}")
        config_path = Path(config_text).resolve()
        selected_traces[component] = (
            TraceConfig(load_trace_config(config_path)), config_path
        )
    if args.buddy_graph_dir is not None:
        args.buddy_graph_dir.mkdir(parents=True, exist_ok=True)

    def component_trace(name: str) -> TraceConfig | None:
        selected = selected_traces.get(name)
        return selected[0] if selected is not None else None

    def component_verbose_path(name: str) -> Path | None:
        if args.buddy_graph_dir is None:
            return None
        return args.buddy_graph_dir / f"{name}.txt"

    if args.skip_params and args.reuse_params_from is not None:
        raise ValueError("--skip-params and --reuse-params-from are mutually exclusive")
    fixed_ids, padded_ids, prompt_metadata = resolve_prompt(args)
    if args.prefill_len <= 0:
        raise ValueError("--prefill-len must be positive")
    if args.max_cache_len <= args.prefill_len:
        raise ValueError(
            "--max-cache-len must be greater than --prefill-len to leave "
            "one decode position"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if fixed_ids is not None:
        print(
            f"prepared {len(fixed_ids)} prompt tokens from "
            f"{prompt_metadata['kind']} with "
            f"{prompt_metadata['padding_tokens']} left-padding tokens",
            flush=True,
        )

    print(f"loading Qwen3 checkpoint from {args.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    model.config.use_cache = False
    if model.config.tie_word_embeddings:
        model.lm_head.weight = nn.Parameter(
            model.lm_head.weight.detach().clone(), requires_grad=False
        )
    apply_static_smoothquant(
        model, args.w8a8_smooth_alpha, args.w8a8_smooth_max_scale
    )
    names_by_pointer = tensor_name_index(model)

    if any(token < 0 or token >= model.config.vocab_size for token in padded_ids):
        raise ValueError("prefill token ID is outside the model vocabulary")
    prefill_ids = torch.tensor([padded_ids], dtype=torch.int64)
    positions = torch.arange(args.prefill_len, dtype=torch.int64).unsqueeze(0)
    prefill_mask = torch.full(
        (1, 1, args.prefill_len, args.prefill_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    ).triu(diagonal=1)
    padding_tokens = int(prompt_metadata["padding_tokens"])
    if padding_tokens:
        prefill_mask[..., :padding_tokens] = torch.finfo(torch.float32).min
    hidden_prefill = torch.zeros(
        (1, args.prefill_len, model.config.hidden_size), dtype=torch.float32
    )
    with torch.no_grad():
        cos_prefill, sin_prefill = rotary_embedding(
            model.model.rotary_emb, hidden_prefill, positions
        )

    decode_position = args.prefill_len
    decode_positions = torch.arange(
        args.prefill_len, args.max_cache_len, dtype=torch.int64
    ).unsqueeze(0)
    decode_hidden = torch.zeros(
        (1, 1, model.config.hidden_size), dtype=torch.float32
    )
    decode_table_hidden = torch.zeros(
        (1, decode_positions.shape[1], model.config.hidden_size),
        dtype=torch.float32,
    )
    with torch.no_grad():
        decode_cos_table, decode_sin_table = rotary_embedding(
            model.model.rotary_emb,
            decode_table_hidden,
            decode_positions,
        )
    # The reusable graph still has the single-token [1,1,head_dim] ABI.  The
    # runtime advances this pointer by head_dim for each decode position.
    decode_cos = decode_cos_table[:, :1, :]
    decode_sin = decode_sin_table[:, :1, :]
    decode_mask = torch.full(
        (1, 1, 1, args.max_cache_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    decode_mask[..., padding_tokens : decode_position + 1] = 0.0
    write_mask = torch.zeros(
        (1, 1, args.max_cache_len, 1), dtype=torch.float32
    )
    write_mask[..., decode_position, :] = 1.0
    layer_cache = torch.zeros(
        (
            1,
            model.config.num_key_value_heads,
            args.max_cache_len,
            model.config.head_dim,
        ),
        dtype=torch.float32,
    )

    simple_patterns = [simply_fuse] if args.no_fusion else [
        simply_fuse, apply_classic_fusion
    ]
    prefill_patterns = [simply_fuse] if args.no_fusion else [
        simply_fuse, apply_classic_fusion, flash_attention_prefill
    ]
    decode_patterns = [simply_fuse] if args.no_fusion else [
        simply_fuse, apply_classic_fusion, gqa_attention_fusion
    ]

    components = {}
    print("importing layer-wise embedding graphs", flush=True)
    components["embedding_prefill"] = import_component(
        "embedding_prefill",
        EmbeddingGraph(model.model.embed_tokens).eval(),
        (prefill_ids,),
        simple_patterns,
        args.output_dir,
        names_by_pointer,
        component_trace("embedding_prefill"),
        component_verbose_path("embedding_prefill"),
    )
    components["embedding_decode"] = import_component(
        "embedding_decode",
        EmbeddingGraph(model.model.embed_tokens).eval(),
        (torch.zeros((1, 1), dtype=torch.int64),),
        simple_patterns,
        args.output_dir,
        names_by_pointer,
        component_trace("embedding_decode"),
        component_verbose_path("embedding_decode"),
    )

    layer0 = model.model.layers[0]
    print("importing reusable decoder-layer graphs", flush=True)
    components["decoder_layer_prefill"] = import_component(
        "decoder_layer_prefill",
        DecoderLayerPrefillGraph(layer0).eval(),
        (hidden_prefill, cos_prefill, sin_prefill, prefill_mask),
        prefill_patterns,
        args.output_dir,
        names_by_pointer,
        component_trace("decoder_layer_prefill"),
        component_verbose_path("decoder_layer_prefill"),
    )
    components["decoder_layer_decode"] = import_component(
        "decoder_layer_decode",
        DecoderLayerDecodeGraph(layer0).eval(),
        (
            decode_hidden,
            decode_cos,
            decode_sin,
            decode_mask,
            write_mask,
            layer_cache,
            layer_cache.clone(),
        ),
        decode_patterns,
        args.output_dir,
        names_by_pointer,
        component_trace("decoder_layer_decode"),
        component_verbose_path("decoder_layer_decode"),
    )
    components["final_head"] = import_component(
        "final_head",
        FinalHeadGraph(model.model.norm, model.lm_head).eval(),
        (decode_hidden,),
        simple_patterns,
        args.output_dir,
        names_by_pointer,
        component_trace("final_head"),
        component_verbose_path("final_head"),
    )

    prefill_layer_names = components["decoder_layer_prefill"][2]
    decode_layer_names = components["decoder_layer_decode"][2]
    if prefill_layer_names != decode_layer_names:
        raise RuntimeError("prefill/decode layer parameter orders differ")

    manifest = None
    reused_config = None
    if args.reuse_params_from is not None:
        print(f"reusing W8A8 packs from {args.reuse_params_from}", flush=True)
        manifest, reused_config = reuse_parameter_pack(
            args.reuse_params_from, args.output_dir
        )
        packed_sizes = {
            "float32": int(reused_config["packed_f32_elements"]),
            "int8": int(reused_config["packed_i8_elements"]),
        }
        expected_bytes = packed_sizes["int8"] + packed_sizes["float32"] * 4
        actual_bytes = (
            (args.output_dir / "params_i8.data").stat().st_size
            + (args.output_dir / "params_f32.data").stat().st_size
        )
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"reused parameter size mismatch: {actual_bytes} != {expected_bytes}"
            )
    else:
        # Import a one-token complete graph only to establish the global
        # parameter order and serialize all 28 layers. It is never linked.
        print("building global W8A8 parameter pack order", flush=True)
        pack_compiler = compiler("parameter_pack")
        pack_graph = import_graph(
            pack_compiler,
            Qwen3PrefillGraph(model).eval(),
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        )
        pack_params = pack_compiler.imported_params[pack_graph]
        pack_source_names = resolve_source_names(pack_params, names_by_pointer)
        # A single coarse group is sufficient here; the graph is used only to
        # establish transformed parameter metadata and is never emitted.
        transform_graph(
            pack_graph, "subgraph0_parameter_pack", [simply_fuse], "w8a8"
        )
        packed_sizes = packed_param_sizes(pack_graph)
        if not args.skip_params:
            print(
                "writing layer-wise global W8A8 packs: "
                f"{packed_sizes['int8'] + packed_sizes['float32'] * 4} bytes",
                flush=True,
            )
            manifest = write_w8a8_params(
                pack_params,
                pack_graph,
                args.output_dir,
                model.config.vocab_size,
                model.config.hidden_size,
                args.w8a8_scale_method,
                args.w8a8_mse_grid,
                source_names=pack_source_names,
            )
    lookup = pack_lookup(manifest) if manifest is not None else {}

    component_records = {
        name: parameter_records(graph, params, source_names)
        for name, (graph, params, source_names, _) in components.items()
    }
    bound_component_records = {
        "embedding_prefill": bind_records(
            component_records["embedding_prefill"], lookup
        ),
        "embedding_decode": bind_records(
            component_records["embedding_decode"], lookup
        ),
        "decoder_layer_prefill": layer_records(
            component_records["decoder_layer_prefill"], 0, lookup
        ),
        "decoder_layer_decode": layer_records(
            component_records["decoder_layer_decode"], 0, lookup
        ),
        "final_head": bind_records(component_records["final_head"], lookup),
    }
    runtime_labels = {
        "embedding_prefill": ["input_ids"],
        "embedding_decode": ["input_ids"],
        "decoder_layer_prefill": ["hidden", "rope_cos", "rope_sin", "attention_mask"],
        "decoder_layer_decode": [
            "hidden",
            "rope_cos",
            "rope_sin",
            "old_key",
            "old_value",
            "cache_write_mask",
            "attention_mask",
        ],
        "final_head": ["hidden"],
    }
    output_labels = {
        "embedding_prefill": ["hidden"],
        "embedding_decode": ["hidden"],
        # Qwen3DecoderLayer returns (value_states, key_states, hidden_states).
        # Keep these labels in the exact generated MLIR result order because
        # the C ABI returns the aggregate by value without reordering fields.
        "decoder_layer_prefill": ["new_value", "new_key", "hidden"],
        "decoder_layer_decode": ["new_value", "new_key", "hidden"],
        "final_head": ["logits"],
    }
    call_abi = {
        name: component_call_abi(
            name,
            graph,
            boundary,
            bound_component_records[name],
            runtime_labels[name],
            output_labels[name],
        )
        for name, (graph, _, _, boundary) in components.items()
    }
    abi = {
        "format": "buddy-qwen3-layerwise-abi-v1",
        "execution": {
            "prefill": [
                "embedding_prefill",
                "decoder_layer_prefill x 28",
                "final_head(last_token)",
            ],
            "decode": [
                "embedding_decode",
                "decoder_layer_decode x 28",
                "commit one [KV heads,1,head dim] cache slice per layer",
                "final_head",
            ],
            "temporary_lifetime": "reset after every decoder layer",
            "persistent": ["hidden_ping", "hidden_pong", "key_cache", "value_cache"],
        },
        "prefill_len": args.prefill_len,
        "max_cache_len": args.max_cache_len,
        "prefill_padding": {
            "side": "left",
            "pad_token_id": 151643,
            "actual_length_file": "prompt_length_i32.data",
        },
        "decode_rope_positions": [args.prefill_len, args.max_cache_len],
        "num_layers": model.config.num_hidden_layers,
        "cache_shape": [
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            args.max_cache_len,
            model.config.head_dim,
        ],
        "decode_cache_update": {
            "mode": "single_position",
            "output_shape": [
                1,
                model.config.num_key_value_heads,
                1,
                model.config.head_dim,
            ],
            "position_source": "cache_write_mask",
        },
        "call_abi": call_abi,
        "graphs": {
            "embedding_prefill": bound_component_records["embedding_prefill"],
            "embedding_decode": bound_component_records["embedding_decode"],
            "decoder_layer_prefill_template": component_records[
                "decoder_layer_prefill"
            ],
            "decoder_layer_decode_template": component_records[
                "decoder_layer_decode"
            ],
            "final_head": bound_component_records["final_head"],
        },
        "layers": [
            {
                "layer": layer,
                "prefill_parameter_bindings": layer_records(
                    component_records["decoder_layer_prefill"], layer, lookup
                ),
                "decode_parameter_bindings": layer_records(
                    component_records["decoder_layer_decode"], layer, lookup
                ),
            }
            for layer in range(model.config.num_hidden_layers)
        ],
    }
    (args.output_dir / "layerwise_abi.json").write_text(
        json.dumps(abi, indent=2) + "\n"
    )
    if manifest is not None:
        generate_bindings(
            args.output_dir / "layerwise_abi.json",
            args.output_dir / "params_manifest.json",
            args.output_dir / "layerwise_bindings.c",
        )

    prefill_ids.numpy().tofile(args.output_dir / "prefill_input_ids_i64.data")
    torch.tensor(
        [prompt_metadata["token_count"]], dtype=torch.int32
    ).numpy().tofile(args.output_dir / "prompt_length_i32.data")
    cos_prefill.numpy().tofile(args.output_dir / "prefill_rope_cos_f32.data")
    sin_prefill.numpy().tofile(args.output_dir / "prefill_rope_sin_f32.data")
    prefill_mask.numpy().tofile(args.output_dir / "prefill_mask_f32.data")
    decode_cos_table.numpy().tofile(args.output_dir / "decode_rope_cos_f32.data")
    decode_sin_table.numpy().tofile(args.output_dir / "decode_rope_sin_f32.data")
    decode_mask.numpy().tofile(args.output_dir / "decode_mask_f32.data")
    write_mask.numpy().tofile(args.output_dir / "cache_write_mask_f32.data")

    metadata = {
        "model": "Qwen3-0.6B",
        "execution_mode": "layerwise",
        "prefill_len": args.prefill_len,
        "prefill_token_ids": fixed_ids,
        "padded_prefill_token_ids": padded_ids,
        "actual_prompt_len": prompt_metadata["token_count"],
        "prompt": prompt_metadata,
        "max_cache_len": args.max_cache_len,
        "decode_rope_positions": [args.prefill_len, args.max_cache_len],
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "q_heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "vocab_size": model.config.vocab_size,
        "packed_f32_elements": packed_sizes["float32"],
        "packed_i8_elements": packed_sizes["int8"],
        "packed_parameter_bytes": packed_sizes["int8"] + packed_sizes["float32"] * 4,
        "params_written": not args.skip_params,
        "params_reused_from": (
            str(args.reuse_params_from.resolve())
            if args.reuse_params_from is not None else None
        ),
        "component_graphs": list(components),
        "cycle_trace": [
            {"component": component, "config": str(config_path)}
            for component, (_, config_path) in selected_traces.items()
        ],
    }
    (args.output_dir / "model_compile_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
