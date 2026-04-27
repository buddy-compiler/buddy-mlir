#!/usr/bin/env python3
# ruff: noqa: E402, F403, E501
# ===- import_model.py - Unified model import (PyTorch → MLIR + weights) --===//
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
# ===----------------------------------------------------------------------===//
#
# Unified replacement for former per-variant scripts, e.g.:
#   - examples/BuddyDeepSeekR1/import-deepseek-r1-w8a16.py
#   - examples/BuddyDeepSeekR1/import-deepseek-r1-w8a32.py
#   - etc.
#
# All variant-specific behaviour is driven by the full config JSON.
#
# Usage:
#   python import_model.py --config deepseek_r1_f32.json --output-dir ./build
#   python import_model.py --config deepseek_r1_w8a16.json --output-dir ./build
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys

# buddy.compiler.* is synced to build/python_packages by target python-package-buddy
# (needs -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON). CMake also sets PYTHONPATH; this
# helps when running import_model.py manually.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_BUDDY_PY_PKG = os.path.join(_REPO_ROOT, "build", "python_packages")
if _BUDDY_PY_PKG not in sys.path:
    sys.path.insert(0, _BUDDY_PY_PKG)

import numpy
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM, StaticCache

try:
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.operation import *
    from buddy.compiler.graph.transform import (
        apply_classic_fusion,
        eliminate_matmul_transpose_reshape,
        eliminate_transpose,
        flash_attention_prefill,
        gqa_attention_fusion,
        simply_fuse,
    )
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.ops import tosa
except ImportError as err:
    print(
        "import_model: cannot import buddy.compiler (needed for DynamoCompiler).\n"
        "  Re-configure with: -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON\n"
        "  Then build sync target: cmake --build <build> --target python-package-buddy\n"
        f"  Expected tree: {_BUDDY_PY_PKG}/buddy/compiler/\n",
        file=sys.stderr,
    )
    raise err


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────


def load_model(config: dict):
    """Load model with appropriate precision."""
    model_path = (
        os.environ.get("DEEPSEEKR1_MODEL_PATH") or config["hf_model_path"]
    )
    activation = config["precision"]["activation_type"]

    if activation == "f16":
        model = AutoModelForCausalLM.from_pretrained(model_path).eval().half()
    elif activation == "bf16":
        model = (
            AutoModelForCausalLM.from_pretrained(model_path).eval().bfloat16()
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32
        ).eval()

    model.config.use_cache = False
    print(
        f"[import] Model loaded: {model_path} ({activation})", file=sys.stderr
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Graph import via DynamoCompiler
# ──────────────────────────────────────────────────────────────────────────────


def compile_graphs(model, config: dict):
    """Run DynamoCompiler to produce prefill and decode graphs."""
    max_token_len = config["shape"]["max_token_len"]

    prefill_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_prefill",
    )
    decode_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_decode",
    )

    with torch.no_grad():
        past_kv_decode = StaticCache(
            config=model.config, max_cache_len=max_token_len
        )

        data_prefill = {
            "input_ids": torch.zeros((1, max_token_len), dtype=torch.int64)
        }
        data_decode = {"input_ids": torch.zeros((1, 1), dtype=torch.int64)}
        cache_position = torch.tensor([200], dtype=torch.int64)

        graphs_prefill = prefill_compiler.importer(
            model,
            input_ids=data_prefill["input_ids"],
            use_cache=True,
            cache_implementation="static",
        )

        model(
            input_ids=data_decode["input_ids"],
            past_key_values=past_kv_decode,
            use_cache=True,
            cache_implementation="static",
        )

        graphs_decode = decode_compiler.importer(
            model,
            input_ids=data_decode["input_ids"],
            use_cache=True,
            cache_position=cache_position,
            past_key_values=past_kv_decode,
            cache_implementation="static",
        )

    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1

    params = prefill_compiler.imported_params[graphs_prefill[0]]
    print(
        f"[import] Graphs imported: {len(params)} parameters", file=sys.stderr
    )
    return graphs_prefill, graphs_decode, params


# ──────────────────────────────────────────────────────────────────────────────
# Graph transforms (shared across all variants)
# ──────────────────────────────────────────────────────────────────────────────


def apply_pre_transforms(graph_prefill, graph_decode):
    """Eliminate transposes and fused matmul reshapes."""
    graph_prefill.perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    graph_decode.perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )


def apply_fusion(graph_prefill, graph_decode):
    """Apply fusion patterns and rename subgraphs."""
    pattern_prefill = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_decode = [simply_fuse, apply_classic_fusion, gqa_attention_fusion]

    graph_prefill.fuse_ops(pattern_prefill)
    graph_decode.fuse_ops(pattern_decode)

    graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
        "subgraph0"
    )
    graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

    graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
        "subgraph0"
    )
    graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU


def create_drivers(graph_prefill, graph_decode):
    """Create GraphDrivers and lower to top-level IR."""
    driver_prefill = GraphDriver(graph_prefill)
    driver_prefill.subgraphs[0].lower_to_top_level_ir()

    driver_decode = GraphDriver(graph_decode)
    driver_decode.subgraphs[0].lower_to_top_level_ir()

    return driver_prefill, driver_decode


# ──────────────────────────────────────────────────────────────────────────────
# Quantization (variant-specific)
# ──────────────────────────────────────────────────────────────────────────────


def apply_quantization(graph_prefill, graph_decode, variant: str):
    """Apply quantization transforms based on the variant name."""
    if variant in ("w8a32", "w8a16"):
        from buddy.compiler.graph.transform.quantization import (
            weight_only_channel_wise,
        )

        weight_only_channel_wise(graph_prefill)
        weight_only_channel_wise(graph_decode)
    elif variant == "w8a8":
        from buddy.compiler.graph.transform.quantization import (
            w8a8_channel_wise,
        )

        w8a8_channel_wise(graph_prefill)
        w8a8_channel_wise(graph_decode)
    elif variant == "w4a16":
        from buddy.compiler.graph.transform.quantization import (
            weight_only_int4_f16_channel_wise,
        )

        weight_only_int4_f16_channel_wise(graph_prefill)
        weight_only_int4_f16_channel_wise(graph_decode)
    else:
        raise ValueError(f"Unknown quantization variant: {variant}")

    print(f"[import] Quantization applied: {variant}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Weight extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_plain_weights(
    params: list, config: dict
) -> dict[str, numpy.ndarray]:
    """Extract weights for non-quantized variants (f32/f16/bf16)."""
    activation = config["precision"]["activation_type"]

    if activation == "bf16":
        all_param = numpy.concatenate(
            [p.detach().float().numpy().reshape([-1]) for p in params]
        )
        all_param_bf16 = numpy.frombuffer(
            all_param.astype(numpy.float32).tobytes(), dtype=numpy.uint16
        )[1::2]
        return {"params": all_param_bf16}
    elif activation == "f16":
        all_param = numpy.concatenate(
            [p.detach().numpy().reshape([-1]) for p in params]
        )
        return {"params": all_param}
    else:
        all_param = numpy.concatenate(
            [p.detach().numpy().reshape([-1]) for p in params]
        )
        return {"params": all_param}


def extract_quantized_weights(
    graph_prefill, original_params: list, config: dict
) -> dict[str, numpy.ndarray]:
    """Extract weights for quantized variants, splitting by dtype bucket."""
    from buddy.compiler.graph.type import TensorDType

    variant = config["variant"]
    activation = config["precision"]["activation_type"]
    num_original = len(original_params)
    # W4A16: int4 range [-8, 7], scales use amax/7; weights packed 2 nibbles/byte (see
    # examples/BuddyDeepSeekR1/import-deepseek-r1-w4a16.py). W8A*: int8 [-128,127].
    is_w4a16 = variant == "w4a16"
    quant_max = 7.0 if is_w4a16 else 127.0

    float_tag = "f16_params" if activation == "f16" else "f32_params"
    int_tag = "i4_params" if variant == "w4a16" else "i8_params"

    float_parts = []
    int_parts = []

    def pack_int4_nibbles(flat: numpy.ndarray) -> numpy.ndarray:
        """Pack int4 values (stored as int8 in [-8,7]) two per byte."""
        assert flat.shape[0] % 2 == 0, f"odd int4 element count {flat.shape[0]}"
        low = flat[0::2].astype(numpy.uint8) & 0x0F
        high = (flat[1::2].astype(numpy.uint8) & 0x0F) << 4
        return (low | high).astype(numpy.int8)

    for i, param_node in enumerate(graph_prefill.params):
        if i < num_original:
            tensor = original_params[i]
            if param_node.tensor_meta.get("dtype") == TensorDType.Int8:
                scaler_name = "scaler_" + param_node.name
                scaler_node = graph_prefill.node_table.get(scaler_name)
                assert scaler_node is not None, (
                    f"Missing scaler for {param_node.name}"
                )

                scaler_shape = list(scaler_node.tensor_meta["shape"])
                quant_axis = next(
                    j for j, s in enumerate(scaler_shape) if s != 1
                )
                reduce_dims = [
                    d for d in range(tensor.dim()) if d != quant_axis
                ]
                if not reduce_dims:
                    reduce_dims = [0]

                amax = tensor.abs().amax(dim=reduce_dims, keepdim=True)
                scale = (amax / quant_max).clamp(min=1e-10)
                if is_w4a16:
                    weight_q = torch.clamp(
                        torch.round(tensor / scale), -8, 7
                    ).to(torch.int8)
                    flat = weight_q.detach().numpy().reshape([-1])
                    int_parts.append(pack_int4_nibbles(flat))
                else:
                    weight_i8 = torch.clamp(
                        torch.round(tensor / scale), -128, 127
                    ).to(torch.int8)
                    int_parts.append(weight_i8.detach().numpy().reshape([-1]))
            else:
                if activation == "f16":
                    float_parts.append(
                        tensor.detach().half().numpy().reshape([-1])
                    )
                else:
                    float_parts.append(tensor.detach().numpy().reshape([-1]))
        else:
            assert param_node.name.startswith("scaler_"), (
                f"Expected scaler param, got {param_node.name}"
            )
            weight_name = param_node.name[len("scaler_") :]
            weight_idx = next(
                j
                for j, p in enumerate(graph_prefill.params[:num_original])
                if p.name == weight_name
            )
            tensor = original_params[weight_idx]
            scaler_shape = list(param_node.tensor_meta["shape"])
            quant_axis = next(j for j, s in enumerate(scaler_shape) if s != 1)
            reduce_dims = [d for d in range(tensor.dim()) if d != quant_axis]
            if not reduce_dims:
                reduce_dims = [0]

            amax = tensor.abs().amax(dim=reduce_dims, keepdim=True)
            scale = (amax / quant_max).clamp(min=1e-10)

            if activation == "f16":
                float_parts.append(scale.half().detach().numpy().reshape([-1]))
            else:
                float_parts.append(scale.detach().numpy().reshape([-1]))

    result = {}
    if float_parts:
        result[float_tag] = numpy.concatenate(float_parts)
    if int_parts:
        result[int_tag] = numpy.concatenate(int_parts)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────


def export_mlir(driver_prefill, driver_decode, config: dict, output_dir: str):
    """Write MLIR files for prefill and decode."""
    variant = config["variant"]
    suffix = f"-{variant}" if variant not in ("f32",) else ""

    files = {
        f"subgraph0_prefill{suffix}.mlir": driver_prefill.subgraphs[
            0
        ]._imported_module,
        f"forward_prefill{suffix}.mlir": driver_prefill.construct_main_graph(
            True
        ),
        f"subgraph0_decode{suffix}.mlir": driver_decode.subgraphs[
            0
        ]._imported_module,
        f"forward_decode{suffix}.mlir": driver_decode.construct_main_graph(
            True
        ),
    }

    for name, content in files.items():
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            print(content, file=f)
        print(f"[import] Written: {name}", file=sys.stderr)


def export_weights(
    weight_buckets: dict[str, numpy.ndarray], config: dict, output_dir: str
):
    """Write weight data files."""
    weights_config = config["weights"]
    tag_to_file = {w["tag"]: w["file"] for w in weights_config}

    actual_sizes = {}
    for tag, data in weight_buckets.items():
        fname = tag_to_file.get(tag, f"arg0-{tag}.data")
        path = os.path.join(output_dir, fname)
        data.tofile(path)
        size_mb = data.nbytes / (1024 * 1024)
        actual_sizes[tag] = len(data)
        print(
            f"[import] Written: {fname} ({size_mb:.1f} MB, {len(data):,} elements)",
            file=sys.stderr,
        )

    return actual_sizes


def update_config(config: dict, actual_sizes: dict, output_dir: str):
    """Update config JSON with actual weight sizes computed during import."""
    changed = False
    for w in config["weights"]:
        if w["tag"] in actual_sizes:
            actual = actual_sizes[w["tag"]]
            if w["num_elements"] != actual:
                print(
                    f"[import] Updated {w['tag']}: {w['num_elements']} → {actual}",
                    file=sys.stderr,
                )
                w["num_elements"] = actual
                changed = True

    if changed:
        config_path = os.path.join(
            output_dir, f"{config['model_id']}_config.json"
        )
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"[import] Updated config: {config_path}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────


def import_model(config: dict, output_dir: str):
    """Full import pipeline: load → compile → transform → export."""
    os.makedirs(output_dir, exist_ok=True)
    variant = config["variant"]
    is_quantized = variant.startswith("w")

    # 1. Load model
    model = load_model(config)

    # 2. Compile graphs
    graphs_prefill, graphs_decode, original_params = compile_graphs(
        model, config
    )

    # 3. Pre-fusion transforms
    apply_pre_transforms(graphs_prefill[0], graphs_decode[0])

    # 4. Quantization (if applicable)
    if is_quantized:
        apply_quantization(graphs_prefill[0], graphs_decode[0], variant)

    # 5. Extract weights (before fusion changes the graph)
    if is_quantized:
        weight_buckets = extract_quantized_weights(
            graphs_prefill[0], original_params, config
        )
    else:
        weight_buckets = extract_plain_weights(original_params, config)

    # 6. Fusion + subgraph rename
    apply_fusion(graphs_prefill[0], graphs_decode[0])

    # 7. Lower to top-level IR
    driver_prefill, driver_decode = create_drivers(
        graphs_prefill[0], graphs_decode[0]
    )

    # 8. Export MLIR + weights
    export_mlir(driver_prefill, driver_decode, config, output_dir)
    actual_sizes = export_weights(weight_buckets, config, output_dir)
    update_config(config, actual_sizes, output_dir)

    print("[import] Import complete.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Unified model import: PyTorch → MLIR + weight data."
    )
    parser.add_argument(
        "--config", required=True, help="Full model config JSON"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    import_model(config, args.output_dir)


if __name__ == "__main__":
    main()
