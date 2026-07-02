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
import importlib
import importlib.util
import json
import os
import sys
import time
import types
from contextlib import contextmanager

# buddy.compiler.* is synced to build/python_packages by target python-package-buddy
# (needs -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON). CMake also sets PYTHONPATH; this
# helps when running import_model.py manually.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_BUDDY_PY_PKG = os.path.join(_REPO_ROOT, "build", "python_packages")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _BUDDY_PY_PKG not in sys.path:
    sys.path.insert(0, _BUDDY_PY_PKG)

if "tomli" not in sys.modules:
    tomli_stub = types.ModuleType("tomli")

    def _tomli_unavailable(*_args, **_kwargs):
        raise RuntimeError("tomli is required only when loading trace configs")

    tomli_stub.load = _tomli_unavailable
    tomli_stub.loads = _tomli_unavailable
    sys.modules["tomli"] = tomli_stub

import numpy
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM, StaticCache

try:
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import (
        GraphDriver,
        PartitionedGraphDriver,
    )
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
# Timing
# ──────────────────────────────────────────────────────────────────────────────


@contextmanager
def timed_import_step(name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        print(f"[import][time] {name}: {elapsed:.2f}s", file=sys.stderr)


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


def is_tiered_kv_cache(config: dict) -> bool:
    return bool(config.get("tiered_kv_cache", {}).get("enabled", False))


def tiered_cache_sizes(config: dict) -> list[int]:
    sizes = config.get("tiered_kv_cache", {}).get("cache_sizes", [])
    return [int(x) for x in sizes]


def compile_and_export_tiered_graphs(
    model,
    config: dict,
    output_dir: str,
    export_layer_partitioned: bool = False,
):
    """Generate one prefill/decode pair for every configured KV cache size."""
    cache_sizes = tiered_cache_sizes(config)
    if not cache_sizes:
        raise ValueError("tiered_kv_cache.enabled requires cache_sizes")

    if config["variant"] != "f32":
        raise ValueError("tiered KV cache import currently supports f32 only")

    pattern_prefill_with_flash = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_prefill_no_flash = [
        simply_fuse,
        apply_classic_fusion,
    ]
    pattern_decode = [simply_fuse, apply_classic_fusion, gqa_attention_fusion]

    params = None
    mlir_output_dir = output_dir
    partition_manifest = {
        "tiered": True,
        "prefill": {},
        "decode": {},
        "decode_partitioned": export_layer_partitioned,
    }
    if export_layer_partitioned:
        mlir_output_dir = os.path.join(output_dir, "layer_partitioned")
        os.makedirs(mlir_output_dir, exist_ok=True)
        for filename in os.listdir(mlir_output_dir):
            if (
                filename.endswith(".mlir")
                or filename == "partition_manifest.json"
            ):
                os.remove(os.path.join(mlir_output_dir, filename))

    for prefill_size in cache_sizes:
        print(
            f"[import] Tiered prefill graph: seq_len={prefill_size}",
            file=sys.stderr,
        )
        torch._dynamo.reset()

        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=inductor_decomp,
            func_name=f"forward_prefill_{prefill_size}",
        )

        with torch.no_grad():
            input_ids = torch.zeros((1, prefill_size), dtype=torch.int64)
            graphs = compiler.importer(
                model,
                input_ids=input_ids,
                use_cache=True,
                cache_implementation="static",
            )
            if params is None:
                params = compiler.imported_params[graphs[0]]

        assert len(graphs) == 1
        graph = graphs[0]
        graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
        if prefill_size >= 64:
            graph.fuse_ops(pattern_prefill_with_flash)
        else:
            graph.fuse_ops(pattern_prefill_no_flash)

        name = (
            f"subgraph0_prefill_{prefill_size}_"
            if export_layer_partitioned
            else f"subgraph0_prefill_{prefill_size}"
        )
        graph.op_groups[name] = graph.op_groups.pop("subgraph0")
        graph.group_map_device[name] = DeviceType.CPU

        files = {}
        if export_layer_partitioned:
            driver = PartitionedGraphDriver(
                graph, layer_split_strategy(config, "prefill")
            )
            for subgraph in driver.subgraphs:
                subgraph.lower_to_top_level_ir()
            prefill_output_count = len(graph.body[-1].args)
            prefill_output_remap = list(range(prefill_output_count))
            if (
                prefill_output_count >= 3
                and (prefill_output_count - 1) % 2 == 0
            ):
                kv_count = prefill_output_count - 1
                prefill_output_remap = [
                    i ^ 1 if i < kv_count else i
                    for i in range(prefill_output_count)
                ]
            for i, subgraph in enumerate(driver.subgraphs):
                files[f"subgraph0_prefill_{prefill_size}_{i}.mlir"] = (
                    subgraph._imported_module
                )
            files[f"forward_prefill_{prefill_size}.mlir"] = (
                driver.construct_combined_main_graph(True, prefill_output_remap)
            )
            partition_manifest["prefill"][str(prefill_size)] = {
                "subgraphs": len(driver.subgraphs),
                "forward": f"forward_prefill_{prefill_size}.mlir",
            }
        else:
            driver = GraphDriver(graph)
            driver.subgraphs[0].lower_to_top_level_ir()
            files = {
                f"subgraph0_prefill_{prefill_size}.mlir": driver.subgraphs[
                    0
                ]._imported_module,
                f"forward_prefill_{prefill_size}.mlir": driver.construct_main_graph(
                    True
                ),
            }
        for filename, content in files.items():
            with open(os.path.join(mlir_output_dir, filename), "w") as f:
                print(content, file=f)
            prefix = "layer_partitioned/" if export_layer_partitioned else ""
            print(f"[import] Written: {prefix}{filename}", file=sys.stderr)

    if params is None:
        raise RuntimeError("tiered prefill import produced no parameters")

    data_decode = {"input_ids": torch.zeros((1, 1), dtype=torch.int64)}
    for cache_size in cache_sizes:
        print(
            f"[import] Tiered decode graph: cache_len={cache_size}",
            file=sys.stderr,
        )
        torch._dynamo.reset()

        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=inductor_decomp,
            func_name=f"forward_decode_{cache_size}",
        )

        with torch.no_grad():
            past_kv = StaticCache(config=model.config, max_cache_len=cache_size)
            cache_position = torch.tensor(
                [min(200 * cache_size // 1024, cache_size - 1)],
                dtype=torch.int64,
            )

            model(
                input_ids=data_decode["input_ids"],
                past_key_values=past_kv,
                use_cache=True,
                cache_implementation="static",
            )

            graphs = compiler.importer(
                model,
                input_ids=data_decode["input_ids"],
                use_cache=True,
                cache_position=cache_position,
                past_key_values=past_kv,
                cache_implementation="static",
            )

        assert len(graphs) == 1
        graph = graphs[0]
        graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
        graph.fuse_ops(pattern_decode)

        name = (
            f"subgraph0_decode_{cache_size}_"
            if export_layer_partitioned
            else f"subgraph0_decode_{cache_size}"
        )
        graph.op_groups[name] = graph.op_groups.pop("subgraph0")
        graph.group_map_device[name] = DeviceType.CPU

        files = {}
        if export_layer_partitioned:
            driver = PartitionedGraphDriver(
                graph, layer_split_strategy(config, "decode")
            )
            for subgraph in driver.subgraphs:
                subgraph.lower_to_top_level_ir()
            for i, subgraph in enumerate(driver.subgraphs):
                files[f"subgraph0_decode_{cache_size}_{i}.mlir"] = (
                    subgraph._imported_module
                )
            files[f"forward_decode_{cache_size}.mlir"] = (
                driver.construct_combined_main_graph(True)
            )
            partition_manifest["decode"][str(cache_size)] = {
                "subgraphs": len(driver.subgraphs),
                "forward": f"forward_decode_{cache_size}.mlir",
            }
        else:
            driver = GraphDriver(graph)
            driver.subgraphs[0].lower_to_top_level_ir()
            files = {
                f"subgraph0_decode_{cache_size}.mlir": driver.subgraphs[
                    0
                ]._imported_module,
                f"forward_decode_{cache_size}.mlir": driver.construct_main_graph(
                    True
                ),
            }
            partition_manifest["decode"][str(cache_size)] = {
                "subgraphs": 1,
                "forward": f"forward_decode_{cache_size}.mlir",
            }
        for filename, content in files.items():
            with open(os.path.join(mlir_output_dir, filename), "w") as f:
                print(content, file=f)
            prefix = "layer_partitioned/" if export_layer_partitioned else ""
            print(f"[import] Written: {prefix}{filename}", file=sys.stderr)

    if export_layer_partitioned:
        with open(
            os.path.join(mlir_output_dir, "partition_manifest.json"), "w"
        ) as f:
            json.dump(partition_manifest, f, indent=2)
            f.write("\n")
        prefill_total = sum(
            item["subgraphs"] for item in partition_manifest["prefill"].values()
        )
        decode_total = sum(
            item["subgraphs"] for item in partition_manifest["decode"].values()
        )
        print(
            "[import] Tiered layer partitioned export complete: "
            f"{prefill_total} prefill + {decode_total} decode subgraphs",
            file=sys.stderr,
        )

    return params


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


def layer_split_strategy(config: dict, kind: str):
    """Load the model-specific layer split strategy."""
    model_family = config.get("model_family")
    if not model_family:
        raise ValueError("layer partitioning requires config.model_family")

    module_name = f"models.{model_family}.codegen.partition_strategy"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name and (
            err.name == module_name or module_name.startswith(err.name + ".")
        ):
            raise ValueError(
                f"layer partitioning is not supported for {model_family}: "
                f"missing {module_name}"
            ) from err
        raise

    try:
        strategy_factory = module.layer_split_strategy
    except AttributeError as err:
        raise ValueError(
            f"{module_name} must define layer_split_strategy(kind)"
        ) from err
    return strategy_factory(kind)


def export_layer_partitioned_mlir(
    graph_prefill,
    graph_decode,
    config: dict,
    output_dir: str,
    export_debug_wrappers: bool = False,
) -> dict[str, int]:
    """Export per-layer subgraph/main MLIR files for compile-time experiments."""
    driver_prefill = PartitionedGraphDriver(
        graph_prefill, layer_split_strategy(config, "prefill")
    )
    for subgraph in driver_prefill.subgraphs:
        subgraph.lower_to_top_level_ir()
    driver_prefill.construct_main_graph(True)

    driver_decode = PartitionedGraphDriver(
        graph_decode, layer_split_strategy(config, "decode")
    )
    for subgraph in driver_decode.subgraphs:
        subgraph.lower_to_top_level_ir()
    driver_decode.construct_main_graph(True)

    partition_dir = os.path.join(output_dir, "layer_partitioned")
    os.makedirs(partition_dir, exist_ok=True)

    for i, module in enumerate(driver_prefill.subgraphs):
        name = f"subgraph0_prefill{i}.mlir"
        with open(os.path.join(partition_dir, name), "w") as f:
            print(module._imported_module, file=f)
        print(f"[import] Written: layer_partitioned/{name}", file=sys.stderr)
    if export_debug_wrappers:
        for i, module in enumerate(driver_prefill.modules):
            name = f"forward_prefill{i}.mlir"
            with open(os.path.join(partition_dir, name), "w") as f:
                print(module, file=f)
            print(
                f"[import] Written: layer_partitioned/{name}", file=sys.stderr
            )
    prefill_output_count = len(graph_prefill.body[-1].args)
    prefill_output_remap = list(range(prefill_output_count))
    if prefill_output_count >= 3 and (prefill_output_count - 1) % 2 == 0:
        kv_count = prefill_output_count - 1
        prefill_output_remap = [
            i ^ 1 if i < kv_count else i for i in range(prefill_output_count)
        ]
    combined_prefill = driver_prefill.construct_combined_main_graph(
        True, prefill_output_remap
    )
    with open(os.path.join(partition_dir, "forward_prefill.mlir"), "w") as f:
        print(combined_prefill, file=f)
    print(
        "[import] Written: layer_partitioned/forward_prefill.mlir",
        file=sys.stderr,
    )

    for i, module in enumerate(driver_decode.subgraphs):
        name = f"subgraph0_decode{i}.mlir"
        with open(os.path.join(partition_dir, name), "w") as f:
            print(module._imported_module, file=f)
        print(f"[import] Written: layer_partitioned/{name}", file=sys.stderr)
    if export_debug_wrappers:
        for i, module in enumerate(driver_decode.modules):
            name = f"forward_decode{i}.mlir"
            with open(os.path.join(partition_dir, name), "w") as f:
                print(module, file=f)
            print(
                f"[import] Written: layer_partitioned/{name}", file=sys.stderr
            )
    combined_decode = driver_decode.construct_combined_main_graph(True)
    with open(os.path.join(partition_dir, "forward_decode.mlir"), "w") as f:
        print(combined_decode, file=f)
    print(
        "[import] Written: layer_partitioned/forward_decode.mlir",
        file=sys.stderr,
    )

    manifest = {
        "prefill_subgraphs": len(driver_prefill.subgraphs),
        "prefill_main_graphs": (
            len(driver_prefill.modules) if export_debug_wrappers else 0
        ),
        "decode_subgraphs": len(driver_decode.subgraphs),
        "decode_main_graphs": (
            len(driver_decode.modules) if export_debug_wrappers else 0
        ),
        "debug_wrappers": export_debug_wrappers,
    }
    with open(os.path.join(partition_dir, "partition_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        "[import] Layer partitioned export complete: "
        f"{manifest['prefill_subgraphs']} prefill + "
        f"{manifest['decode_subgraphs']} decode subgraphs",
        file=sys.stderr,
    )
    return manifest


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


def plain_weight_dtype(config: dict):
    activation = config["precision"]["activation_type"]
    if activation == "bf16":
        return numpy.uint16
    if activation == "f16":
        return numpy.float16
    return numpy.float32


def export_plain_weights_direct(
    params: list, config: dict, output_dir: str
) -> dict:
    """Write non-quantized weights directly without a giant concatenate."""
    if len(config["weights"]) != 1:
        raise ValueError("plain direct weight export expects one weight bucket")

    weight = config["weights"][0]
    if weight["tag"] != "params":
        raise ValueError("plain direct weight export expects the params bucket")

    dtype = plain_weight_dtype(config)
    total_elements = sum(p.numel() for p in params)
    if int(weight["num_elements"]) != total_elements:
        print(
            f"[import] Updated {weight['tag']}: "
            f"{weight['num_elements']} → {total_elements}",
            file=sys.stderr,
        )
        weight["num_elements"] = total_elements

    path = os.path.join(output_dir, weight["file"])
    mm = numpy.memmap(path, dtype=dtype, mode="w+", shape=(total_elements,))
    offset = 0
    activation = config["precision"]["activation_type"]
    for p in params:
        count = p.numel()
        tensor = p.detach()
        if activation == "bf16":
            param_f32 = tensor.float().numpy().reshape([-1])
            part = numpy.frombuffer(
                param_f32.astype(numpy.float32).tobytes(), dtype=numpy.uint16
            )[1::2]
            mm[offset : offset + count] = part
        elif activation == "f16":
            mm[offset : offset + count] = tensor.numpy().reshape([-1])
        else:
            mm[offset : offset + count] = tensor.numpy().reshape([-1])
        offset += count
    mm.flush()
    del mm

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(
        f"[import] Written: {weight['file']} ({size_mb:.1f} MB, "
        f"{total_elements:,} elements)",
        file=sys.stderr,
    )
    return {"params": total_elements}


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
    weight_buckets: dict[str, numpy.ndarray],
    config: dict,
    output_dir: str,
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
            f"[import] Written: {fname} ({size_mb:.1f} MB, "
            f"{actual_sizes[tag]:,} elements)",
            file=sys.stderr,
        )

    return actual_sizes


def weights_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".buddy_weights_manifest.json")


def expected_weight_entries(config: dict, output_dir: str) -> list[dict]:
    entries = []
    for weight in config["weights"]:
        entries.append(
            {
                "tag": weight["tag"],
                "file": weight["file"],
                "bytes_per_element": int(weight["bytes_per_element"]),
                "num_elements": int(weight["num_elements"]),
                "path": os.path.join(output_dir, weight["file"]),
            }
        )
    return entries


def weights_manifest_payload(config: dict, output_dir: str) -> dict:
    model_path = (
        os.environ.get("DEEPSEEKR1_MODEL_PATH") or config["hf_model_path"]
    )
    return {
        "model_path": model_path,
        "model_id": config.get("model_id"),
        "model_family": config.get("model_family"),
        "variant": config.get("variant"),
        "precision": config.get("precision", {}),
        "weights": [
            {
                "tag": entry["tag"],
                "file": entry["file"],
                "bytes_per_element": entry["bytes_per_element"],
                "num_elements": entry["num_elements"],
                "size_bytes": entry["bytes_per_element"]
                * entry["num_elements"],
            }
            for entry in expected_weight_entries(config, output_dir)
        ],
    }


def write_weights_manifest(config: dict, output_dir: str) -> None:
    path = weights_manifest_path(output_dir)
    with open(path, "w") as f:
        json.dump(weights_manifest_payload(config, output_dir), f, indent=2)
        f.write("\n")
    print(f"[import] Written: {os.path.basename(path)}", file=sys.stderr)


def can_reuse_existing_weights(config: dict, output_dir: str) -> bool:
    manifest_file = weights_manifest_path(output_dir)
    if not os.path.exists(manifest_file):
        return False

    try:
        with open(manifest_file) as f:
            existing_manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected_manifest = weights_manifest_payload(config, output_dir)
    if existing_manifest != expected_manifest:
        return False

    for entry in expected_weight_entries(config, output_dir):
        expected_size = entry["bytes_per_element"] * entry["num_elements"]
        try:
            actual_size = os.path.getsize(entry["path"])
        except OSError:
            return False
        if actual_size != expected_size:
            return False

    return True


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


def import_model(
    config: dict,
    output_dir: str,
    export_layer_partitioned: bool = False,
    export_layer_partition_debug_wrappers: bool = False,
    export_full_mlir: bool = True,
    direct_plain_weight_export: bool = True,
    reuse_existing_weights: bool = False,
    skip_weights: bool = False,
):
    """Full import pipeline: load → compile → transform → export."""
    os.makedirs(output_dir, exist_ok=True)
    variant = config["variant"]
    is_quantized = variant.startswith("w")

    # 1. Load model
    with timed_import_step("load_model"):
        model = load_model(config)

    if is_tiered_kv_cache(config):
        if is_quantized:
            raise ValueError(
                "tiered KV cache import does not support quantized variants"
            )
        with timed_import_step("compile_tiered_graphs"):
            original_params = compile_and_export_tiered_graphs(
                model,
                config,
                output_dir,
                export_layer_partitioned=export_layer_partitioned,
            )
        with timed_import_step("export_weights"):
            if reuse_existing_weights and can_reuse_existing_weights(
                config, output_dir
            ):
                print("[import] Reusing existing weight data.", file=sys.stderr)
            elif skip_weights:
                print("[import] Skipped weight export.", file=sys.stderr)
            elif direct_plain_weight_export:
                actual_sizes = export_plain_weights_direct(
                    original_params, config, output_dir
                )
                update_config(config, actual_sizes, output_dir)
                write_weights_manifest(config, output_dir)
            else:
                weight_buckets = extract_plain_weights(original_params, config)
                actual_sizes = export_weights(
                    weight_buckets, config, output_dir
                )
                update_config(config, actual_sizes, output_dir)
                write_weights_manifest(config, output_dir)
        print("[import] Tiered KV cache import complete.", file=sys.stderr)
        return

    # 2. Compile graphs
    with timed_import_step("compile_graphs"):
        graphs_prefill, graphs_decode, original_params = compile_graphs(
            model, config
        )

    # 3. Pre-fusion transforms
    with timed_import_step("pre_transforms"):
        apply_pre_transforms(graphs_prefill[0], graphs_decode[0])

    # 4. Quantization (if applicable)
    if is_quantized:
        with timed_import_step("quantization"):
            apply_quantization(graphs_prefill[0], graphs_decode[0], variant)

    # 5. Extract weights (before fusion changes the graph)
    reused_weights = False
    direct_weight_export = False
    with timed_import_step("extract_weights"):
        if reuse_existing_weights and can_reuse_existing_weights(
            config, output_dir
        ):
            weight_buckets = {}
            reused_weights = True
            print("[import] Reusing existing weight data.", file=sys.stderr)
        elif skip_weights:
            if is_quantized:
                raise ValueError(
                    "--skip-weights is only supported for f32/f16/bf16"
                )
            weight_buckets = {}
        elif is_quantized:
            weight_buckets = extract_quantized_weights(
                graphs_prefill[0], original_params, config
            )
        elif direct_plain_weight_export:
            weight_buckets = {}
            direct_weight_export = True
        else:
            weight_buckets = extract_plain_weights(original_params, config)

    # 6. Fusion + subgraph rename
    with timed_import_step("fusion"):
        apply_fusion(graphs_prefill[0], graphs_decode[0])

    if export_layer_partitioned:
        with timed_import_step("export_layer_partitioned_mlir"):
            export_layer_partitioned_mlir(
                graphs_prefill[0],
                graphs_decode[0],
                config,
                output_dir,
                export_debug_wrappers=export_layer_partition_debug_wrappers,
            )

    # 7. Export whole-graph MLIR when requested. Partitioned runtime builds
    # consume layer_partitioned/forward_*.mlir instead, so this is optional.
    if export_full_mlir:
        with timed_import_step("export_full_mlir"):
            driver_prefill, driver_decode = create_drivers(
                graphs_prefill[0], graphs_decode[0]
            )
            export_mlir(driver_prefill, driver_decode, config, output_dir)
    else:
        print("[import] Skipped whole-graph MLIR export.", file=sys.stderr)

    # 8. Export weights.
    if direct_weight_export:
        with timed_import_step("export_weights_direct"):
            actual_sizes = export_plain_weights_direct(
                original_params, config, output_dir
            )
            update_config(config, actual_sizes, output_dir)
            write_weights_manifest(config, output_dir)
    elif weight_buckets:
        with timed_import_step("export_weights"):
            actual_sizes = export_weights(weight_buckets, config, output_dir)
            update_config(config, actual_sizes, output_dir)
            write_weights_manifest(config, output_dir)
    elif reused_weights:
        print("[import] Reused existing weight data.", file=sys.stderr)
    else:
        print("[import] Skipped weight export.", file=sys.stderr)

    print("[import] Import complete.", file=sys.stderr)


def import_qwen3_vl_model(config: dict, output_dir: str) -> None:
    """Dispatch Qwen3-VL's multimodal importer through the shared entry."""
    model_path = os.environ.get("QWEN3_VL_MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "QWEN3_VL_MODEL_PATH is required for model_family=qwen3_vl"
        )

    os.environ["QWEN3_VL_OUT_DIR"] = os.path.join(output_dir, "artifacts")
    os.environ["QWEN3_VL_PKG"] = output_dir

    script = os.path.join(
        _REPO_ROOT, "models", "qwen3_vl", "codegen", "qwen3_vl_codegen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "buddy_qwen3_vl_codegen", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Qwen3-VL importer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ImportArgs:
        no_import = False
        seq_len = int(config.get("max_seq_len", 160))

    module.cmd_import_vision(ImportArgs())
    module.cmd_import_decoder_rt(ImportArgs())


def main():
    parser = argparse.ArgumentParser(
        description="Unified model import: PyTorch → MLIR + weight data."
    )
    parser.add_argument(
        "--config", required=True, help="Full model config JSON"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--experimental-layer-partitioned",
        action="store_true",
        help="Also export per-layer MLIR under output-dir/layer_partitioned.",
    )
    parser.add_argument(
        "--layer-partition-debug-wrappers",
        action="store_true",
        help=(
            "With --experimental-layer-partitioned, also export per-partition "
            "forward_* debug wrapper MLIR files."
        ),
    )
    parser.add_argument(
        "--skip-full-mlir",
        action="store_true",
        help=(
            "Skip standard whole-graph forward/subgraph MLIR export. This is "
            "intended for partitioned runtime builds that only need "
            "layer_partitioned/*.mlir plus weights."
        ),
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight data export for compile-time MLIR experiments.",
    )
    parser.add_argument(
        "--reuse-existing-weights",
        action="store_true",
        help=(
            "If output-dir already contains matching weight files and a "
            "matching .buddy_weights_manifest.json, skip weight extraction and "
            "export."
        ),
    )
    parser.add_argument(
        "--no-direct-plain-weight-export",
        action="store_true",
        help=(
            "For non-quantized weights, use the legacy concatenate-then-write "
            "path instead of direct memmap export."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    try:
        if config.get("model_family") == "qwen3_vl":
            import_qwen3_vl_model(config, args.output_dir)
        else:
            import_model(
                config,
                args.output_dir,
                export_layer_partitioned=args.experimental_layer_partitioned,
                export_layer_partition_debug_wrappers=(
                    args.layer_partition_debug_wrappers
                ),
                export_full_mlir=not args.skip_full_mlir,
                direct_plain_weight_export=not args.no_direct_plain_weight_export,
                reuse_existing_weights=args.reuse_existing_weights,
                skip_weights=args.skip_weights,
            )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
