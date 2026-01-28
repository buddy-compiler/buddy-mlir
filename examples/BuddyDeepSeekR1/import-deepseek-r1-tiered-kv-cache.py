#!/usr/bin/env python3
# ===- import-deepseek-r1-tiered-kv-cache.py -------------------------------
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
# This is the tiered KV cache version of DeepSeekR1 model importer (F32 only).
# It generates multiple decode subgraphs with different KV cache sizes
# (32, 64, 128, 256, 512, 1024) to enable dynamic cache size selection at runtime.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.operation import *

# Add argument parser
parser = argparse.ArgumentParser(
    description="DeepSeekR1 Model AOT Importer (Tiered KV Cache, F32)"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--max-cache-len",
    type=int,
    default=1024,
    help="Maximum cache length for prefill stage.",
)
parser.add_argument(
    "--cache-sizes",
    type=str,
    default="32,64,128,256,512,1024",
    help="Comma-separated list of cache sizes for decode subgraphs.",
)
args = parser.parse_args()

# Parse cache sizes
cache_sizes = [int(x) for x in args.cache_sizes.split(",")]
max_cache_len = args.max_cache_len

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading model from: {model_path}")
print(f"Cache sizes for decode: {cache_sizes}")
print(f"Max cache length for prefill: {max_cache_len}")

# Initialize the model (F32 only)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torchscript=True
).eval()
model.config.use_cache = False

# Initialize prefill compiler
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

# Generate prefill graph (only once, with max cache length)
print(f"\n=== Generating prefill subgraph (cache_len={max_cache_len}) ===")
with torch.no_grad():
    past_key_values_prefill = StaticCache(
        config=model.config, max_cache_len=max_cache_len
    )

    data_prefill = {
        "input_ids": torch.zeros((1, max_cache_len), dtype=torch.int64),
    }

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        cache_implementation="static",
    )

    params = dynamo_compiler_prefill.imported_params[graphs_prefill[0]]

# Process prefill graph
assert len(graphs_prefill) == 1
graph_prefill = graphs_prefill[0]

graphs_prefill[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)

pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
graphs_prefill[0].fuse_ops(pattern_list_prefill)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

# Save prefill files
with open(
    os.path.join(output_dir, "subgraph0_prefill_mc.mlir"), "w"
) as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)

with open(
    os.path.join(output_dir, "forward_prefill_mc.mlir"), "w"
) as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

# Save parameters (F32)
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0_mc.data"))

print("Prefill subgraph saved successfully!")

# Generate decode graphs for each cache size
data_decode = {
    "input_ids": torch.zeros((1, 1), dtype=torch.int64),
}

pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,
]

for cache_size in cache_sizes:
    print(f"\n=== Generating decode subgraph (cache_len={cache_size}) ===")

    # Reset dynamo state between runs
    torch._dynamo.reset()

    # Create new compiler for each cache size
    dynamo_compiler_decode = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=f"forward_decode_{cache_size}",
    )

    with torch.no_grad():
        # Create StaticCache with current cache size
        past_key_values_decode = StaticCache(
            config=model.config, max_cache_len=cache_size
        )

        # Use a cache_position value similar to original (200 for 1024, scaled for smaller)
        cache_position = torch.tensor(
            [min(200 * cache_size // 1024, cache_size - 1)], dtype=torch.int64
        )

        # Initialize past_key_values once during the first forward call
        model(
            input_ids=data_decode["input_ids"],
            past_key_values=past_key_values_decode,
            use_cache=True,
            cache_implementation="static",
        )

        # Import the model
        graphs_decode = dynamo_compiler_decode.importer(
            model,
            input_ids=data_decode["input_ids"],
            use_cache=True,
            cache_position=cache_position,
            past_key_values=past_key_values_decode,
            cache_implementation="static",
        )

    assert len(graphs_decode) == 1
    graph_decode = graphs_decode[0]

    # Apply transformations
    graphs_decode[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    graphs_decode[0].fuse_ops(pattern_list_decode)

    # Rename subgraph with cache size suffix
    graph_decode.op_groups[f"subgraph0_decode_{cache_size}"] = (
        graph_decode.op_groups.pop("subgraph0")
    )
    graph_decode.group_map_device[f"subgraph0_decode_{cache_size}"] = (
        DeviceType.CPU
    )

    driver_decode = GraphDriver(graphs_decode[0])
    driver_decode.subgraphs[0].lower_to_top_level_ir()

    # Save decode files with cache size suffix
    with open(
        os.path.join(output_dir, f"subgraph0_decode_{cache_size}.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)

    with open(
        os.path.join(output_dir, f"forward_decode_{cache_size}.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)

    print(f"Decode subgraph (cache_size={cache_size}) saved successfully!")

print("\n=== All subgraphs generated successfully! ===")
print(f"Output directory: {output_dir}")
print("Generated files:")
print("  - subgraph0_prefill_mc.mlir")
print("  - forward_prefill_mc.mlir")
for cache_size in cache_sizes:
    print(f"  - subgraph0_decode_{cache_size}.mlir")
    print(f"  - forward_decode_{cache_size}.mlir")
print("  - arg0_mc.data")
