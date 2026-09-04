#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
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
# Weather-LLM-SFT Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: LlamaForCausalLM (1.26B params)
#   - 24 layers, hidden=2048, heads=16, kv_heads=8, head_dim=128
#
# ===---------------------------------------------------------------------------

import argparse
import os
import types

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
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
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)

# ==============================================================================
# 1. Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description="Weather-LLM-SFT Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f32",
    choices=["f32"],
    help="Precision mode. Currently only 'f32' is supported.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 2. Load model
# ==============================================================================

model_path = os.environ.get("WEATHER_LLM_MODEL_PATH")
if model_path is None:
    model_path = "AuraWorxAI/weather-llm-sft"

print("[WeatherLLM-Import] Loading LlamaForCausalLM model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
).eval()
model.config.use_cache = False

# Unwrap HF decorators to expose raw forward function
for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

print(f"   hidden_size = {model.config.hidden_size}")
print(f"   num_hidden_layers = {model.config.num_hidden_layers}")
print(f"   num_kv_heads = {model.config.num_key_value_heads}")
print(f"   vocab_size = {model.config.vocab_size}")

# ==============================================================================
# 3. Initialize Dynamo Compilers
# ==============================================================================

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

# ==============================================================================
# 4. Dummy inputs
# ==============================================================================

max_seq_len = 1024

data_prefill = {
    "input_ids": torch.zeros((1, max_seq_len), dtype=torch.int64),
}

data_decode = {
    "input_ids": torch.zeros((1, 1), dtype=torch.int64),
}

cache_position = torch.tensor([200], dtype=torch.int64)
cache_position_prefill = torch.arange(max_seq_len, dtype=torch.int64)

print(f"[WeatherLLM-Import] Dummy inputs prepared.")
print(f"   prefill input_ids:  {data_prefill['input_ids'].shape}")
print(f"   decode input_ids:   {data_decode['input_ids'].shape}")

# ==============================================================================
# 5. Trace the model
# ==============================================================================

print("\n[WeatherLLM-Import] Tracing prefill graph...")
past_key_values_prefill = StaticCache(
    config=model.config, max_cache_len=max_seq_len
)
past_key_values_decode = StaticCache(
    config=model.config, max_cache_len=max_seq_len
)

with torch.no_grad():
    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        past_key_values=past_key_values_prefill,
        cache_position=cache_position_prefill,
        cache_implementation="static",
    )

    # Initialize decode KV cache with a warm-up forward pass
    model(
        input_ids=data_decode["input_ids"],
        past_key_values=past_key_values_decode,
        use_cache=True,
        cache_implementation="static",
    )

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=data_decode["input_ids"],
        use_cache=True,
        cache_position=cache_position,
        past_key_values=past_key_values_decode,
        cache_implementation="static",
    )

assert len(graphs_prefill) == 1, f"Expected 1 prefill graph, got {len(graphs_prefill)}"
assert len(graphs_decode) == 1, f"Expected 1 decode graph, got {len(graphs_decode)}"
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]
print(f"[WeatherLLM-Import] Graphs captured. Params: {len(params)} tensors.")

# ==============================================================================
# 6. Graph optimizations
# ==============================================================================

print("[WeatherLLM-Import] Running graph transforms...")
graph_prefill.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
graph_decode.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,
]

graph_prefill.fuse_ops(pattern_list_prefill)
graph_decode.fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop("subgraph0")
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop("subgraph0")
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graph_prefill)
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graph_decode)
driver_decode.subgraphs[0].lower_to_top_level_ir()

# ==============================================================================
# 7. Save outputs
# ==============================================================================

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)
print(f"\n[WeatherLLM-Import] Writing MLIR files to: {layer_dir}")

with open(os.path.join(layer_dir, "subgraph0_prefill.mlir"), "w") as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_prefill.mlir"), "w") as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

with open(os.path.join(layer_dir, "subgraph0_decode.mlir"), "w") as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_decode.mlir"), "w") as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

print(f"[WeatherLLM-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[WeatherLLM-Import] Done!\n")
