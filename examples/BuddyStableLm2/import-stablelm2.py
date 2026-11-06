#!/usr/bin/env python3
# ===- import-stablelm2.py ------------------------------------------------
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
# ===--------------------------------------------------------------------
#
# This is the AOT importer of the StableLM-2 1.6B model.
#
# ===--------------------------------------------------------------------

import argparse
import os

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_matmul_transpose_reshape,
    eliminate_transpose,
    flash_attention_prefill,
    simply_fuse,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)

parser = argparse.ArgumentParser(
    description="StableLM2-1.6B Model AOT Importer"
)
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
    help="Precision mode (kept for CMake parity with the Qwen3 example).",
)
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Note: env var name must not contain '-' (bash cannot export such names;
# import-qwen3.py has that latent bug).
model_path = os.environ.get("STABLELM2_MODEL_PATH")
if model_path is None:
    model_path = "stabilityai/stablelm-2-1_6b"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
).eval()
model.config.use_cache = False

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

MaxTokenLength = 1024

with torch.no_grad():
    past_key_values_prefill = StaticCache(
        config=model.config, max_cache_len=MaxTokenLength
    )
    past_key_values_decode = StaticCache(
        config=model.config, max_cache_len=MaxTokenLength
    )

    data_prefill = {
        "input_ids": torch.zeros((1, MaxTokenLength), dtype=torch.int64),
    }
    data_decode = {
        "input_ids": torch.zeros((1, 1), dtype=torch.int64),
    }

    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        cache_implementation="static",
    )
    print("prefill graphs:", len(graphs_prefill), flush=True)

    # Initialize past_key_values once during the first forward call
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
    print("decode graphs:", len(graphs_decode), flush=True)

assert len(graphs_prefill) == 1, "PREFILL GRAPH BREAK"
assert len(graphs_decode) == 1, "DECODE GRAPH BREAK"
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]
total_elements = sum(int(p.numel()) for p in params)
print("param tensors:", len(params), "total elements:", total_elements)
print("ParamsSize for main.cpp:", total_elements)

graph_prefill.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
graph_decode.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

# StableLM-2 is MHA (num_key_value_heads == num_attention_heads): the
# gqa_attention_fusion pattern (View<-Clone<-Expand<-...) never matches, so
# it is intentionally not imported here.
pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
]

graph_prefill.fuse_ops(pattern_list_prefill)
graph_decode.fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()

S = "stablelm2_1_6b"
with open(
    os.path.join(output_dir, f"subgraph0_prefill_{S}.mlir"), "w"
) as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(
    os.path.join(output_dir, f"forward_prefill_{S}.mlir"), "w"
) as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

with open(
    os.path.join(output_dir, f"subgraph0_decode_{S}.mlir"), "w"
) as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(
    os.path.join(output_dir, f"forward_decode_{S}.mlir"), "w"
) as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

# Streaming write: concatenate+tofile built a second 6.58 GB buffer and
# peaked at 30.8/31.3 GiB on this board. Writing one param at a time keeps
# the exact same bytes on disk without the copy.
with open(os.path.join(output_dir, f"arg0_{S}.data"), "wb") as data_file:
    for param in params:
        data_file.write(param.detach().numpy().tobytes())
print("WROTE:", sorted(os.listdir(output_dir)))
print("IMPORT_STABLELM2_DONE")
