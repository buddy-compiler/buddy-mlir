#!/usr/bin/env python3
# ===- import-deepseek-r1.py ---------------------------------------------------
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
# This is the test of DeepSeekR1 model.
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

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import SplitStrategy, GraphDriver
# from buddy.compiler.graph import GraphDriver
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

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="DeepSeekR1 Model AOT Importer")
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
    choices=["f32", "f16", "bf16"],
    help="Precision mode for generated MLIR and input data. Choose from 'f32', 'f16', or 'bf16'.",
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


model = AutoModelForCausalLM.from_pretrained(
    model_path, torchscript=True
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

# Import the model into MLIR module and parameters.
with torch.no_grad():
    past_key_values_prefill = StaticCache(
        config=model.config, max_cache_len=1024
    )
    past_key_values_decode = StaticCache(
        config=model.config, max_cache_len=1024
    )

    data_prefill = {
        "input_ids": torch.zeros((1, 1024), dtype=torch.int64),
    }
    data_decode = {
        "input_ids": torch.zeros((1, 1), dtype=torch.int64),
    }

    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        # past_key_values=past_key_values_prefill,
        cache_implementation="static",
    )
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


assert len(graphs_prefill) == 1
assert len(graphs_decode) == 1
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]
# Enable verbose mode for debugging eliminate_matmul_transpose_reshape
graphs_prefill[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
graphs_decode[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
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

graphs_prefill[0].fuse_ops(pattern_list_prefill)
graphs_decode[0].fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU


DECODE_STRATEGY = SplitStrategy(
    name="decode",
    parallel_num=2,
    # ops_count=[56,19],
    ops_count=[6, 48, 2, 6, 11, 2],
    stage_boundary_op=PowOp,
    stage_boundary_op_num = 57,
    paral_input_positions={
        0: [-1, -1, -1, -1],
        169: [-1, -1, -1],
        "default": [
            [-1, -1],
            [1,0,1,0,1,0,0,-1,1,1,-1,-1,-1,-1],
            [-1, -1],
            [-1, -1],
            [1, 1, 0, -1],
            [-1, -1]
        ]
    }
)

PREFILL_STRATEGY = SplitStrategy(
    name="prefill",
    parallel_num=2,
    ops_count=[6, 61, 2, 6, 11, 2],
    stage_boundary_op=PowOp,
    stage_boundary_op_num = 57 ,
    paral_input_positions={
        0: [-1, -1, -1],
        169: [-1, -1, -1],
        "default": [
            [-1, 1],
            [1,0,1,0,1,0,0,-1,-1,-1,-1],
            [1, 0],
            [-1, 1], [1, 1, 0, -1],[1, 0]
        ]
    }
)

driver_prefill = GraphDriver(graphs_prefill[0], PREFILL_STRATEGY)
for i in range(len(driver_prefill.subgraphs)):
    driver_prefill.subgraphs[i].lower_to_top_level_ir()
driver_prefill.construct_main_graph(True)
for i in range(len(driver_prefill.subgraphs)): 
    with open(os.path.join(output_dir, f"subgraph0_prefill{i}.mlir"), "w") as module_file:
        print(driver_prefill.subgraphs[i]._imported_module, file=module_file)
    with open(os.path.join(output_dir, f"forward_prefill{i}.mlir"), "w") as module_file:
        print(driver_prefill.modules[i], file=module_file)
for entry in driver_prefill._subgraph_param_info.items():
    driver_prefill.construct_sub_params(params, entry, output_dir)


driver_decode = GraphDriver(graphs_decode[0], DECODE_STRATEGY)
for i in range(len(driver_decode.subgraphs)):
    driver_decode.subgraphs[i].lower_to_top_level_ir()
driver_decode.construct_main_graph(True)
for i in range(len(driver_decode.subgraphs)): 
    with open(os.path.join(output_dir, f"subgraph0_decode{i}.mlir"), "w") as module_file:
        print(driver_decode.subgraphs[i]._imported_module, file=module_file)
    with open(os.path.join(output_dir, f"forward_decode{i}.mlir"), "w") as module_file:
        print(driver_decode.modules[i], file=module_file)