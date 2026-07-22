#!/usr/bin/env python3
# ===- import-deepseek-r1-gpu.py -----------------------------------------------
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
# This is the GPU AOT importer for the DeepSeekR1 model.
# The subgraph is assigned DeviceType.GPU so that the GraphDriver emits MLIR
# suitable for the GPU lowering pipeline (parallel-loops → NVVM).
#
# ===---------------------------------------------------------------------------

import argparse
import os

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import (
    eliminate_matmul_transpose_reshape,
    eliminate_transpose,
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
    description="DeepSeekR1 GPU Model AOT Importer"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(model_path).eval().float()
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
        cache_implementation="static",
    )
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

graphs_prefill[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
graphs_decode[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)

pattern_list_prefill = [
    simply_fuse,
]
pattern_list_decode = [
    simply_fuse,
]

graphs_prefill[0].fuse_ops(pattern_list_prefill)
graphs_decode[0].fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.GPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.GPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()

with open(
    os.path.join(output_dir, "subgraph0_prefill.mlir"), "w"
) as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward_prefill.mlir"), "w") as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

with open(
    os.path.join(output_dir, "subgraph0_decode.mlir"), "w"
) as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward_decode.mlir"), "w") as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))
