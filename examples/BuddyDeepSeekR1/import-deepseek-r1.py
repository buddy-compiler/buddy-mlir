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
import time
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
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
    choices=["f32", "f16"],
    help="Precision mode for generated MLIR and input data. Choose from 'f32' or 'f16'.",
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Initialize the model from the specified model path.
if args.precision == "f16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)
        .eval()
        .half()
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torchscript=True
    ).eval()
model.config.use_cache = True

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
    # verbose=True,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    past_key_values_prefill = StaticCache(config=model.config, max_cache_len=40)
    past_key_values_decode = StaticCache(config=model.config, max_cache_len=512)
    # print(past_key_values_decode)
    # for i in past_key_values_decode:
    #     print("debug1:")
    #     print(i)
    # print(past_key_values)
    data_prefill = {
        "input_ids": torch.zeros((1, 40), dtype=torch.int64),
        # "attention_mask": torch.zeros((1, 40), dtype=torch.int64),
    }
    data_decode = {
        "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        # "attention_mask": torch.zeros((1, 1), dtype=torch.int64),
    }

    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        # attention_mask=data_prefill["attention_mask"],
        past_key_values=past_key_values_prefill,
        use_cache=True,
        cache_implementation="static",
    )
    # print(past_key_values_decode)
    model(
        input_ids=data_decode["input_ids"],
        past_key_values=past_key_values_decode,
        use_cache=True,
        cache_implementation="static",
    )
    # print(past_key_values_decode)

    outputs = model(
        input_ids=data_decode["input_ids"],
        past_key_values=past_key_values_decode,
        use_cache=True,
        # cache_position=cache_position,   # ✅ 显式传入位置
        cache_implementation="static",
    )
    # for i in past_key_values_decode:
    #     print("debug2:")
    #     print(i)

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=data_decode["input_ids"],
        cache_position=cache_position,
        # attention_mask=data_decode["attention_mask"],
        past_key_values=past_key_values_decode,
        use_cache=True,
        fullgraph=True,
        cache_implementation="static",
    )

assert len(graphs_prefill) == 1
assert len(graphs_decode) == 1
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]


group_prefill = []
for op in graph_prefill.body:
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
        continue
    group_prefill.append(op)
graph_prefill.op_groups["subgraph0_prefill"] = group_prefill
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

group_decode = []
for op in graph_decode.body:
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
        continue
    group_decode.append(op)
graph_decode.op_groups["subgraph0_decode"] = group_decode
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

params = dynamo_compiler_prefill.imported_params[graph_prefill]
pattern_list = [simply_fuse]

graphs_prefill[0].fuse_ops(pattern_list)
graphs_decode[0].fuse_ops(pattern_list)

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()
driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()
# print(dir(graph_decode._fake_params[0]))
# print("fake_params:")
# for i in graph_decode._fake_params:
#     print(i.dtype)
#     print(i.shape)


# Save the generated files to the specified output directory.
if args.precision == "f16":
    with open(
        os.path.join(output_dir, "subgraph0-f16.mlir"), "w"
    ) as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward-f16.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0-f16.data"))
else:
    with open(
        os.path.join(output_dir, "subgraph0_prefill.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_prefill.mlir"), "w"
    ) as module_file:
        print(driver_prefill.construct_main_graph(True), file=module_file)
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))

    with open(
        os.path.join(output_dir, "subgraph0_decode.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
