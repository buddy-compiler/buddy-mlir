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
    flash_attention,
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

# Initialize the model from the specified model path.
if args.precision == "f16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)
        .eval()
        .half()
    )
elif args.precision == "bf16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)
        .eval()
        .bfloat16()
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torchscript=True
    ).eval()
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
if args.precision == "f16":
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
else:
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
    if args.precision == "f16":
        data = {
            "input_ids": torch.zeros((1, 20), dtype=torch.int64),
        }
        graphs = dynamo_compiler.importer(
            model,
            input_ids=data["input_ids"],
        )
    else:
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

if args.precision == "f16":
    assert len(graphs) == 1
    graph = graphs[0]
    graph.perform([eliminate_transpose])
    params = dynamo_compiler.imported_params[graph]
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()
else:
    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    graph_prefill = graphs_prefill[0]
    graph_decode = graphs_decode[0]

    params = dynamo_compiler_prefill.imported_params[graph_prefill]
    graphs_prefill[0].perform([eliminate_transpose])
    graphs_decode[0].perform([eliminate_transpose])
    pattern_list_prefill = [simply_fuse]
    pattern_list_decode = [simply_fuse,flash_attention]

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

    driver_prefill = GraphDriver(graphs_prefill[0])
    driver_prefill.subgraphs[0].lower_to_top_level_ir()

    driver_decode = GraphDriver(graphs_decode[0])
    driver_decode.subgraphs[0].lower_to_top_level_ir()

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
elif args.precision == "bf16":
    with open(
        os.path.join(output_dir, "subgraph0-bf16.mlir"), "w"
    ) as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward-bf16.mlir"), "w"
    ) as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    # Convert BF16 parameters to float32 first, then to numpy
    all_param = numpy.concatenate(
        [param.detach().float().numpy().reshape([-1]) for param in params]
    )
    # Convert float32 to BF16 format (uint16) for storage
    all_param_bf16 = numpy.frombuffer(
        all_param.astype(numpy.float32).tobytes(), dtype=numpy.uint16
    )[1::2]
    all_param_bf16.tofile(os.path.join(output_dir, "arg0-bf16.data"))
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
