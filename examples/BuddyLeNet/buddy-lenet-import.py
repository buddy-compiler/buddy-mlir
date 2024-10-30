# ===- buddy-lenet-import.py ---------------------------------------------------
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
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path
import argparse

import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    gpu_fuse,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from buddy.compiler.graph.json_decoder import json_to_graph
from buddy.compiler.graph.operation import *
from model import LeNet

# Retrieve the LeNet model path from environment variables.
model_path = os.environ.get("LENET_EXAMPLE_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LENET_MODEL_PATH' is not set or is invalid."
    )

model = LeNet()
model = torch.load(model_path + "/lenet-model.pth")
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

data = torch.randn([1, 1, 28, 28])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--DEVICE_TYPE", type=str, required=True, choices=["cpu", "heter"]
)
args = parser.parse_args()
type = args.DEVICE_TYPE

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

if type == "cpu":
    pattern_list = [simply_fuse]
    graph.fuse_ops(pattern_list)
elif type == "heter":
    group = []
    for i, op in enumerate(graph._body):
        if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp) or i == 25:
            continue
        group.append(op)
        subgraph_name = "subgraph0"
        graph.group_map_device[subgraph_name] = DeviceType.CPU
        graph.op_groups[subgraph_name] = group
    new_group = [graph._body[25]]
    subgraph_name = "subgraph1"
    graph.group_map_device[subgraph_name] = DeviceType.GPU
    graph.op_groups[subgraph_name] = new_group
path_prefix = os.path.dirname(os.path.abspath(__file__))
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
with open(
    os.path.join(path_prefix, f"subgraph0-{type}.mlir"), "w"
) as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
# Add heterogeneous hardware partition
if type == "heter":
    driver.subgraphs[1].lower_to_top_level_ir()
    with open(
        os.path.join(path_prefix, f"subgraph1-{type}.mlir"), "w"
    ) as module_file:
        print(driver.subgraphs[1]._imported_module, file=module_file)
with open(
    os.path.join(path_prefix, f"forward-{type}.mlir"), "w"
) as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]
current_path = os.path.dirname(os.path.abspath(__file__))

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)

float32_param.tofile(Path(current_path) / "arg0.data")
