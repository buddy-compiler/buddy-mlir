# ===- import-resnet18.py ------------------------------------------------------
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
# This is the test of resnet18 model.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path

import torch
import numpy
import torchvision
import torch._dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.graph import Graph
from buddy.compiler.graph.graph_driver import GraphDriver
from buddy.compiler.graph.json_encoder import BuddyGraphEncoder, GraphList
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa


from torch import nn
from torch.nn.functional import max_pool2d, relu

import json
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x
    
model = LeNet()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

data = torch.randn((1, 1, 28, 28))
# Import the model into MLIR module and parameters.

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.lower_to_top_level_ir()
print(graph._imported_module)
print(len(graph._body))
graph.graph2dot()
graph_list = GraphList(graph)
json_str = json.dumps(graph_list, cls=BuddyGraphEncoder)
print(json_str)

current_path = os.path.dirname(os.path.abspath(__file__))
print(graph.init_op_group(Path(current_path) / "subgraphs.json"))

# subgraph1 = []
# subgraph2 = []

# for i in range(len(graph._body)):
#     if i < 6 or 11 < i < 18:
#         subgraph1.append(graph._body[i])
#         print(graph._body[i]._name)
#     else:
#         subgraph2.append(graph._body[i])

# subgraph_name = "subgraph{}".format(0)
# graph.group_map_device[subgraph_name] = DeviceType.CPU
# graph.op_groups[subgraph_name] = subgraph1

# subgraph_name = "subgraph{}".format(1)
# graph.group_map_device[subgraph_name] = DeviceType.CPU
# graph.op_groups[subgraph_name] = subgraph2



driver = GraphDriver(graph)

print(len(driver.subgraphs[0]._body))
print(len(driver.subgraphs[1]._body))
json_str = json.dumps(driver.subgraphs[0], cls=BuddyGraphEncoder)
print(json_str)
json_str = json.dumps(driver.subgraphs[1], cls=BuddyGraphEncoder)
print(json_str)



# driver.subgraphs[0].lower_to_top_level_ir()
# path_prefix = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
#     print(driver.subgraphs[0]._imported_module, file=module_file)

# driver.subgraphs[1].lower_to_top_level_ir()
# path_prefix = os.path.dirname(os.path.abspath(__file__))
# with open(os.path.join(path_prefix, "subgraph1.mlir"), "w") as module_file:
#     print(driver.subgraphs[1]._imported_module, file=module_file)

current_path = os.path.dirname(os.path.abspath(__file__))

with open(Path(current_path) / "lenet.json", "w") as module_file:
    module_file.write(json_str)

with open(Path(current_path) / "lenet.mlir", "w") as module_file:
    module_file.write(str(graph._imported_module))


all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)

all_param.tofile(Path(current_path) / "arg0.data")
