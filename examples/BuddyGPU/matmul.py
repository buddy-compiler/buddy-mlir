# ===- matmul.py --------------------------------------------------------------
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
# ===--------------------------------------------------------------------------
#
# This file demonstrates the usage of Buddy's frontend for PyTorch module.
#
# ===--------------------------------------------------------------------------

import os
import time

import numpy
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._functorch.aot_autograd import aot_autograd_decompositions
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

dtype = torch.float32

# model with simple Matmul
class MatmulModule(torch.nn.Module):
    def __init__(self, sizes):
        super(MatmulModule, self).__init__()
        assert len(sizes) == 3
        self.sizes = sizes
        self.weight = torch.nn.Parameter(torch.randn(sizes[1], sizes[2]))

    def forward(self, input):
        assert input.shape[0] == self.sizes[0]
        assert input.shape[1] == self.sizes[1]
        return torch.matmul(input, self.weight)

model = MatmulModule((16, 16, 16))
model.type(dtype)

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    a_shape = model.sizes[:2]
    data = torch.randn(*a_shape, dtype=dtype)
    graphs = dynamo_compiler.importer(model, data)

graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.lower_to_top_level_ir(True)
path_prefix = os.path.dirname(os.path.abspath(__file__))
# Write the MLIR module to the file.
with open(os.path.join(path_prefix, "matmul.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)
