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

def foo(x, y):
    return torch.matmul(x, y)

in1 = torch.ones([5376, 2048], dtype=torch.float32)
in2 = torch.ones([2048, 5376], dtype=torch.float32)
# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()

path_prefix = os.path.dirname(os.path.abspath(__file__))
# Write the MLIR module to the file.
with open(os.path.join(path_prefix, "matmul.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)
