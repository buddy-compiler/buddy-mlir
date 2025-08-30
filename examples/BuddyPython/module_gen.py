# ===- module_gen.py -----------------------------------------------------------
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
# The example of MLIR module generation.
#
# ===---------------------------------------------------------------------------

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


# Define the target function or model.
def foo(x, y):
    return x * y + x


# Define the input data.
float32_in1 = torch.randn(10).to(torch.float32)
float32_in2 = torch.randn(10).to(torch.float32)
int32_in1 = torch.randint(0, 10, (10,)).to(torch.int32)
int32_in2 = torch.randint(0, 10, (10,)).to(torch.int32)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Pass the function and input data to the dynamo compiler's importer, the
# importer will first build a graph. Then, lower the graph to top-level IR.
# (tosa, linalg, etc.). Finally, accepts the generated module and weight parameters.
graphs = dynamo_compiler.importer(foo, float32_in1, float32_in2)
graph = graphs[0]
graph.lower_to_top_level_ir()

print(graph._imported_module)
