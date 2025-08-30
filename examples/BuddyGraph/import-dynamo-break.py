# ===- import-dynamo-break.py --------------------------------------------------
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
# The example for dynamo graph break, import, and execute.
#
# ===---------------------------------------------------------------------------

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, b, c):
        if torch.nn.functional.silu(b)[0][0]:
            return torch.add(b, c)
        else:
            return torch.matmul(b, c)

# Define a PyTorch model and run it with PyTorch runtime.
model = TestModule()
a, b = torch.randn((1024, 1024)), torch.randn((1024, 1024))
print(model(a, b))

# JIT Mode
# Initialize Buddy Dynamo Compiler to compile and execute the PyTorch model.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions
)
model_opt = torch.compile(model, backend=dynamo_compiler)
print(model_opt(a, b))

torch._dynamo.reset()

# AOT Mode
# Import PyTorch model to Buddy Graph and MLIR/LLVM IR.
graphs = dynamo_compiler.importer(
    model, a, b
)
for g in graphs:
    g.lower_to_top_level_ir()
    print(g._imported_module)
