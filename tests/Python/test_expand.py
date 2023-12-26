# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, new_size):
    return torch.ops.aten.expand(x, new_size)

x = torch.randn(1, 3)
new_size = (6, 3)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(x, new_size)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.add
# CHECK: return %{{.*}} : tensor<6x3xf32>
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)
