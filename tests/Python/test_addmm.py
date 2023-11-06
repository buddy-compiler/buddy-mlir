# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y, z):
    return torch.ops.aten.addmm(z, x, y)


in1 = torch.randn(4, 2)
in2 = torch.randn(2, 4)
in3 = torch.randn(4, 4)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(in1, in2, in3)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.matmul
# CHECK: %{{.*}} = tosa.add
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)
