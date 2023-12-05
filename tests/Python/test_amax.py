# RUN: %PYTHON %s 2>&1 | FileCheck %s

import random
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, dim):
    return torch.ops.aten.amax(x, dim, True)


in1 = torch.randn(4, 5, 2, 9)
dim = [random.randint(0, 3)]

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(in1, dim)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reduce_max
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)
