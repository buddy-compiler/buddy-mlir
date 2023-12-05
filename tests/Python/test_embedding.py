# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(weight, indices):
    return torch.ops.aten.embedding(weight, indices)


# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# test trivial case
weight = torch.randn(10, 5)
indices = torch.randint(10, (3, 3)).to(torch.int32)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(weight, indices)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.gather
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)


# test cast case
weight = torch.randn(10, 5)
indices = torch.randint(10, (3, 3)).to(torch.int64)


foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(weight, indices)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.cast
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.gather
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)
