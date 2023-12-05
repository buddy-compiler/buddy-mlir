# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return torch.ops.aten.var_mean(x)


def foo_keepdim(x):
    return torch.ops.aten.var_mean(x, keepdim=True)


x = torch.randn(10, 2, 4)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
foo_mlir(x)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.reciprocal
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.sub
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.reciprocal
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}} : tensor<f32>, tensor<f32>
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)

foo_keepdim_mlir = dynamo.optimize(dynamo_compiler)(foo_keepdim)
foo_keepdim_mlir(x)
# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.reciprocal
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.sub
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.reciprocal
# CHECK: %{{.*}} = tosa.mul
# CHECK: return %{{.*}} : tensor<1x1x1xf32>, tensor<1x1x1xf32>
# CHECK: }
# CHECK: }
print(dynamo_compiler.imported_module)
