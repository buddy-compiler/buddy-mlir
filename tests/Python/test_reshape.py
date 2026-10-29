# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def foo(x, new_shape):
    return torch.ops.aten.reshape(x, new_shape)


x = torch.randn(2, 3)
new_shape = (3, 2)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, x, new_shape)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.empty() : tensor<3x2xf32>
# CHECK: %{{.*}} = linalg.generic
# CHECK: arith.remui
# CHECK: tensor.extract
# CHECK: linalg.yield
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
