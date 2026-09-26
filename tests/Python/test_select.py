# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def foo(x, dim, index):
    return torch.ops.aten.select(x, dim, index)


x = torch.randn(3, 5, 2)
dim = 1
index = 2

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, x, dim, index)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[0, 2, 0] [3, 1, 2] [1, 1, 1] : tensor<3x5x2xf32> to tensor<3x2xf32>
# CHECK: return %{{.*}} : tensor<3x2xf32>
# CHECK: }
# CHECK: }
