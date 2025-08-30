# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import math


def foo(x):
    return torch.ops.aten.sqrt(x)


x = torch.randn(10, 3, 6)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=math.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = torch.compile(foo, backend=dynamo_compiler)
assert torch.allclose(foo_mlir(x), foo(x), equal_nan=True)

graphs = dynamo_compiler._imported_graphs
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = math.sqrt
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
