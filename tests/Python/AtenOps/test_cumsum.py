# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg
from torch._functorch.aot_autograd import aot_autograd_decompositions


def foo(x):
    return torch.cumsum(x, dim=1)


in1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.empty
# CHECK: %{{.*}} = linalg.fill
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
# CHECK: linalg.index 1
# CHECK: linalg.index 2
# CHECK: arith.cmpi ule
# CHECK: arith.select
# CHECK: arith.addf
# CHECK: linalg.yield
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
