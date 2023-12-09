# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_autograd_decompositions
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x):
    return torch.arange(x)


in1 = 13

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
# CHECK: %{{.*}} = arith.constant
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
