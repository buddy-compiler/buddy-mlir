# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x1, x2):
    # Compute pairwise distances using L2 norm
    return torch.ops.aten._cdist_forward(x1, x2, 2.0, None)


# Two sets of vectors
x1 = torch.randn(5, 3)  # 5 vectors of dim 3
x2 = torch.randn(4, 3)  # 4 vectors of dim 3

# Initialize the dynamo compiler without decomposition
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=None,
)

graphs = dynamo_compiler.importer(foo, x1, x2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module
# CHECK: func.func
# CHECK: tosa.const
# CHECK: return
