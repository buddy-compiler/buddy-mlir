# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    # Compute pairwise distances within a set of vectors
    return torch.ops.aten._pdist_forward(x, 2.0)


# Set of vectors
x = torch.randn(5, 3)  # 5 vectors of dim 3
# Output size: 5*(5-1)/2 = 10

# Initialize the dynamo compiler without decomposition
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=None,
)

graphs = dynamo_compiler.importer(foo, x)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module
# CHECK: func.func
# CHECK: scf.for
# CHECK: math.sqrt
# CHECK: return
