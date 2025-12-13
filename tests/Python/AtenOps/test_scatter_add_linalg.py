# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, index, src):
    """Function to test scatter_add operator."""
    return x.scatter_add(0, index, src)


# Test input: 2D tensor
x = torch.zeros(3, 3)
index = torch.tensor([[0, 1, 2], [0, 1, 2]])
src = torch.ones(2, 3)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, x, index, src)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: memref.alloc
# CHECK: scf.for
# CHECK: bufferization.to_tensor
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
