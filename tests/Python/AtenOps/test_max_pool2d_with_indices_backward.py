# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._decomp import core_aten_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x):
    """Function to test max_pool2d_with_indices for backward verification."""
    # Max pooling with indices
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    y, indices = pool(x)
    return y


# Test input
in1 = torch.randn(1, 3, 8, 8, requires_grad=True)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=core_aten_decompositions(),
)

graphs = dynamo_compiler.importer(foo, in1)
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
