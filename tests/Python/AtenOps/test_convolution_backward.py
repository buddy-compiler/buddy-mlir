# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, weight):
    """Function to test convolution backward."""
    # Use functional conv2d
    y = torch.nn.functional.conv2d(x, weight, padding=1)
    return y


# Test input
in1 = torch.randn(1, 3, 8, 8, requires_grad=True)
weight = torch.randn(16, 3, 3, 3, requires_grad=True)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, weight)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.conv2d
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
