# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform import eliminate_transpose


def foo(x, weight):
    """
    Test function: transpose a weight parameter (function argument) and use it.
    The transpose should be eliminated by the optimization since the weight is
    only used in this one place.
    """
    transposed_weight = torch.transpose(weight, 0, 1)
    return torch.matmul(x, transposed_weight)


x = torch.ones([4, 3], dtype=torch.float32)
weight = torch.ones([5, 3], dtype=torch.float32)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, x, weight)
assert len(graphs) == 1
graph = graphs[0]

# Apply the optimization to eliminate weight transpose
graph.perform([eliminate_transpose])

graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# The transpose operation should be eliminated, so we should NOT see tosa.transpose
# CHECK-NOT: tosa.transpose
# We should see matmul operation directly using the transposed weight shape
# CHECK: tosa.matmul
# CHECK: return
# CHECK: }
# CHECK: }
