# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x):
    """Function to test avg_pool3d operator."""
    return torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)


# Test input: [N, C, D, H, W] = [1, 2, 4, 4, 4]
in1 = torch.randn(1, 2, 4, 4, 4)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
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
