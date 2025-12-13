# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, weight, bias):
    """Function to test native_layer_norm for backward verification."""
    # Layer normalization
    y = torch.nn.functional.layer_norm(
        x, normalized_shape=[4, 4], weight=weight, bias=bias, eps=1e-5
    )
    return y


# Test input
in1 = torch.randn(2, 3, 4, 4, requires_grad=True)
weight = torch.randn(4, 4, requires_grad=True)
bias = torch.randn(4, 4, requires_grad=True)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, weight, bias)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.{{reshape|reduce_sum|mul}}
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
