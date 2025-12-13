# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class AdaptiveAvgPool2dBackwardModule(torch.nn.Module):
    """Test module for adaptive_avg_pool2d backward pass."""

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        # Forward pass
        y = self.pool(x)
        # Backward pass (sum to create scalar loss)
        loss = y.sum()
        return loss


def foo(x):
    """Function to test adaptive_avg_pool2d backward."""
    pool = torch.nn.AdaptiveAvgPool2d((2, 2))
    y = pool(x)
    # Return pooled output directly for IR verification
    return y


# Test input
in1 = torch.randn(1, 3, 8, 8, requires_grad=True)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.transpose
# CHECK: tosa.avg_pool2d
# CHECK: tosa.transpose
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
