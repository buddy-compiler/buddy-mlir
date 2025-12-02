# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, bn):
    """Test batch normalization with running statistics."""
    return bn(x)


# Create test input: (N=2, C=4, H=8, W=8)
x = torch.randn(2, 4, 8, 8)

# Create BatchNorm2d in eval mode
bn = torch.nn.BatchNorm2d(4, track_running_stats=True)
bn.eval()

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition={},
)

graphs = dynamo_compiler.importer(foo, x, bn)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.sub
# CHECK: %{{.*}} = tosa.rsqrt
# CHECK: %{{.*}} = tosa.mul
# CHECK: %{{.*}} = tosa.add
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
