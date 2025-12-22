# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


# Input: [N, C, H, W]
in1 = torch.randn(1, 3, 4, 4, dtype=torch.float32)

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
# CHECK: return
# CHECK: }
# CHECK: }
