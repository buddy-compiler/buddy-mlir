# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    # fold requires input shape [N, C*kernel_h*kernel_w, L]
    # output_size = (H, W)
    return F.fold(x, output_size=(4, 4), kernel_size=(2, 2), stride=1)


# Input shape: [N, C*kernel_h*kernel_w, L]
# For output (4,4) with kernel (2,2) stride 1: L = (4-2+1) * (4-2+1) = 9
# C*kernel_h*kernel_w = 1*2*2 = 4
in1 = torch.randn(1, 4, 9, dtype=torch.float32)

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

# CHECK: func.func
# CHECK: tosa.reshape
# CHECK: tosa.transpose
# CHECK: return
