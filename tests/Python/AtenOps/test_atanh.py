# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import math


def foo(x):
    return torch.atanh(x)


# atanh requires input in (-1, 1)
in1 = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=math.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: math.atanh
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
