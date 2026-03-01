# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y):
    return torch.masked_fill(x, y, 0.0)


in1 = torch.randn(3, 4)
mask = torch.tensor(
    [
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False],
    ]
)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, mask)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: func.func
# CHECK: arith.constant
# CHECK: tensor.empty
# CHECK: linalg.generic
# CHECK: arith.select
# CHECK: return
