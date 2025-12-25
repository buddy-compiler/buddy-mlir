# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return torch.nn.functional.softplus(x)


in1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

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

# CHECK: func.func @forward
# CHECK: tosa.exp
# CHECK: math.log1p
# CHECK: tosa.greater
# CHECK: tensor.empty
# CHECK: linalg.generic
# CHECK: arith.select
# CHECK: return
