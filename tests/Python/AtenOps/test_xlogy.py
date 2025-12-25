# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y):
    return torch.xlogy(x, y)


in1 = torch.randn(3, 4)
in2 = torch.randn(3, 4).abs() + 0.1

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: func.func
# CHECK: arith.constant
# CHECK: tensor.splat
# CHECK: arith.cmpf
# CHECK: tosa.log
# CHECK: tosa.mul
# CHECK: tensor.empty
# CHECK: linalg.generic
# CHECK: arith.select
# CHECK: return
