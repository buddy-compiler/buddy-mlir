# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y, z):
    return torch.stack([x, y, z], dim=0)


in1 = torch.randn(4, 4)
in2 = torch.randn(4, 4)
in3 = torch.randn(4, 4)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2, in3)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: func.func @forward
# CHECK: tensor.insert_slice
# CHECK: tosa.reshape
# CHECK: return
