# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, index, y):
    return x.scatter_(dim=0, index=index, src=y)


in1 = torch.zeros(3, 5)
in2 = torch.arange(1, 11).reshape(2, 5).float()
in3 = torch.tensor([[0, 0, 0, 0, 0], [2, 2, 2, 2, 2]], dtype=torch.int64)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in3, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
