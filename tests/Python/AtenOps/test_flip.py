# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return torch.flip(x, [0, 1])


in1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Initialize the dynamo compiler.
# Note: Use empty decomposition because inductor_decomp decomposes flip into
# prims.rev.default which is not yet implemented.
# TODO: Implement prims.rev.default to support inductor_decomp.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition={},
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.reverse
# CHECK: tosa.reverse
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
