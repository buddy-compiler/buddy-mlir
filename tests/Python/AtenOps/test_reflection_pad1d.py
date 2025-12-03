# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x):
    return torch.ops.aten.reflection_pad1d(x, [2, 2])


in1 = torch.randn(1, 3, 8)

# Initialize the dynamo compiler without decomposition to preserve the op
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=None,
)

graphs = dynamo_compiler.importer(foo, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: builtin.module
# CHECK: func.func
# CHECK: tosa.pad
# CHECK: func.return
