# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions


def reshape_equal_dimensions(value):
    return value.reshape(1, 2, 8, 4)


compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
)
graphs = compiler.importer(
    reshape_equal_dimensions,
    torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4),
)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK-LABEL: func.func @forward
# CHECK-NOT: tosa.transpose
# CHECK: tosa.reshape
# CHECK-NOT: tosa.transpose
# CHECK: return
