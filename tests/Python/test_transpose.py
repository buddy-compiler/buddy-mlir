# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y, z):
    return torch.transpose(x, y, z)


in1 = torch.ones([13, 13, 13], dtype=torch.float32)
in2 = 1
in3 = 2
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

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = "tosa.const"()
# CHECK: %{{.*}} = tosa.transpose
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
