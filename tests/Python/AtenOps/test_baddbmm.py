# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(a, b, c):
    return torch.baddbmm(a, b, c, beta=1.0, alpha=1.0)


batch_size = 2
m, n, k = 3, 4, 5
a = torch.randn(batch_size, m, n)
b = torch.randn(batch_size, m, k)
c = torch.randn(batch_size, k, n)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, a, b, c)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
