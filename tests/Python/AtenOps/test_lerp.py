# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(start, end, weight):
    return torch.lerp(start, end, weight)


start = torch.tensor([1.0, 2.0, 3.0, 4.0])
end = torch.tensor([5.0, 6.0, 7.0, 8.0])
weight = torch.tensor([0.0, 0.25, 0.5, 1.0])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, start, end, weight)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.sub
# CHECK: tosa.add
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
