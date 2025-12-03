# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, min_tensor, max_tensor):
    return torch.clamp(x, min=min_tensor, max=max_tensor)


in1 = torch.randn(4, 4)
min_tensor = torch.full((4, 4), -0.5)
max_tensor = torch.full((4, 4), 0.5)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, min_tensor, max_tensor)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.maximum
# CHECK: %{{.*}} = tosa.minimum
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
