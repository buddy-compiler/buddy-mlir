# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(weight, indices):
    return torch.ops.aten.embedding(weight, indices)


# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# test trivial case
weight = torch.randn(10, 5)
indices = torch.randint(10, (3, 3)).to(torch.int32)

graphs = dynamo_compiler.importer(foo, weight, indices)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.gather
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }

# test cast case
weight = torch.randn(10, 5)
indices = torch.randint(10, (3, 3)).to(torch.int64)

graphs = dynamo_compiler.importer(foo, weight, indices)
print(graphs)
assert len(graphs) == 2
graphs[0].lower_to_top_level_ir()
print(graphs[0]._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.gather
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }

graphs[1].lower_to_top_level_ir()
print(graphs[1]._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.cast
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.gather
# CHECK: %{{.*}} = tosa.reshape
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }