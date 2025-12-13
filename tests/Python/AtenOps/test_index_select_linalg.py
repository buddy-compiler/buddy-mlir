# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, idx):
    """Test index_select on dim=1 (the previously failing case)."""
    return torch.index_select(x, 1, idx)


# Test: 2D tensor, dim=1
# This was the failing case before fixing index_op
in1 = torch.randn(3, 3)
idx = torch.tensor([0, 2])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, idx)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.empty
# CHECK: %{{.*}} = linalg.generic
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
