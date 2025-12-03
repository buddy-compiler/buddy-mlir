# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(weight, indices, offsets):
    # mode=0 is sum
    return torch.ops.aten._embedding_bag(
        weight,
        indices,
        offsets,
        scale_grad_by_freq=False,
        mode=0,
        sparse=False,
        per_sample_weights=None,
        include_last_offset=False,
        padding_idx=-1,
    )


# Embedding table: 10 embeddings of dimension 4
weight = torch.randn(10, 4)
# Indices to look up
indices = torch.tensor([0, 1, 2, 3, 4, 5])
# Offsets: bag 0 = indices[0:3], bag 1 = indices[3:6]
offsets = torch.tensor([0, 3])

# Initialize the dynamo compiler without decomposition
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=None,
)

graphs = dynamo_compiler.importer(foo, weight, indices, offsets)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module
# CHECK: func.func
# CHECK: tosa.gather
# CHECK: return
