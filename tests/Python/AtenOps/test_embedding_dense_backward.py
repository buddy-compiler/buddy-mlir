# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(weight, indices):
    """Function to test embedding operation for backward verification."""
    # Embedding lookup
    y = torch.nn.functional.embedding(indices, weight)
    return y


# Test input
num_embeddings = 10
embedding_dim = 4
weight = torch.randn(num_embeddings, embedding_dim, requires_grad=True)
indices = torch.tensor([1, 2, 3, 4])

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, weight, indices)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: func.func @forward
# CHECK: tosa.gather
# CHECK: return
