
import os

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg

# define your own operation here
def foo(x, y):
    return x + y


in1 = torch.randn(10).to(torch.float16)
in2 = torch.randn(10).to(torch.float16)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()

path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)
