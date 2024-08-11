import os
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def squared_sum(x, y):
    t1 = (x * x).to(torch.float32)
    t2 = (y * y).to(torch.float32)
    return t1 + t2

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
in1 = torch.randn(5, dtype=torch.bfloat16)
in2 = torch.randn(5, dtype=torch.float16)
graphs = dynamo_compiler.importer(squared_sum, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "f16test.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)