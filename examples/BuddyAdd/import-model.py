# RUN: %PYTHON %s 2>&1 | FileCheck %s
import os
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse

def foo(x, y):
    return x + y

in1 = torch.randn(10)
in2 = torch.randn(10)

in1 = in1.to(torch.float16)
in2 = in2.to(torch.float16)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
with torch.no_grad():
    graphs = dynamo_compiler.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

