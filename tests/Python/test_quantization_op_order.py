# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# Weight-only quantization inserts dequantize ops; re-sorting the graph
# afterwards must keep the original ops in their order, since GraphDriver
# returns the graph outputs in the order of the ops computing them.

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.transform.quantization import (
    weight_only_channel_wise,
)
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


class Model(torch.nn.Module):
    """Like a decoder layer of the decode graph: an output computed from a
    quantized weight, then one computed from an input only."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 16, bias=False)

    def forward(self, x, pos):
        return self.proj(x), pos + 1


dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        Model(), torch.randn(4, 8), torch.tensor([3], dtype=torch.int64)
    )
assert len(graphs) == 1
graph = graphs[0]
weight_only_channel_wise(graph)
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
print(driver.construct_main_graph(True))

# CHECK-LABEL: func.func @forward
# CHECK-SAME: -> (memref<4x16xf32>, memref<1xi64>)
