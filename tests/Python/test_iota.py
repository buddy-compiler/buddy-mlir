# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


class foo(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, a):
        return torch.arange(a)


model = foo()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
in1 = 40
graphs = dynamo_compiler.importer(model, in1)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK-LABEL: func.func @forward
#       CHECK: %[[C0:.*]] = arith.constant 0 : index
#       CHECK: %[[C1:.*]] = arith.constant 1 : index
#       CHECK: %[[IOTA:.*]] = tensor.generate
#       CHECK: arith.index_cast
#       CHECK: tensor.yield
#       CHECK: return %[[IOTA]]
