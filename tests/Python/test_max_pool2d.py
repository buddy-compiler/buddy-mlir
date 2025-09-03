# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = torch.nn.MaxPool2d((5, 5), 3, (2, 2))

    def forward(self, a):
        return self.pool(a)


model = TestModule()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
)

in1 = torch.randn((1, 3, 640, 480))

model_opt = torch.compile(model, backend=dynamo_compiler)
assert torch.allclose(model_opt(in1), model(in1), equal_nan=True)

graphs = dynamo_compiler._imported_graphs
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK-LABEL: func.func @forward
#       CHECK: %[[const_1:.*]] = "tosa.const"
#       CHECK: %[[transpose_1:.*]] = tosa.transpose
#       CHECK: %[[max_pool2d:.*]] = tosa.max_pool2d
#       CHECK: %[[const_2:.*]] = "tosa.const"
#       CHECK: %[[transpose_2:.*]] = tosa.transpose
#       CHECK: return %[[transpose_2]]s

