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
# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.transpose
# CHECK: %{{.*}} = tosa.max_pool2d
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.transpose
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
