import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

class MaxPool(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool = torch.nn.MaxPool2d((4, 4))

    def forward(self, a):
        return self.maxpool(a)

model = MaxPool()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions
)
gm, params = dynamo_compiler.importer(
    model, torch.randn((1, 3, 640, 480), device='cpu')
)
print(gm)
# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = "tosa.max_pool2d"
# CHECK: "func.return"(%{{.*}})
# CHECK: }
# CHECK: }