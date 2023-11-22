import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

class TransposeConvolution(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.ConvTranspose2d(3, 255, (5, 5), 3, bias=False)

    def forward(self, a):
        return self.conv(a)

model = TransposeConvolution()
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
# CHECK: %{{.*}} = tensor.extract_slice
# CHECK: %{{.*}} = tensor.expand_shape
# CHECK: %{{.*}} = "tosa.const"()
# CHECK: %{{.*}} = "tosa.transpose_conv2d"
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
