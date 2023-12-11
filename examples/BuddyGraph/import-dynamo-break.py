import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
# torch._dynamo.reset()
# torch._logging.set_logs(graph_breaks=True)
class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.Conv2d(3, 255, (5, 5), 3, bias=False)

    def forward(self, b, c):
        if not torch.nn.functional.silu(b)[0][0]:
            return torch.mm(b, c)
        else:
            return torch.add(b, c)

model = TestModule()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions
)
gm, params = dynamo_compiler.importer(
    model, torch.randn((1024, 1024)), torch.randn((1024, 1024))
)
print(gm)