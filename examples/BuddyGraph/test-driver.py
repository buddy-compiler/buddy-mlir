import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, b, c):
        return torch.matmul(b, c)

model = TestModule()
a, b = torch.randn((1024, 1024)), torch.randn((1024, 1024))
print(model(a, b))

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp
)

# AOT Mode
# Import PyTorch model to Buddy Graph and MLIR/LLVM IR.
graphs = dynamo_compiler.importer(
    model, a, b
)

assert len(graphs) == 1

for g in graphs:
    g.lower_to_top_level_ir()
    print(g._imported_module)