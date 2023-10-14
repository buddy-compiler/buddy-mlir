import torch
import torch._dynamo as dynamo
from buddy.compiler import BuddyDynamoCompiler
from torch._inductor.decomposition import decompositions as inductor_decomp


def foo(x, y):
    return x + y


dynamo_compiler = BuddyDynamoCompiler(
    func_name="forward", aot_autograd_decomposition=inductor_decomp
)
foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
float32_in1 = torch.randn(10).to(torch.float32)
float32_in2 = torch.randn(10).to(torch.float32)
foo_mlir(float32_in1, float32_in2)

int32_in1 = torch.randint(0, 10, (10,)).to(torch.int32)
int32_in2 = torch.randint(0, 10, (10,)).to(torch.int32)
foo_mlir(int32_in1, int32_in2)
