from buddy.compiler import dynamo_compiler
import torch
import torch._dynamo as dynamo


def foo(x, y):
    return x + y


foo_mlir = dynamo.optimize(dynamo_compiler)(foo)
float32_in1 = torch.randn(10).to(torch.float32)
float32_in2 = torch.randn(10).to(torch.float32)
foo_mlir(float32_in1, float32_in2)

int32_in1 = torch.randint(0, 10, (10,)).to(torch.int32)
int32_in2 = torch.randint(0, 10, (10,)).to(torch.int32)
foo_mlir(int32_in1, int32_in2)
