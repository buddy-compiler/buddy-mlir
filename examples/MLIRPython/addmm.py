from buddy.compiler import dynamo_compiler
import torch
import torch._dynamo as dynamo


def foo(c, a, b):
    return torch.addmm(c, a, b)


foo_mlir = dynamo.optimize(dynamo_compiler)(foo)

a_float32 = torch.randn(3, 2)
b_float32 = torch.randn(2, 3)
c_float32 = torch.randn(3, 3)
foo_mlir(c_float32, a_float32, b_float32)

a_int32 = torch.randint(10, (3, 2)).to(torch.int32)
b_int32 = torch.randint(10, (2, 3)).to(torch.int32)
c_int32 = torch.randint(10, (3, 3)).to(torch.int32)
foo_mlir(c_int32, a_int32, b_int32)
