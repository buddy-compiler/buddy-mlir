from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo


def foo(c, a, b):
  return torch.addmm(c, a, b)


foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
a = torch.randn(3, 2)
b = torch.randn(2, 3)
c = torch.randn(3, 3)
foo_mlir(c, a, b)
