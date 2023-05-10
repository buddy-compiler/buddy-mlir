from buddy import compiler
import torch
import torch._dynamo as dynamo

def foo(x, dim0, dim1):
  return torch.transpose(x, dim0, dim1)

foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(foo)
x = torch.randn(2, 3)
dim0 = 0
dim1 = 1
foo_mlir(x, dim0, dim1)