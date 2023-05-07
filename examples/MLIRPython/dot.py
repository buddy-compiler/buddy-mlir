from buddy import compiler
import torch
import torch._dynamo as dynamo

def foo(x, y):
  return torch.dot(x, y)

foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(foo)
in1 = torch.randn(3)
in2 = torch.randn(3)
foo_mlir(in1, in2)
