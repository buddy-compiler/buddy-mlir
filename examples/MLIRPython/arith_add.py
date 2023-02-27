from buddy import compiler
import torch
import torch._dynamo as dynamo

def foo(x, y):
  return x + y

foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(foo)
in1 = torch.randn(10)
in2 = torch.randn(10)
foo_mlir(in1, in2)
