from buddy import compiler
import torch
import torch._dynamo as dynamo

def softmax(x):
  return torch.softmax(x, 0)

# foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(foo)
foo_mlir = dynamo.optimize(compiler.DynamoCompiler)(softmax)
in1 = torch.randn(2, 3)
print(foo_mlir(in1)==torch.softmax(in1, 0))
