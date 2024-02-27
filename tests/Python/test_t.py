import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
 
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
 
 
def foo(x, dim, start_idx, end_idx):
    return torch.ops.aten.slice(x, dim, start_idx, end_idx)
 
 
# x = torch.randn(3, 5, 2)
x = torch.arange(10).reshape(2, 5)
dim = 1
start_idx = 0
end_idx = 1
 
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
 
foo_mlir = torch.compile(foo, backend=dynamo_compiler)
print("foo_mlir result:", foo_mlir(x, dim, start_idx, end_idx))