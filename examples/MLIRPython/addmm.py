import torch
import torch._dynamo as dynamo
from buddy.compiler import BuddyDynamoCompiler
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.operators.tosa_operators import (
    operators_registry as tosa_operators_registry,
)
from buddy.operators.math_operators import (
    operators_registry as math_operators_registry,
)


def foo(c, a, b):
    return torch.addmm(c, a, b)


dynamo_compiler = BuddyDynamoCompiler(
    func_name="forward",
    aot_autograd_decomposition=inductor_decomp,
    operators_registry={**tosa_operators_registry, **math_operators_registry},
)

foo_mlir = dynamo.optimize(dynamo_compiler)(foo)

a_float32 = torch.randn(3, 2)
b_float32 = torch.randn(2, 3)
c_float32 = torch.randn(3, 3)
foo_mlir(c_float32, a_float32, b_float32)

a_int32 = torch.randint(10, (3, 2)).to(torch.int32)
b_int32 = torch.randint(10, (2, 3)).to(torch.int32)
c_int32 = torch.randint(10, (3, 3)).to(torch.int32)
foo_mlir(c_int32, a_int32, b_int32)
