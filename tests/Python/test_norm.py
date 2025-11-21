# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

from buddy.compiler.graph.transform import simply_fuse,apply_classic_fusion


def kernel(x, gamma):
    eps = 1e-5
    x_pow = torch.pow(x, 2)
    var = torch.mean(x_pow, dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    norm = x * inv_std
    out = gamma * norm  
    return out

x = torch.randn(1, 40, 1536)
gamma = torch.randn(1536)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = torch.compile(kernel, backend=dynamo_compiler)
assert torch.allclose(foo_mlir(x, gamma), kernel(x, gamma), equal_nan=True)

graphs = dynamo_compiler._imported_graphs
assert len(graphs) == 1 
graph = graphs[0]

pattern_list = [apply_classic_fusion]
graphs[0].fuse_ops(pattern_list)

graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK-LABEL: func.func @forward
# CHECK: arith.constant {{.*}} : f32
# CHECK: arith.constant {{.*}} : f32
# CHECK: arith.constant {{.*}} : index
# CHECK: arith.constant {{.*}} : index
# CHECK: arith.constant {{.*}} : index
# CHECK: arith.constant {{.*}} : index
# CHECK: arith.constant {{.*}} : f32
# CHECK: bufferization.to_memref
# CHECK: bufferization.to_memref
# CHECK: memref.alloc
# CHECK: scf.for
# CHECK: scf.parallel
# CHECK: memref.load
# CHECK: arith.mulf
# CHECK: scf.reduce
# CHECK: arith.divf
# CHECK: arith.addf
# CHECK: math.rsqrt
# CHECK: scf.for
# CHECK: memref.load
# CHECK: memref.load
# CHECK: arith.mulf
# CHECK: arith.mulf
# CHECK: memref.store
# CHECK: bufferization.to_tensor
# CHECK: return


