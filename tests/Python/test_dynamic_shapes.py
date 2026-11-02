# RUN: %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=SIZES
#
# Dynamic sizes: a symbolic size of the trace is a `?` of the tensor types,
# never the sentinel of a dynamic size in an attribute.
# SIZES-NOT: 9223372036854775808

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def importer():
    return DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )


def lower(function, *inputs):
    (graph,) = importer().importer(function, *inputs)
    graph.lower_to_top_level_ir()
    print(graph._imported_module)


# Every function takes a token count the trace leaves symbolic: its size is
# a `?` of the tensor types and a tensor.dim of the argument that has it.
x = torch.randn(1, 6, 64)
torch._dynamo.mark_dynamic(x, 1)
w = torch.randn(32, 64)


def linear(x, w):
    return torch.nn.functional.linear(x, w)


lower(linear, x, w)
# CHECK-LABEL: func.func @forward
# CHECK-SAME: tensor<1x?x64xf32>
# CHECK: tosa.reshape {{.*}} -> tensor<?x64xf32>
# CHECK: tensor.dim
# CHECK: linalg.matmul
# CHECK: tosa.reshape {{.*}} -> tensor<1x?x32xf32>
# CHECK: return {{.*}} : tensor<1x?x32xf32>


def softmax(x):
    return torch.softmax(x, -1)


lower(softmax, x)
# CHECK-LABEL: func.func @forward
# CHECK-SAME: tensor<1x?x64xf32>
# CHECK: tosa.reduce_max {{.*}} -> tensor<1x?x1xf32>
# Both operands of x - max share the token count: no broadcast check.
# CHECK: linalg.generic
# CHECK: arith.subf
# CHECK: tosa.exp
# CHECK: tosa.reduce_sum {{.*}} -> tensor<1x?x1xf32>
# CHECK: return {{.*}} : tensor<1x?x64xf32>


def heads(x):
    return x.view(1, x.shape[1], 4, 16).transpose(1, 2)


lower(heads, x)
# CHECK-LABEL: func.func @forward
# CHECK: tosa.const_shape {{.*}}dense<[1, -1, 4, 16]>
# CHECK: tosa.reshape {{.*}} -> tensor<1x?x4x16xf32>
# CHECK: tosa.transpose {{.*}} -> tensor<1x4x?x16xf32>


def halves(x):
    return torch.cat([x[..., 32:], x[..., :32]], -1)


lower(halves, x)
# CHECK-LABEL: func.func @forward
# CHECK: %[[TOKENS:[a-z0-9_]+]] = tensor.dim %arg0
# CHECK: tensor.extract_slice %arg0[0, 0, 32] [1, %[[TOKENS]], 32] [1, 1, 1] : tensor<1x?x64xf32> to tensor<1x?x32xf32>
# CHECK: tensor.extract_slice %arg0[0, 0, 0] [1, %[[TOKENS]], 32] [1, 1, 1] : tensor<1x?x64xf32> to tensor<1x?x32xf32>
# CHECK: tensor.concat dim(2) {{.*}} -> tensor<1x?x64xf32>


def mask(x):
    positions = torch.arange(x.shape[1])
    visible = torch.arange(16)[None, :] <= positions[:, None]
    return torch.where(visible, 0.0, -1.0) + x[0, :, :16]


lower(mask, x)
# CHECK-LABEL: func.func @forward
# CHECK: %[[TOKENS:[a-z0-9_]+]] = tensor.dim %arg0
# CHECK: tensor.generate %[[TOKENS]]
# CHECK: tosa.select {{.*}} -> tensor<?x16xf32>
# CHECK: return {{.*}} : tensor<?x16xf32>


def broadcast(x, bias):
    return bias.expand(1, x.shape[1], 64) + x


lower(broadcast, x, torch.randn(64))
# CHECK-LABEL: func.func @forward
# CHECK: %[[TOKENS:[a-z0-9_]+]] = tensor.dim %arg1
# CHECK: tensor.empty(%[[TOKENS]]) : tensor<1x?x64xf32>
# CHECK: linalg.fill
# CHECK: tosa.add {{.*}} -> tensor<1x?x64xf32>


cache = torch.zeros(1, 4, 16, 16)
start = torch.tensor([3])


def attention(q, cache, start):
    # The new keys written into the cache from `start`, then the scores of
    # every query over the whole cache.
    positions = start + torch.arange(q.shape[2])
    keys = cache.index_copy(2, positions, q)
    return torch.matmul(q, keys.transpose(2, 3)).softmax(-1)


q = torch.randn(1, 4, 6, 16)
torch._dynamo.mark_dynamic(q, 2)
lower(attention, q, cache, start)
# CHECK-LABEL: func.func @forward
# CHECK-SAME: tensor<1x4x?x16xf32>
# CHECK: %[[TOKENS:[a-z0-9_]+]] = tensor.dim %arg0
# CHECK: tensor.generate %[[TOKENS]]
# CHECK: scf.parallel
# CHECK: memref.store
# CHECK: tosa.matmul {{.*}} -> tensor<4x?x16xf32>
# CHECK: tosa.reduce_max {{.*}} -> tensor<1x4x?x1xf32>
# CHECK: return {{.*}} : tensor<1x4x?x16xf32>


def self_scores(q):
    # Scores of every token over every token: two dynamic sizes.
    return torch.matmul(q, q.transpose(2, 3)).softmax(-1)


lower(self_scores, q)
# CHECK-LABEL: func.func @forward
# CHECK: %[[TOKENS:[a-z0-9_]+]] = tensor.dim %arg0
# CHECK: tosa.matmul {{.*}} -> tensor<4x?x?xf32>
# CHECK: %[[SHAPE:.*]] = tensor.from_elements %{{.*}}, %{{.*}}, %[[TOKENS]], %[[TOKENS]] : tensor<4xindex>
# CHECK: tensor.reshape %{{.*}}(%[[SHAPE]]) : (tensor<4x?x?xf32>, tensor<4xindex>) -> tensor<1x4x?x?xf32>
# CHECK: return {{.*}} : tensor<1x4x?x?xf32>


def windowed(x):
    # A causal mask within a window: two masks of the token count joined.
    positions = torch.arange(x.shape[1])
    causal = positions[None, :] <= positions[:, None]
    window = (positions[:, None] - positions[None, :]) < 4
    return torch.where(causal & window, x[0, :, :1], 0.0)


lower(windowed, x)
# CHECK-LABEL: func.func @forward
# CHECK: arith.andi
# CHECK: arith.select
# CHECK: } -> tensor<?x?xf32>


def pairs(q):
    # A [tokens, tokens] mask meets [1, 4, tokens, tokens] scores: the mask
    # gains its leading sizes of 1 through expand_shape.
    scores = torch.matmul(q, q.transpose(2, 3))
    positions = torch.arange(q.shape[2])
    mask = positions[None, :] <= positions[:, None]
    return scores * mask


lower(pairs, q)
# CHECK-LABEL: func.func @forward
# CHECK: tensor.expand_shape {{.*}} into tensor<1x1x?x?xf32>
# CHECK: return {{.*}} : tensor<1x4x?x?xf32>


def masked(x):
    # A comparison with a scalar splats it to the dynamic sizes.
    return torch.where(x == float("-inf"), 0.0, x)


lower(masked, x)
# CHECK-LABEL: func.func @forward
# CHECK: tensor.splat %{{.*}}[%{{.*}}] : tensor<1x?x64xf32>
# CHECK: arith.cmpf oeq


# Two symbols: the sizes of x and y are distinct. An elementwise operation
# whose shared dynamic sizes are one symbol needs no check; one over distinct
# symbols keeps tosa's check at run time (either may be 1).
def twice(x, y):
    return x + x, y + y


a = torch.randn(6, 8)
b = torch.randn(7, 8)
torch._dynamo.mark_dynamic(a, 0)
torch._dynamo.mark_dynamic(b, 0)
(graph,) = importer().importer(twice, a, b)
x, y = (node.name for node in graph.inputs)
mixed = [node for node in graph.body if type(node).__name__ == "AddOp"][1]
mixed._arguments = [x, y]
mixed._parents = [x, y]
graph.lower_to_top_level_ir()
print(graph._imported_module)
# CHECK-LABEL: func.func @forward
# CHECK-SAME: tensor<?x8xf32>, %arg1: tensor<?x8xf32>
# CHECK: linalg.generic
# CHECK: arith.addf
# CHECK: tosa.add %arg0, %arg1 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>


# After the import a decision on a symbolic size raises: it would hold for
# the traced size only.
z = torch.randn(1, 6, 64)
torch._dynamo.mark_dynamic(z, 1)
(graph,) = importer().importer(softmax, z)
tokens = graph.inputs[0].tensor_meta["shape"][1]
try:
    if tokens % 32:
        pass
except Exception as error:
    print(f"DECISION: {error}")
# CHECK: DECISION: Guard attempted while ShapeEnv guards are frozen


# LayerNorm over a dynamic token count: the values its lowering builds share
# the token count, so no operation keeps a broadcast check.
norm = torch.nn.LayerNorm(64)
n = torch.randn(1, 6, 64)
torch._dynamo.mark_dynamic(n, 1)
with torch.no_grad():
    lower(norm, n)
# CHECK-LABEL: func.func @forward
# CHECK-NOT: tosa.sub {{.*}}: (tensor<1x?x64xf32>, tensor<1x?x1xf32>)
# CHECK-NOT: tosa.mul {{.*}}: (tensor<1x?x64xf32>, tensor<1x?x1xf32>
# CHECK: return {{.*}} : tensor<1x?x64xf32>


# Cache updates preserve the dynamic capacity of their destination.
def update_cache(cache, positions, updates):
    return torch.index_put(cache, (positions,), updates)


cache = torch.zeros(32, 16)
torch._dynamo.mark_dynamic(cache, 0)
lower(update_cache, cache, torch.tensor([2, 5]), torch.ones(2, 16))
# CHECK-LABEL: func.func @forward
# CHECK-SAME: tensor<?x16xf32>
# CHECK: bufferization.to_tensor {{.*}} : memref<?x16xf32>
# CHECK: return {{.*}} : tensor<?x16xf32>
