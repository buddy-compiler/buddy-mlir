# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# CallExternalOp nodes that call the same function share one declaration;
# call sites of one function with different signatures are rejected.

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform import replace_matmul_with_onednn
from buddy.compiler.ops import func, tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def chain(x, w1, w2):
    return torch.matmul(torch.matmul(x, w1), w2)


def import_chain(w2_cols):
    """Two matmuls, both replaced by calls to onednn_matmul_f32."""
    dynamo_compiler = DynamoCompiler(
        primary_registry={**tosa.ops_registry, **func.ops_registry},
        aot_autograd_decomposition=inductor_decomp,
        enable_external_calls=True,
    )
    graphs = dynamo_compiler.importer(
        chain,
        torch.randn(4, 8),
        torch.randn(8, 8),
        torch.randn(8, w2_cols),
    )
    assert len(graphs) == 1
    graph = graphs[0]
    graph.fuse_ops([replace_matmul_with_onednn])
    return graph


# Same signature: one declaration, two calls.
graph = import_chain(8)
graph.lower_to_top_level_ir()
module = graph._imported_module
assert module.operation.verify()
print(module)

# CHECK-LABEL: func.func @forward
# CHECK: call @onednn_matmul_f32({{.*}}) : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
# CHECK: call @onednn_matmul_f32({{.*}}) : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
# CHECK: func.func private @onednn_matmul_f32(tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
# CHECK-NOT: func.func private @onednn_matmul_f32

# Different signatures under one name: an error naming both types.
graph = import_chain(6)
try:
    graph.lower_to_top_level_ir()
except ValueError as e:
    print(e)

# CHECK: external function 'onednn_matmul_f32' is called with type (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>, but it is already declared with type (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
