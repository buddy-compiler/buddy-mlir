# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# CallExternalOp nodes that call the same function share one declaration;
# call sites of one function with different signatures are rejected. With
# written_args, the declaration says which arguments the function writes
# (bufferization.access), and one-shot bufferization copies none of the
# arguments it only reads.

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.operation import CallExternalOp
from buddy.compiler.graph.transform import replace_matmul_with_onednn
from buddy.compiler.ops import func, tosa
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager
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


def calls(graph):
    return [n for n in graph.body if isinstance(n, CallExternalOp)]


def copies(module):
    """memref.copy ops after one-shot bufferization of `module`."""
    with module.context:
        m = ir.Module.parse(str(module))
        PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries"
            " allow-unknown-ops})"
        ).run(m.operation)
        return str(m).count("memref.copy")


# Unknown (the default): every argument may be written, x is copied before
# each call.
graph = import_chain(8)
graph.lower_to_top_level_ir()
print("unknown: copies", copies(graph._imported_module))
# CHECK: unknown: copies 2

# Read only.
graph = import_chain(8)
for n in calls(graph):
    n.written_args = []
graph.lower_to_top_level_ir()
module = graph._imported_module
print(
    "read only:",
    [line.strip() for line in str(module).splitlines() if "private" in line][0],
)
print("read only: copies", copies(module))
# CHECK: read only: func.func private @onednn_matmul_f32(tensor<4x8xf32> {bufferization.access = "read"}, tensor<8x8xf32> {bufferization.access = "read"}) -> tensor<4x8xf32>
# CHECK-NEXT: read only: copies 0

# The call sites of one function must agree.
graph = import_chain(8)
calls(graph)[0].written_args = []
try:
    graph.lower_to_top_level_ir()
except ValueError as e:
    print(e)
# CHECK: external function 'onednn_matmul_f32' is called with written_args None, but it is already declared with written_args []
