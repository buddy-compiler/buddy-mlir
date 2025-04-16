# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y, keepdim):
    return torch.mean(x, y, keepdim=keepdim)


in1 = torch.ones([13, 13], dtype=torch.float32)
in2 = [-1]
in3 = True
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = torch.compile(foo, backend=dynamo_compiler)
assert torch.allclose(
    foo_mlir(in1, in2, keepdim=in3), foo(in1, in2, keepdim=in3), equal_nan=True
)
graphs = dynamo_compiler._imported_graphs
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.reduce_sum
# CHECK: %{{.*}} = "tosa.const"
# CHECK: %{{.*}} = tosa.reciprocal
# CHECK: %{{.*}} = tosa.reshape
# CHECK: %{{.*}} = tosa.mul
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
