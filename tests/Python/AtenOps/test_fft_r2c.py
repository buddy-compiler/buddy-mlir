# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x):
    # aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided)
    return torch.ops.aten._fft_r2c(x, [0, 1], 0, True)


torch.manual_seed(0)
x = torch.randn(4, 8, dtype=torch.float32)

dynamo_compiler = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=None,
)

graphs = dynamo_compiler.importer(foo, x)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: func.func
# CHECK-SAME: (tensor<4x8xf32>) -> tensor<4x5xcomplex<f32>>
# CHECK: complex.create
