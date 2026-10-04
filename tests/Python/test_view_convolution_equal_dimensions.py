# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions


def flatten_convolution(value, weight):
    return torch.nn.functional.conv2d(value, weight).reshape(1, 64)


def reshape_convolution_relu(value, weight):
    return torch.nn.functional.conv2d(value, weight).relu().reshape(1, 2, 8, 4)


for function in (flatten_convolution, reshape_convolution_relu):
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
    )
    graphs = compiler.importer(
        function,
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1),
    )
    assert len(graphs) == 1
    graphs[0].lower_to_top_level_ir()
    print(function.__name__)
    print(graphs[0]._imported_module)

# CHECK-LABEL: flatten_convolution
# CHECK: %[[CONV:.*]] = tosa.conv2d
# CHECK: %[[NCHW:.*]] = tosa.transpose %[[CONV]] {perms = array<i32: 0, 3, 1, 2>}
# CHECK: tosa.reshape %[[NCHW]]
# CHECK: return

# CHECK-LABEL: reshape_convolution_relu
# CHECK: tosa.conv2d
# CHECK: %[[RELU:.*]] = tosa.maximum
# CHECK: %[[NCHW_RELU:.*]] = tosa.transpose %[[RELU]] {perms = array<i32: 0, 3, 1, 2>}
# CHECK: tosa.reshape %[[NCHW_RELU]]
# CHECK: return
