# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch.nn as nn
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

# Regression test: in eval mode, BatchNorm2d is decomposed into elementwise
# ops whose per-channel operands have NCHW-style shapes such as [C, 1, 1],
# while the activation is NHWC ([N, H, W, C]) after the Conv2d lowering.
# Importing such a graph used to fail with
#   "Incompatible broadcast shapes [1, 112, 112, 32] and [32, 1, 1]".
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
).eval()
data = torch.randn(1, 3, 224, 224)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: tosa.conv2d
# CHECK: tosa.sub %{{.*}}, %{{.*}} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
# CHECK: tosa.mul
# CHECK: return
