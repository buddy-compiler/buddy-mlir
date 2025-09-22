# RUN: %PYTHON %s 2>&1 | FileCheck %s

import os
from pathlib import Path
import argparse

import numpy as np
import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion
from buddy.compiler.ops import tosa
from model import LeNet

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="LeNet model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
args = parser.parse_args()

# Ensure output directory exists.
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve the LeNet model path.
model_path = os.path.dirname(os.path.abspath(__file__))

model = LeNet()
model = torch.load(model_path + "/lenet-model.pth", weights_only=False)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry
)

data = torch.randn([1, 1, 28, 28])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()


with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
    print(driver.subgraphs[0]._imported_module)
with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)

float32_param.tofile(output_dir / "arg0.data")

# CHECK-LABEL: func.func @subgraph0(
# CHECK-SAME:     %arg0: tensor<1x1x28x28xf32>
# CHECK-SAME:     %arg1: tensor<6x1x5x5xf32>
# CHECK-SAME:     %arg2: tensor<6xf32>
# CHECK-SAME:     %arg3: tensor<16x6x5x5xf32>
# CHECK-SAME:     %arg4: tensor<16xf32>
# CHECK-SAME:     %arg5: tensor<120x256xf32>
# CHECK-SAME:     %arg6: tensor<120xf32>
# CHECK-SAME:     %arg7: tensor<84x120xf32>
# CHECK-SAME:     %arg8: tensor<84xf32>
# CHECK-SAME:     %arg9: tensor<10x84xf32>
# CHECK-SAME:     %arg10: tensor<10xf32>) -> tensor<1x10xf32>

# CHECK-DAG: %[[PERM0:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
# CHECK-DAG: %[[INPUT_TRANSPOSE:.*]] = tosa.transpose %arg0, %[[PERM0]] : (tensor<1x1x28x28xf32>, tensor<4xi32>) -> tensor<1x28x28x1xf32>
# CHECK-DAG: %[[PERM1:.*]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
# CHECK-DAG: %[[WEIGHT1_TRANSPOSE:.*]] = tosa.transpose %arg1, %[[PERM1]] : (tensor<6x1x5x5xf32>, tensor<4xi32>) -> tensor<6x5x5x1xf32>

# CHECK: %[[CONV1_OUTPUT:.*]] = tosa.conv2d %[[INPUT_TRANSPOSE]], %[[WEIGHT1_TRANSPOSE]], %arg2 {
# CHECK-SAME: acc_type = f32,
# CHECK-SAME: dilation = array<i64: 1, 1>,
# CHECK-SAME: pad = array<i64: 0, 0, 0, 0>,
# CHECK-SAME: stride = array<i64: 1, 1>
# CHECK-SAME: } : (tensor<1x28x28x1xf32>, tensor<6x5x5x1xf32>, tensor<6xf32>) -> tensor<1x24x24x6xf32>

# CHECK-DAG: %[[ZERO1:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x6x24x24xf32>}> : () -> tensor<1x6x24x24xf32>
# CHECK-DAG: %[[RELU1_INPUT:.*]] = tosa.transpose %[[CONV1_OUTPUT]], {{.*}} : (tensor<1x24x24x6xf32>, tensor<4xi32>) -> tensor<1x6x24x24xf32>
# CHECK: %[[RELU1_OUTPUT:.*]] = tosa.maximum %[[RELU1_INPUT]], %[[ZERO1]] : (tensor<1x6x24x24xf32>, tensor<1x6x24x24xf32>) -> tensor<1x6x24x24xf32>

# CHECK-DAG: %[[POOL1_INPUT:.*]] = tosa.transpose %[[RELU1_OUTPUT]], {{.*}} : (tensor<1x6x24x24xf32>, tensor<4xi32>) -> tensor<1x24x24x6xf32>
# CHECK: %[[POOL1_OUTPUT:.*]] = tosa.max_pool2d %[[POOL1_INPUT]] {
# CHECK-SAME: kernel = array<i64: 2, 2>,
# CHECK-SAME: pad = array<i64: 0, 0, 0, 0>,
# CHECK-SAME: stride = array<i64: 2, 2>
# CHECK-SAME: } : (tensor<1x24x24x6xf32>) -> tensor<1x12x12x6xf32>

# CHECK-DAG: %[[WEIGHT2_TRANSPOSE:.*]] = tosa.transpose %arg3, {{.*}} : (tensor<16x6x5x5xf32>, tensor<4xi32>) -> tensor<16x5x5x6xf32>
# CHECK: %[[CONV2_OUTPUT:.*]] = tosa.conv2d {{.*}}, %[[WEIGHT2_TRANSPOSE]], %arg4 {
# CHECK-SAME: acc_type = f32,
# CHECK-SAME: dilation = array<i64: 1, 1>,
# CHECK-SAME: pad = array<i64: 0, 0, 0, 0>,
# CHECK-SAME: stride = array<i64: 1, 1>
# CHECK-SAME: } : (tensor<1x12x12x6xf32>, tensor<16x5x5x6xf32>, tensor<16xf32>) -> tensor<1x8x8x16xf32>

# CHECK: %[[FLATTEN:.*]] = tosa.reshape {{.*}} {new_shape = array<i64: 1, 256>} : (tensor<1x16x4x4xf32>) -> tensor<1x256xf32>


# CHECK-DAG: %[[FC1_WEIGHT_TRANSPOSE:.*]] = tosa.transpose %arg5, {{.*}} : (tensor<120x256xf32>, tensor<2xi32>) -> tensor<256x120xf32>
# CHECK: %[[FC1_OUTPUT:.*]] = tosa.matmul {{.*}}, {{.*}} : (tensor<1x1x256xf32>, tensor<1x256x120xf32>) -> tensor<1x1x120xf32>

# CHECK: %[[FC3_OUTPUT:.*]] = tosa.matmul {{.*}}, {{.*}} : (tensor<1x1x84xf32>, tensor<1x84x10xf32>) -> tensor<1x1x10xf32>

# CHECK: %[[FINAL_OUTPUT:.*]] = tosa.reshape %[[FC3_OUTPUT]] {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
# CHECK: %[[ADD:.*]] = {{.*}}, %[[FINAL_OUTPUT]] : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
# CHECK: return %[[ADD]] : tensor<1x10xf32>
