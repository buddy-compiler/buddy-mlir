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
    primary_registry=tosa.ops_registry, verbose=True
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
with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)

float32_param.tofile(output_dir / "arg0.data")


print("==================================================")
print("MLIR results of LeNet model structure conversion:")
print("==================================================")
print("(Subgraph MLIR)")
print(driver.subgraphs[0]._imported_module)
print("\n==================================================")
print("(Main graph MLIR)")
print(driver.construct_main_graph(True))
print("==================================================")

# CHECK: (Subgraph MLIR)
# CHECK: module {
# CHECK-LABEL: func.func @subgraph0
# CHECK-SAME: tensor<1x1x28x28xf32>
# CHECK-SAME: -> tensor<1x10xf32>

# CHECK: %[[INPUT_TRANSPOSE:.*]] = tosa.transpose {{.*}} -> tensor<1x28x28x1xf32>
# CHECK: %[[KERNEL_TRANSPOSE:.*]] = tosa.transpose {{.*}} -> tensor<6x5x5x1xf32>
# CHECK: %[[CONV1:.*]] = tosa.conv2d {{.*}} -> tensor<1x24x24x6xf32>
# CHECK: %[[RELU1:.*]] = tosa.maximum
# CHECK: %[[POOL1:.*]] = tosa.max_pool2d
# CHECK: %[[CONV2:.*]] = tosa.conv2d {{.*}} -> tensor<1x8x8x16xf32>
# CHECK: %[[RELU2:.*]] = tosa.maximum
# CHECK: %[[POOL2:.*]] = tosa.max_pool2d
# CHECK: %[[FLATTEN:.*]] = tosa.reshape {{.*}} -> tensor<1x256xf32>
# CHECK: %[[FC1:.*]] = tosa.matmul {{.*}} -> tensor<1x1x120xf32>
# CHECK: %[[FC2:.*]] = tosa.matmul {{.*}} -> tensor<1x1x84xf32>
# CHECK: %[[OUTPUT_FC:.*]] = tosa.matmul {{.*}} -> tensor<1x1x10xf32>
# CHECK: %[[OUTPUT_BIAS:.*]] = tosa.add {{.*}} -> tensor<1x10xf32>
# CHECK: return %[[OUTPUT_BIAS]]
# CHECK: }
# CHECK: }

