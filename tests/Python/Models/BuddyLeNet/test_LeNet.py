# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from model import LeNet


model = LeNet()
model.eval()
input_data = torch.randn([1, 1, 28, 28])


dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, input_data)


assert len(graphs) == 1, "The model should generate exactly 1 computation graph"
graph = graphs[0]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()


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
# CHECK-SAME: tensor<6x1x5x5xf32>
# CHECK-SAME: tensor<6xf32>
# CHECK-SAME: tensor<16x6x5x5xf32>
# CHECK-SAME: tensor<16xf32>
# CHECK-SAME: tensor<120x256xf32>
# CHECK-SAME: tensor<120xf32>
# CHECK-SAME: tensor<84x120xf32>
# CHECK-SAME: tensor<84xf32>
# CHECK-SAME: tensor<10x84xf32>
# CHECK-SAME: tensor<10xf32>
# CHECK-SAME: -> tensor<1x10xf32>

#  < Key operator order >
# CHECK: tosa.transpose{{.*}}tensor<1x28x28x1xf32>
# CHECK: tosa.transpose{{.*}}tensor<6x5x5x1xf32>
# CHECK: tosa.conv2d{{.*}}tensor<1x24x24x6xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x6x24x24xf32>
# CHECK: tosa.max_pool2d{{.*}}tensor<1x12x12x6xf32>
# CHECK: tosa.conv2d{{.*}}tensor<1x8x8x16xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x16x8x8xf32>
# CHECK: tosa.max_pool2d{{.*}}tensor<1x4x4x16xf32>
# CHECK: tosa.reshape{{.*}}tensor<1x256xf32>
# CHECK: linalg.matmul{{.*}}
# CHECK: tosa.add{{.*}}tensor<1x120xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x120xf32>
# CHECK: linalg.matmul{{.*}}
# CHECK: tosa.add{{.*}}tensor<1x84xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x84xf32>
# CHECK: linalg.matmul{{.*}}
# CHECK: tosa.add{{.*}}tensor<1x10xf32>
# CHECK: return{{.*}}tensor<1x10xf32>
# CHECK: }
# CHECK: }

# CHECK: (Main graph MLIR)
# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK-SAME: memref<44426xf32>
# CHECK-SAME: memref<1x1x28x28xf32>
# CHECK-SAME: -> memref<1x10xf32>

# CHECK: memref.subview{{.*}} to memref<150xf32>
# CHECK: memref.expand_shape{{.*}} into memref<6x1x5x5xf32>
# CHECK: memref.subview{{.*}} to memref<2400xf32
# CHECK: memref.expand_shape{{.*}} into memref<16x6x5x5xf32
# CHECK: memref.subview{{.*}} to memref<30720xf32
# CHECK: memref.expand_shape{{.*}} into memref<120x256xf32
# CHECK: memref.subview{{.*}} to memref<10080xf32
# CHECK: memref.expand_shape{{.*}} into memref<84x120xf32
# CHECK: memref.subview{{.*}} to memref<840xf32
# CHECK: memref.expand_shape{{.*}} into memref<10x84xf32
# CHECK: memref.subview{{.*}} to memref<10xf32
# CHECK: call @subgraph0{{.*}}memref<1x10xf32>
# CHECK: return{{.*}}memref<1x10xf32>
# CHECK: }
# CHECK: }
