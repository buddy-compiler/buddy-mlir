# RUN: %PYTHON %s 2>&1 | FileCheck %s
import torch
import os

from buddy.compiler.graph import GraphDriver
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.operation import *


# Define the target function or model.
def foo(x, y):
    return x * y + x


# Define the input data.
float32_in1 = torch.randn(10).to(torch.float32)
float32_in2 = torch.randn(10).to(torch.float32)

dynamo_compiler = DynamoCompiler()
graphs = dynamo_compiler.importer(foo, *(float32_in1, float32_in2))
graph = graphs[0]
graphs[0].lower_to_top_level_ir()
params = dynamo_compiler.imported_params[graph]

#Divide the subgraphs
group = [graph._body[2]]
subgraph_name = "subgraph0"
graph.group_map_device[subgraph_name] = DeviceType.CPU
graph.op_groups[subgraph_name] = group

new_group = [graph._body[3]]
subgraph_name = "subgraph1"
graph.group_map_device[subgraph_name] = DeviceType.CPU
graph.op_groups[subgraph_name] = new_group

path_prefix = os.path.dirname(os.path.abspath(__file__))
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
driver.subgraphs[1].lower_to_top_level_ir()
print(driver.construct_main_graph(True))
# CHECK: module {
# CHECK-NEXT:   func.func private @subgraph0(memref<10xf32, strided<[1], offset: ?>>, memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32>
# CHECK-NEXT:   func.func private @subgraph1(memref<10xf32, strided<[1], offset: ?>>, memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32>
# CHECK-NEXT:   func.func @forward(%arg0: memref<10xf32>, %arg1: memref<10xf32>) -> memref<10xf32> {
# CHECK-NEXT:     %cast = memref.cast %arg0 : memref<10xf32> to memref<10xf32, strided<[1], offset: ?>>
# CHECK-NEXT:     %cast_0 = memref.cast %arg1 : memref<10xf32> to memref<10xf32, strided<[1], offset: ?>>
# CHECK-NEXT:     %0 = call @subgraph0(%cast, %cast_0) : (memref<10xf32, strided<[1], offset: ?>>, memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32>
# CHECK-NEXT:     %cast_1 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[1], offset: ?>>
# CHECK-NEXT:     %cast_2 = memref.cast %arg0 : memref<10xf32> to memref<10xf32, strided<[1], offset: ?>>
# CHECK-NEXT:     %1 = call @subgraph1(%cast_1, %cast_2) : (memref<10xf32, strided<[1], offset: ?>>, memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32>
# CHECK-NEXT:     return %1 : memref<10xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: } 

