# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torchvision.models as models
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._decomp import remove_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa


model = models.resnet18(weights=None)
model = model.eval()
data = torch.randn([1, 3, 224, 224])


for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
        if hasattr(layer, "num_batches_tracked"):
            del layer.num_batches_tracked

DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.max_pool2d_with_indices.default,
]

remove_decompositions(inductor_decomp, DEFAULT_DECOMPOSITIONS)


dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)


with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()


print("==================================================")
print("MLIR results of ResNet18 model structure conversion:")
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
# CHECK: tosa.transpose{{.*}}tensor<1x224x224x3xf32>
# CHECK: tosa.conv2d{{.*}}tensor<1x112x112x64xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x64x112x112xf32>
# CHECK: tosa.max_pool2d{{.*}}tensor<1x56x56x64xf32>
# CHECK: tosa.conv2d{{.*}}tensor<1x56x56x64xf32>
# CHECK: tosa.maximum{{.*}}tensor<1x64x56x56xf32>
# CHECK: tosa.add{{.*}}tensor<64xf32>
# CHECK: tosa.reshape{{.*}}tensor<64x1xf32>
# CHECK: tosa.matmul{{.*}}tensor<1x1x1000xf32>
# CHECK: tosa.add{{.*}}tensor<1x1000xf32> 
# CHECK: return{{.*}}tensor<1x1000xf32>
# CHECK: }
# CHECK: }

