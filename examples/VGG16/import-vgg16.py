import os
import numpy
from pathlib import Path
import torch
import torchvision
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

from torchvision import transforms
from PIL import Image

import pdb

model = torchvision.models.vgg16(pretrained=True)

model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

#pdb.set_trace()
data = torch.randn([1, 3, 244, 244])
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

path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)


params = dynamo_compiler.imported_params[graph]
current_path = os.path.dirname(os.path.abspath(__file__))

float32_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
float32_param.tofile(Path(current_path) / "arg0.data")

# i64_param = numpy.concatenate(
#     [params[-1].detach().numpy().reshape([-1])   for param in params]
# )
# i64_param.tofile(Path(current_path) / "arg1.data")