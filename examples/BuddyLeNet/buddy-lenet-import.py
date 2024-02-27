# ===- buddy-lenet-import.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os

import numpy
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from model import LeNet

# Retrieve the LeNet model path from environment variables.
model_path = os.environ.get("LENET_EXAMPLE_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LENET_MODEL_PATH' is not set or is invalid."
    )

model = LeNet()
model = torch.load(model_path + "/lenet-model.pth")
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

data = torch.randn([1, 1, 28, 28])
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.lower_to_top_level_ir(do_params_pack=True)
path_prefix = os.path.dirname(os.path.abspath(__file__))
# Write the MLIR module to the file.
with open(os.path.join(path_prefix, "lenet.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)

# Concatenate all parameters into a single numpy array and write to a file.
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(path_prefix, "arg0.data"))
