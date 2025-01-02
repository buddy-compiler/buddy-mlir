# ===- import-whisper.py -------------------------------------------------------
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
# This is an example for whisper model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse

# Parse command-line arguments for output directory.
parser = argparse.ArgumentParser(description="Whisper model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the Whisper model path from environment variables.
model_path = os.environ.get("WHISPER_MODEL_PATH")
if model_path is None:
    model_path = "openai/whisper-base"

# Initialize the model from the specified model path.
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.config.use_cache = False

# Generate placeholder for inputs.
input_features = torch.zeros(size=(1, 80, 3000), dtype=torch.float32)
decoder_input_ids = torch.zeros(size=(1, 448), dtype=torch.long)
inputs = {
    "input_features": input_features,
    "decoder_input_ids": decoder_input_ids,
}

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Save the MLIR files and parameter data to the specified output directory.
with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))
