# ===- import-deepseek-r1.py ---------------------------------------------------
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
# This is the test of DeepSeekR1 model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import time
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="DeepSeekR1 Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f32",
    choices=["f32", "f16"],
    help="Precision mode for generated MLIR and input data. Choose from 'f32' or 'f16'.",
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Initialize the model from the specified model path.
if args.precision == "f16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)
        .eval()
        .half()
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torchscript=True
    ).eval()
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = {
        "input_ids": torch.zeros((1, 40), dtype=torch.int64),
        "attention_mask": torch.zeros((1, 40), dtype=torch.int64),
    }
    graphs = dynamo_compiler.importer(
        model,
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
    )

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Save the generated files to the specified output directory.
if args.precision == "f16":
    with open(
        os.path.join(output_dir, "subgraph0-f16.mlir"), "w"
    ) as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward-f16.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0-f16.data"))
else:
    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))
