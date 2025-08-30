# ===- import-bert.py ----------------------------------------------------------
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
# This is the test of BERT model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertForSequenceClassification, BertTokenizer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="BERT model AOT importer")
parser.add_argument(
    "--output-dir", 
    type=str, 
    default="./", 
    help="Directory to save output files"
)
args = parser.parse_args()

# Ensure output directory exists
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

model = BertForSequenceClassification.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
model.eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

tokenizer = BertTokenizer.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
inputs = {
    "input_ids": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
    "token_type_ids": torch.tensor([[0 for _ in range(5)]], dtype=torch.int64),
    "attention_mask": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
}
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, **inputs)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Write the MLIR module and forward graph to the specified output directory
with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

params = dynamo_compiler.imported_params[graph]

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params[:-1]]
)
float32_param.tofile(Path(output_dir) / "arg0.data")

int64_param = params[-1].detach().numpy().reshape([-1])
int64_param.tofile(Path(output_dir) / "arg1.data")
