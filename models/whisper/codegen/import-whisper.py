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
# AOT importer for the Whisper model used by models/whisper (buddy-cli / .rax).
#
# Produces three artifacts in --output-dir:
#   forward.mlir    main graph (calls into subgraph0)
#   subgraph0.mlir  lowered top-level IR for the compute subgraph
#   arg0.data       concatenated model parameters (f32)
#
# The model is resolved in this order:
#   1. $WHISPER_MODEL_PATH        (local HF snapshot, set by CMake)
#   2. --spec base.json hf_model_path
#   3. "openai/whisper-base"      (downloaded on demand)
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Whisper model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--spec",
    type=str,
    default=None,
    help="Path to the variant spec JSON (for hf_model_path fallback).",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Resolve the Whisper model path.
model_path = os.environ.get("WHISPER_MODEL_PATH")
if not model_path and args.spec:
    with open(args.spec) as f:
        model_path = json.load(f).get("hf_model_path")
if not model_path:
    model_path = "openai/whisper-base"

print(f"[import-whisper] Loading model from: {model_path}")

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

# ── Work around a transformers tracing bug ───────────────────────────────────
# transformers' masking_utils.find_packed_sequence_indices() keeps a
# "packed-sequence" attention mask whenever it runs under tracing (its
# `if not is_tracing(...): return None` single-sequence early-out is skipped).
# Whisper decoding is always a single sequence, so this spurious mask gets baked
# into the exported graph and collapses decoder self-attention to the diagonal
# (every position attends only to itself) at run time, producing empty output.
# Force the single-sequence behaviour (return None) before tracing.
try:
    import transformers.masking_utils as _mu

    _mu.find_packed_sequence_indices = lambda *a, **k: None
except (
    Exception
) as _e:  # pragma: no cover - older transformers without this util
    print(f"[import-whisper] packed-sequence patch skipped: {_e}")

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
print(
    f"[import-whisper] Wrote forward.mlir, subgraph0.mlir, arg0.data → {output_dir}"
)
