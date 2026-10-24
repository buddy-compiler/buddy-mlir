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
from buddy.compiler.graph import (
    GraphDriver,
    TemplatePartitionedGraphDriver,
    build_transformer_partition_plan,
)
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
parser.add_argument(
    "--experimental-template-partitioned",
    action="store_true",
    help="Export template-partitioned MLIR for internal build integration.",
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
graph.fuse_ops(pattern_list)

if args.experimental_template_partitioned:
    mlir_dir = os.path.join(output_dir, "layer_partitioned")
    os.makedirs(mlir_dir, exist_ok=True)
    partition_dir = os.path.join(output_dir, "layer_partitioned")
    os.makedirs(partition_dir, exist_ok=True)

    for filename in os.listdir(mlir_dir):
        if filename.startswith("subgraph0_forward_") and filename.endswith(
            ".mlir"
        ):
            os.remove(os.path.join(mlir_dir, filename))

    plan = build_transformer_partition_plan(graph)
    driver = TemplatePartitionedGraphDriver(graph, plan)
    subgraphs = driver.build_template_subgraphs()
    if len(subgraphs) != len(plan.templates):
        raise ValueError(
            "Whisper template count does not match materialized subgraph count: "
            f"templates={len(plan.templates)}, subgraphs={len(subgraphs)}"
        )

    template_files = []
    for unit, subgraph in zip(plan.templates, subgraphs, strict=True):
        subgraph.lower_to_top_level_ir()
        filename = f"{driver.template_symbol(unit.template_id)}.mlir"
        with open(os.path.join(partition_dir, filename), "w") as module_file:
            print(subgraph._imported_module, file=module_file)
        template_files.append(filename)

    with open(os.path.join(partition_dir, "forward.mlir"), "w") as module_file:
        print(
            # Preserve the existing runtime ABI: (encoder output, decoder logits).
            driver.construct_template_combined_main_graph(
                True, output_remap=[1, 0]
            ),
            file=module_file,
        )

    manifest = {
        "template_materialization": True,
        "graphs": [
            {
                "name": "forward",
                "component": "model",
                "forward": "forward.mlir",
                "templates": template_files,
            }
        ],
    }
    with open(
        os.path.join(partition_dir, "partition_manifest.json"),
        "w",
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
else:
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))
if args.experimental_template_partitioned:
    print(
        "[import-whisper] Wrote layer_partitioned MLIR and "
        f"arg0.data → {output_dir}"
    )
else:
    print(
        "[import-whisper] Wrote forward.mlir, subgraph0.mlir, "
        f"arg0.data → {output_dir}"
    )
