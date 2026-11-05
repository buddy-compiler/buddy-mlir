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
# The encoder and the decoder are imported as separate graphs so that one
# transcription runs the encoder once and only the decoder for each token.
#
# Produces five artifacts in --output-dir:
#   forward_encoder.mlir    encoder main graph (calls into subgraph0_encoder)
#   subgraph0_encoder.mlir  lowered top-level IR for the encoder
#   forward_decoder.mlir    decoder main graph (calls into subgraph0_decoder)
#   subgraph0_decoder.mlir  lowered top-level IR for the decoder
#   arg0.data               encoder parameters followed by decoder parameters
#                           (f32); the split point is encoder_params_size in
#                           the variant spec
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
from buddy.compiler.graph.type import DeviceType
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

spec = {}
if args.spec:
    with open(args.spec) as f:
        spec = json.load(f)

# Resolve the Whisper model path.
model_path = os.environ.get("WHISPER_MODEL_PATH") or spec.get("hf_model_path")
if not model_path:
    model_path = "openai/whisper-base"

print(f"[import-whisper] Loading model from: {model_path}")

# Initialize the model from the specified model path.
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.config.use_cache = False


class WhisperEncoderGraph(torch.nn.Module):
    """Log-mel features -> encoder hidden states."""

    def __init__(self, whisper):
        super().__init__()
        self.config = whisper.config
        self.encoder = whisper.model.encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


class WhisperDecoderGraph(torch.nn.Module):
    """Encoder hidden states and decoder tokens -> vocabulary logits."""

    def __init__(self, whisper):
        super().__init__()
        self.config = whisper.config
        self.decoder = whisper.model.decoder
        self.proj_out = whisper.proj_out

    def forward(self, encoder_hidden_states, decoder_input_ids):
        hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        ).last_hidden_state
        return self.proj_out(hidden_states)


# Each graph with the placeholder inputs that fix its shapes, and the input
# shapes WhisperRuntime passes to the compiled function, in order. The importer
# orders graph inputs by first use, not by the order they are passed in.
graph_definitions = [
    (
        "encoder",
        WhisperEncoderGraph(model).eval(),
        (torch.zeros(size=(1, 80, 3000), dtype=torch.float32),),
        [[1, 80, 3000]],
    ),
    (
        "decoder",
        WhisperDecoderGraph(model).eval(),
        (
            torch.zeros(
                size=(
                    1,
                    spec.get("enc_seq", 1500),
                    spec.get("enc_dim", model.config.d_model),
                ),
                dtype=torch.float32,
            ),
            torch.zeros(size=(1, 448), dtype=torch.long),
        ),
        [
            [1, 448],
            [
                1,
                spec.get("enc_seq", 1500),
                spec.get("enc_dim", model.config.d_model),
            ],
        ],
    ),
]

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


def import_graph(name, graph_module, inputs, expected_input_shapes):
    """Import one graph as MLIR-ready buddy graph plus flat parameters."""
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=f"forward_{name}",
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(graph_module, *inputs)
    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    input_shapes = [list(meta.shape) for meta in graph.inputs_shapes]
    if input_shapes != expected_input_shapes:
        raise ValueError(
            f"Whisper {name} graph inputs are {input_shapes}; WhisperRuntime "
            f"passes {expected_input_shapes}"
        )
    graph.fuse_ops([simply_fuse])
    return graph, params


def write_template_partitioned(name, graph, partition_dir):
    """Write one graph's template subgraphs and return its manifest entry."""
    for filename in os.listdir(partition_dir):
        if filename.startswith(f"subgraph0_{name}_") and filename.endswith(
            ".mlir"
        ):
            os.remove(os.path.join(partition_dir, filename))

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

    forward_file = f"forward_{name}.mlir"
    with open(os.path.join(partition_dir, forward_file), "w") as module_file:
        print(
            driver.construct_template_combined_main_graph(True),
            file=module_file,
        )
    return {
        "name": f"forward_{name}",
        "component": "model",
        "forward": forward_file,
        "templates": template_files,
    }


def write_single_subgraph(name, graph):
    """Write one graph as its main graph and one compute subgraph."""
    # Both graphs link into one library, so their subgraph symbols must differ.
    subgraph_name = f"subgraph0_{name}"
    graph.op_groups[subgraph_name] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[subgraph_name] = DeviceType.CPU

    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(
        os.path.join(output_dir, f"{subgraph_name}.mlir"), "w"
    ) as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, f"forward_{name}.mlir"), "w"
    ) as module_file:
        print(driver.construct_main_graph(True), file=module_file)


# Import both graphs. The decoder and its output projection share the embedding
# weights, and neither graph shares parameters with the other.
imported_graphs = [
    (name, *import_graph(name, graph_module, inputs, expected_input_shapes))
    for name, graph_module, inputs, expected_input_shapes in graph_definitions
]

if args.experimental_template_partitioned:
    partition_dir = os.path.join(output_dir, "layer_partitioned")
    os.makedirs(partition_dir, exist_ok=True)

    manifest = {
        "template_materialization": True,
        "graphs": [
            write_template_partitioned(name, graph, partition_dir)
            for name, graph, _ in imported_graphs
        ],
    }
    with open(
        os.path.join(partition_dir, "partition_manifest.json"),
        "w",
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
else:
    for name, graph, _ in imported_graphs:
        write_single_subgraph(name, graph)

graph_params = [
    numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    for _, _, params in imported_graphs
]
encoder_params_size = graph_params[0].size
all_param = numpy.concatenate(graph_params)
for key, actual in (
    ("encoder_params_size", encoder_params_size),
    ("params_size", all_param.size),
):
    if key in spec and spec[key] != actual:
        raise ValueError(
            f"Whisper spec {key}={spec[key]} does not match the "
            f"{actual} parameters imported from {model_path}"
        )
all_param.tofile(os.path.join(output_dir, "arg0.data"))
if args.experimental_template_partitioned:
    print(
        "[import-whisper] Wrote layer_partitioned MLIR and "
        f"arg0.data → {output_dir}"
    )
else:
    print(
        "[import-whisper] Wrote forward_{encoder,decoder}.mlir, "
        f"subgraph0_{{encoder,decoder}}.mlir, arg0.data → {output_dir}"
    )
