# ===- import_whisper.py -------------------------------------------------------
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
#   arg0.data       concatenated model parameters (f32 or f16 from the spec)
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
import shutil

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
parser.add_argument("--split-encoder-decoder", action="store_true")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Resolve the Whisper model path and selected precision.
spec = {}
if args.spec:
    with open(args.spec) as f:
        spec = json.load(f)
precision = spec.get("precision", "f32")
if precision not in ("f32", "f16"):
    raise ValueError(f"Unsupported Whisper precision: {precision}")
max_token_len = spec.get("max_token_len", 448)
if not isinstance(max_token_len, int) or not 2 <= max_token_len <= 448:
    raise ValueError(
        "Whisper max_token_len must be an integer between 2 and 448"
    )
model_path = os.environ.get("WHISPER_MODEL_PATH") or spec.get("hf_model_path")
if not model_path:
    model_path = "openai/whisper-base"

print(f"[import_whisper] Loading model from: {model_path}")

# Initialize the model from the specified model path.
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.config.use_cache = False

# Dynamo tracing otherwise inserts a packed-sequence mask for this
# single-sequence decoder and changes the exported attention graph.
try:
    import transformers.masking_utils as _mu

    _mu.find_packed_sequence_indices = lambda *a, **k: None
except (
    Exception
) as _e:  # pragma: no cover - older transformers without this util
    print(f"[import_whisper] packed-sequence patch skipped: {_e}")

if args.split_encoder_decoder:
    if precision != "f16":
        raise ValueError("Split Whisper encoder/decoder requires FP16")
    from transformers.cache_utils import EncoderDecoderCache, StaticCache
    from transformers.modeling_outputs import BaseModelOutput

    kv_cache = bool(spec.get("kv_cache", False))
    if kv_cache and not spec.get("last_logits_only", False):
        raise ValueError("Whisper KV cache requires last_logits_only")

    if precision == "f16":
        model = model.half()

    class EncoderCore(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_features):
            if precision == "f16":
                input_features = input_features.half()
            hidden = self.inner.model.encoder(input_features).last_hidden_state
            return hidden.float()

    class DecoderCore(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, encoder_hidden_state, decoder_input_ids):
            if precision == "f16":
                encoder_hidden_state = encoder_hidden_state.half()
            result = self.inner(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_hidden_state
                ),
                decoder_input_ids=decoder_input_ids,
            )
            return result.logits.float()

    class KvDecoderCore(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(
            self, encoder_hidden_state, decoder_input_ids, past_key_values
        ):
            result = self.inner(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_hidden_state.half()
                ),
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return result.logits.float()

    def make_kv_cache():
        self_cache = StaticCache(model.config, max_token_len)
        cross_cache = StaticCache(model.config, 1500)
        for cache in (self_cache, cross_cache):
            cache.early_initialization(
                1, 8, 64, torch.float16, torch.device("cpu")
            )
        return EncoderDecoderCache(self_cache, cross_cache)

    def export_stage(stage, module, inputs):
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=inductor_decomp,
        )
        with torch.no_grad():
            graphs = compiler.importer(module, **inputs)
        assert len(graphs) == 1
        graph = graphs[0]
        params = compiler.imported_params[graph]
        graph.fuse_ops([simply_fuse])
        driver = GraphDriver(graph)
        driver.subgraphs[0].lower_to_top_level_ir()
        compute = str(driver.subgraphs[0]._imported_module).replace(
            "@subgraph0", f"@subgraph_{stage}"
        )
        forward = (
            str(driver.construct_main_graph(True))
            .replace("@subgraph0", f"@subgraph_{stage}")
            .replace("@forward(", f"@forward_{stage}(")
        )
        with open(os.path.join(output_dir, f"{stage}_subgraph.mlir"), "w") as f:
            f.write(compute)
        with open(os.path.join(output_dir, f"{stage}_forward.mlir"), "w") as f:
            f.write(forward)
        numpy.concatenate(
            [param.detach().numpy().reshape([-1]) for param in params]
        ).tofile(os.path.join(output_dir, f"arg_{stage}.data"))

    export_stage(
        "encoder",
        EncoderCore(model),
        {"input_features": torch.zeros((1, 80, 3000), dtype=torch.float32)},
    )
    if kv_cache:
        kv_core = KvDecoderCore(model)
        kv_inputs = {
            "encoder_hidden_state": torch.zeros(
                (1, 1500, 512), dtype=torch.float32
            ),
            "decoder_input_ids": torch.full((1, 1), 50258, dtype=torch.long),
        }
        init_cache = make_kv_cache()
        export_stage(
            "kv_init", kv_core, {**kv_inputs, "past_key_values": init_cache}
        )
        step_cache = make_kv_cache()
        with torch.no_grad():
            kv_core(
                kv_inputs["encoder_hidden_state"],
                kv_inputs["decoder_input_ids"],
                step_cache,
            )
        export_stage(
            "kv_step", kv_core, {**kv_inputs, "past_key_values": step_cache}
        )
        stages = ("encoder", "kv_init", "kv_step")
    else:
        decoder_inputs = {
            "encoder_hidden_state": torch.zeros(
                (1, 1500, 512), dtype=torch.float32
            ),
            "decoder_input_ids": torch.zeros(
                (1, max_token_len), dtype=torch.long
            ),
        }
        export_stage("decoder", DecoderCore(model), decoder_inputs)
        stages = ("encoder", "decoder")
    with open(os.path.join(output_dir, "arg0.data"), "wb") as combined:
        for stage in stages:
            with open(
                os.path.join(output_dir, f"arg_{stage}.data"), "rb"
            ) as part:
                shutil.copyfileobj(part, combined)
    encoder_bytes = os.path.getsize(
        os.path.join(output_dir, "arg_encoder.data")
    )
    if encoder_bytes != spec["encoder_params_size"] * 2:
        raise ValueError("Split Whisper encoder parameters do not match spec")
    if kv_cache:
        init_bytes = os.path.getsize(
            os.path.join(output_dir, "arg_kv_init.data")
        )
        if init_bytes != spec["decoder_init_params_size"] * 2:
            raise ValueError("Whisper KV init parameters do not match spec")
    expected_bytes = spec["params_size"] * (2 if precision == "f16" else 4)
    if os.path.getsize(os.path.join(output_dir, "arg0.data")) != expected_bytes:
        raise ValueError("Split Whisper parameters do not match params_size")
    print("[import_whisper] Wrote split encoder and decoder graphs")
    raise SystemExit(0)

if precision == "f16":
    # Keep the runtime's audio and logits descriptors in f32. Only the model
    # weights and internal activations use f16, so the existing ASR API works
    # for both precisions.
    class WhisperF16Core(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner.half()

        def forward(self, input_features, decoder_input_ids):
            result = self.inner(
                input_features=input_features.half(),
                decoder_input_ids=decoder_input_ids,
            )
            return (
                result.encoder_last_hidden_state.float(),
                result.logits.float(),
            )

    model = WhisperF16Core(model)

# Generate placeholder for inputs.
input_features = torch.zeros(size=(1, 80, 3000), dtype=torch.float32)
decoder_input_ids = torch.zeros(size=(1, max_token_len), dtype=torch.long)
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
    print(f"[import_whisper] packed-sequence patch skipped: {_e}")
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
            driver.construct_template_combined_main_graph(True),
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
        "[import_whisper] Wrote layer_partitioned MLIR and "
        f"arg0.data → {output_dir}"
    )
else:
    print(
        "[import_whisper] Wrote forward.mlir, subgraph0.mlir, "
        f"arg0.data → {output_dir}"
    )
