#!/usr/bin/env python3
# ===- import-gemma4.py ---------------------------------------------------
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
# This is the AOT importer for the Gemma-4-E2B model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    StaticCache,
)
from transformers.models.gemma4 import Gemma4ForCausalLM
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.graph.operation import *

parser = argparse.ArgumentParser(description="Gemma-4-E2B Model AOT Importer")
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
    choices=["f32"],
    help="Precision mode for generated MLIR and input data.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_path = os.environ.get("GEMMA4_E2B_MODEL_PATH")
if model_path is None:
    model_path = "google/gemma-4-E2B-it"

MaxTokenLength = 512

# Load the full multimodal model and extract language model weights.
print("Loading full model from:", model_path)
full_model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.float32, low_cpu_mem_usage=True
)
full_sd = full_model.state_dict()

causal_sd = {}
for k, v in full_sd.items():
    if k.startswith("model.language_model."):
        causal_sd[k.replace("model.language_model.", "model.")] = v
    elif k.startswith("lm_head."):
        causal_sd[k] = v

if (
    "lm_head.weight" not in causal_sd
    and "model.embed_tokens.weight" in causal_sd
):
    causal_sd["lm_head.weight"] = causal_sd["model.embed_tokens.weight"]

full_config = AutoConfig.from_pretrained(model_path)
tc = full_config.text_config
model = Gemma4ForCausalLM(tc)
model.load_state_dict(causal_sd, strict=True)
model.eval()
model.config.use_cache = False
del full_model, full_sd, causal_sd

# Pre-initialize StaticCache to avoid Dynamo graph breaks.
print("Pre-initializing StaticCache for prefill...")
cache_prefill = StaticCache(
    config=tc, max_cache_len=MaxTokenLength, batch_size=1
)
with torch.no_grad():
    model(
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        past_key_values=cache_prefill,
        use_cache=True,
        cache_implementation="static",
    )
cache_prefill.reset()

dynamo.reset()

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

with torch.no_grad():
    data_prefill = {
        "input_ids": torch.zeros((1, MaxTokenLength), dtype=torch.int64),
    }

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        past_key_values=cache_prefill,
        cache_implementation="static",
    )

    print("Pre-initializing StaticCache for decode...")
    cache_decode = StaticCache(
        config=tc, max_cache_len=MaxTokenLength, batch_size=1
    )
    model(
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        past_key_values=cache_decode,
        use_cache=True,
        cache_implementation="static",
    )

    # Remove mark_static_address from cumulative_length to prevent dynamo
    # from constant-folding it. Without this, the attention mask in the
    # traced graph only works for the cumlen value at trace time.
    for layer in cache_decode.layers:
        if hasattr(layer.cumulative_length, "_dynamo_static_input_type"):
            delattr(layer.cumulative_length, "_dynamo_static_input_type")
        layer.cumulative_length.fill_(200)

    dynamo.reset()

    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        use_cache=True,
        cache_position=cache_position,
        past_key_values=cache_decode,
    )

assert (
    len(graphs_prefill) == 1
), f"Expected 1 prefill graph, got {len(graphs_prefill)}"
assert (
    len(graphs_decode) == 1
), f"Expected 1 decode graph, got {len(graphs_decode)}"
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]

graphs_prefill[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
graphs_decode[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,
]


graphs_prefill[0].fuse_ops(pattern_list_prefill)
graphs_decode[0].fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()

with open(
    os.path.join(output_dir, "subgraph0_prefill_e2b.mlir"), "w"
) as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(
    os.path.join(output_dir, "forward_prefill_e2b.mlir"), "w"
) as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0_e2b.data"))

with open(
    os.path.join(output_dir, "subgraph0_decode_e2b.mlir"), "w"
) as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(
    os.path.join(output_dir, "forward_decode_e2b.mlir"), "w"
) as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

# Export vocabulary file for the C++ tokenizer.
print("Exporting vocabulary...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
vocab_path = os.path.join(output_dir, "vocab.txt")
with open(vocab_path, "w", encoding="utf-8") as vf:
    for token, _ in sorted_vocab:
        vf.write(token.replace("\n", "\\n") + "\n")
print(f"Exported {len(sorted_vocab)} tokens to {vocab_path}")

print("All files saved to:", output_dir)

# Post-process: patch the constant-folded cumulative_length in decode subgraph.
# torch._dynamo bakes cumulative_length as a compile-time constant, but it must
# be dynamic for correct attention masking across different cache positions.
import subprocess

patch_script = os.path.join(os.path.dirname(__file__), "patch_decode_mlir.py")
decode_mlir = os.path.join(output_dir, "subgraph0_decode_e2b.mlir")
if os.path.exists(patch_script) and os.path.exists(decode_mlir):
    print("Patching constant-folded cumulative_length in decode subgraph...")
    subprocess.run(["python3", patch_script, decode_mlir], check=True)
    print("Patch applied successfully.")
