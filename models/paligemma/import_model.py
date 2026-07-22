#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
#
# PaliGemma-3B-224 Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: PaliGemmaForConditionalGeneration (VLM)
#   - Vision: SigLIP (27 layers, hidden=1152)
#   - Text: Gemma (18 layers, hidden=2048, heads=8, kv_heads=1)
#   - Projector: Linear(1152→2048)
#
# ===---------------------------------------------------------------------------

import argparse, os
import numpy, torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    simply_fuse, apply_classic_fusion, eliminate_transpose,
    eliminate_matmul_transpose_reshape,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

parser = argparse.ArgumentParser(description="PaliGemma Model AOT Importer")
parser.add_argument("--output-dir", type=str, default="./")
parser.add_argument("--precision", type=str, default="f32", choices=["f32"])
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("[PaliGemma-Import] Loading PaliGemma-3B-224...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-224", dtype=torch.float32
).eval()

import types
for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

print(f"   text_hidden={model.config.text_config.hidden_size}, text_layers={model.config.text_config.num_hidden_layers}")
print(f"   vision_hidden={model.config.vision_config.hidden_size}, vision_layers={model.config.vision_config.num_hidden_layers}")

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# Dummy inputs: 1 image (256 tokens) + text
# PaliGemma expects <image> token (id=257152) at start, followed by text
n_img_tokens = 256  # num_image_tokens from config
seq_len = 280       # 256 img + 24 text
pixel_values = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
input_ids = torch.ones((1, seq_len), dtype=torch.int64)
input_ids[0, :n_img_tokens] = 257152  # <image> token
attention_mask = torch.ones((1, seq_len), dtype=torch.int64)

print(f"[PaliGemma-Import] pixel_values: {pixel_values.shape}, input_ids: {input_ids.shape}")

# ── Monkey-patch get_placeholder_mask for fullgraph tracing ──
# The original method checks image token count vs image features and raises
# ValueError if they don't match. This is data-dependent (numel() on masked
# tensor) and causes Dynamo graph breaks. Replace with a no-op version.
_orig_get_placeholder_mask = model.model.get_placeholder_mask

def _patched_get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
    # Skip the data-dependent assert — our dummy inputs are pre-validated
    if input_ids is None:
        special_image_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
    else:
        special_image_mask = input_ids == self.config.image_token_id
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    return special_image_mask

model.model.get_placeholder_mask = _patched_get_placeholder_mask.__get__(model.model)
print("[PaliGemma-Import] monkey-patched get_placeholder_mask.")

with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model, input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values,
    )

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
print(f"[PaliGemma-Import] 1 graph, {len(params)} params")

graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
graph.fuse_ops([simply_fuse, apply_classic_fusion])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)

with open(os.path.join(layer_dir, "subgraph0.mlir"), "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)
with open(os.path.join(layer_dir, "forward.mlir"), "w") as f:
    print(driver.construct_main_graph(True), file=f)

all_param = numpy.concatenate([p.detach().cpu().numpy().reshape([-1]) for p in params])
all_param.tofile(os.path.join(output_dir, "arg0.data"))
print("[PaliGemma-Import] Done!")
