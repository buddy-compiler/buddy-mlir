#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
#
# SmolVLM2-500M-Instruct Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: SmolVLMForConditionalGeneration (VLM)
#   - Vision: SmolVLM vision encoder (hidden=768, 12 heads, patch=16)
#   - Text: Llama3 decoder (32 layers, hidden=960, 15 heads, 5 KV heads)
#   - Input: 5D pixel_values (batch, num_images, 3, 512, 512)
#
# ===---------------------------------------------------------------------------

import argparse, os, numpy, torch
import torch._dynamo; torch._dynamo.config.suppress_errors = True
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
from transformers import AutoModel

parser = argparse.ArgumentParser(description="SmolVLM2 Model AOT Importer")
parser.add_argument("--output-dir", type=str, default="./")
parser.add_argument("--precision", type=str, default="f32", choices=["f32"])
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("[SmolVLM2-Import] Loading SmolVLM2-500M-Instruct...")
model = AutoModel.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Instruct", dtype=torch.float32
).eval()

import types
for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

print(f"   params: {sum(p.numel() for p in model.parameters()):,}")

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# SmolVLM2 expects 5D pixel_values: (batch, num_images, channels, H, W)
dummy_pix = torch.zeros((1, 1, 3, 512, 512), dtype=torch.float32)
dummy_ids = torch.ones((1, 64), dtype=torch.int64)
dummy_mask = torch.ones((1, 64), dtype=torch.int64)

print(f"[SmolVLM2-Import] pixel_values: {dummy_pix.shape}, input_ids: {dummy_ids.shape}")

with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model, input_ids=dummy_ids, attention_mask=dummy_mask,
        pixel_values=dummy_pix,
    )

print(f"[SmolVLM2-Import] {len(graphs)} graph(s) captured")
graph = graphs[0]
params = dynamo_compiler.imported_params.get(graph, [])
print(f"[SmolVLM2-Import] {len(params)} params in first graph")

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

print("[SmolVLM2-Import] Writing weight data...")
all_param = numpy.concatenate(
    [p.detach().cpu().numpy().reshape([-1]) for p in model.parameters()]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[SmolVLM2-Import] Done!")
