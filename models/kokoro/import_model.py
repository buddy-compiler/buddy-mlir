#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
#
# Kokoro-82M TTS Model Importer (buddy-mlir Pipeline)
#
# Architecture: KModel — text-to-speech with:
#   - Albert-based phoneme encoder (12 layers, hidden=768, 12 heads)
#   - Duration predictor (LSTM + prosody)
#   - ISTFTNet vocoder (generator + discriminator)
#   - 81.8M params, 178-token phoneme vocabulary
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
from kokoro import KModel
import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser(description="Kokoro-82M TTS Model AOT Importer")
parser.add_argument("--output-dir", type=str, default="./")
parser.add_argument("--precision", type=str, default="f32", choices=["f32"])
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("[Kokoro-Import] Loading Kokoro-82M TTS model...")
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
print(f"   params: {sum(p.numel() for p in model.parameters()):,}")

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

dummy_ids = torch.randint(0, 100, (1, 30), dtype=torch.int64)
dummy_ref = torch.randn(1, 256, dtype=torch.float32)

print(f"[Kokoro-Import] Tracing forward_with_tokens... input_ids={dummy_ids.shape}, ref_s={dummy_ref.shape}")

# The main forward does string→token conversion (untraceable).
# Trace forward_with_tokens which takes tensors directly.
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model.forward_with_tokens, input_ids=dummy_ids, ref_s=dummy_ref, speed=1.0,
    )

graph_count = len(graphs)
print(f"[Kokoro-Import] {graph_count} graph(s) captured")
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
print(f"[Kokoro-Import] {len(params)} params in first graph")

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

# Export weights from all params
all_param = numpy.concatenate([p.detach().cpu().numpy().reshape([-1]) for p in model.parameters()])
all_param.tofile(os.path.join(output_dir, "arg0.data"))
print(f"[Kokoro-Import] Done! {len(list(model.parameters()))} parameter tensors, {graph_count} graph(s)")
