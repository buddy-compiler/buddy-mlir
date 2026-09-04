#!/usr/bin/env python3
# ===- import-smolvlm2.py - SmolVLM2 AOT importer ------------------------===//
#
# Adapted from the original PR `import_model.py` to the buddy-codegen
# `single_forward` interface: `--spec` + `--output-dir`, with the MLIR
# artifacts and `arg0.data` weights written to the output-dir ROOT.
#
# Architecture: SmolVLMModel (base VLM, loaded via AutoModel)
#   - Vision: SmolVLM vision encoder (hidden=768, 12 heads, patch=16)
#   - Text: Llama3 decoder (32 layers, hidden=960, 15 heads, 5 KV heads)
#   - Input: 5D pixel_values (batch, num_images, 3, 512, 512)
#   - Output: last_hidden_state over the text sequence
#
# IMPORTANT -- trace target.  The FULL VLM forward (pixel_values + input_ids +
# attention_mask) does NOT trace as a single Dynamo graph: it graph-breaks into
# ~12 segments (embed, vision tower, connector, decoder, cache bookkeeping), so
# no single `_mlir_ciface_forward` covers the whole VLM.  As the buddy-codegen
# `single_forward` path compiles exactly one forward function, we follow the
# task guidance and trace the LM core / text-only path: `text_model`, the 32-
# layer Llama3 decoder (hidden=960) that produces the text last_hidden_state.
#
# The traced forward ABI is produced by `construct_main_graph(True)`
# (do_param_pack): arg0 is the single flat f32 memref of all params+buffers,
# followed by one memref per runtime input in FX placeholder order
# (input_ids, attention_mask), returning one memref per output
# (last_hidden_state).
#
# ===----------------------------------------------------------------------===//
import argparse
import json
import os

import numpy
import torch

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

p = argparse.ArgumentParser(description="SmolVLM2 AOT importer")
p.add_argument("--spec", required=True, help="Variant spec JSON")
p.add_argument("--output-dir", required=True, help="Root output directory")
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)
model_path = (
    os.environ.get("SMOLVLM2_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "HuggingFaceTB/SmolVLM2-500M-Instruct")
)
os.makedirs(a.output_dir, exist_ok=True)

print("[import-smolvlm2] Loading SmolVLM2-500M-Instruct from:", model_path)
m = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).eval()
m.config.use_cache = False

# Unwrap any decorated `forward`s so Dynamo can trace the real computation.
import types

for mod in m.modules():
    if hasattr(mod.forward, "__wrapped__"):
        mod.forward = types.MethodType(mod.forward.__wrapped__, mod)

print(f"  model class: {type(m).__name__}, "
      f"params: {sum(pp.numel() for pp in m.parameters()):,}")

# Trace the text-only LM core (SmolVLMModel.text_model, a 32-layer Llama3
# decoder).  The full VLM forward is not a single Dynamo graph (see header).
tm = m.text_model
print(f"  trace target: {type(tm).__name__} (text_model), "
      f"params: {sum(pp.numel() for pp in tm.parameters()):,}")

dc = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# Replicate the original PR's text dummy inputs.
dummy_ids = torch.ones((1, 64), dtype=torch.int64)
dummy_mask = torch.ones((1, 64), dtype=torch.int64)

print(f"[import-smolvlm2] input_ids: {tuple(dummy_ids.shape)}, "
      f"attention_mask: {tuple(dummy_mask.shape)}")

with torch.no_grad():
    g = dc.importer(tm, input_ids=dummy_ids, attention_mask=dummy_mask)
print(f"[import-smolvlm2] {len(g)} graph(s)")
graph = g[0]
params = dc.imported_params[graph]
print(f"[import-smolvlm2] first graph: {len(params)} params, "
      f"{sum(pp.numel() for pp in params):,} elems")

# Fusion: simply_fuse ONLY -- no other graph transforms.
graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU
dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()

with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(dr.subgraphs[0]._imported_module, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)

# Flattened weights (arg0.data) must use the graph's param/buffer order so it
# lines up with the packed `memref<params_size x f32>` in forward.mlir.
all_param = numpy.concatenate(
    [pp.detach().cpu().numpy().reshape([-1]) for pp in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))

print(f"[import-smolvlm2] Wrote forward.mlir, subgraph0.mlir, arg0.data "
      f"({all_param.shape[0]} f32) to {a.output_dir}")
