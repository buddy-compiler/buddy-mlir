#!/usr/bin/env python3
# ===- import-e5-mistral.py - E5-Mistral AOT importer ---------------------===//
#
# Adapted from the original PR `import_model.py` to the buddy-codegen
# `single_forward` interface: `--spec` + `--output-dir`, MLIR + weights at the
# output-dir ROOT (no `layer_partitioned/`).
#
# E5-Mistral-7B-Instruct is a Mistral-architecture encoder-only model
# (config.architectures = ["MistralModel"]). It is imported exactly like the
# ColBERTv2 BERT encoder: `AutoModel.from_pretrained` + trace `.forward` with
# fixed `input_ids` / `attention_mask` shapes (batch 1, max_seq_len).
#
# The traced forward ABI (confirmed by the generated `forward.mlir`) is:
#
#   forward(weights: memref<params_size x f32>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           attention_mask: memref<1 x max_seq_len x i64>)
#     -> (last_hidden_state: memref<1 x max_seq_len x hidden_size x f32>)
#
# i.e. a SINGLE result. `_mlir_ciface_forward` therefore takes
#   (MemRef<float,3>* result, MemRef<float,1>* weights,
#    MemRef<int64_t,2>* input_ids, MemRef<int64_t,2>* attention_mask).
#
# The model path is read from the E5_MISTRAL_MODEL_PATH env var (set by
# `buddy_add_model(... LOCAL_MODEL_ENV E5_MISTRAL_MODEL_PATH ...)` at build
# time), falling back to spec["hf_model_path"].
#
# Fusing: use ONLY graph.fuse_ops([simply_fuse]) — no classic fusion or
# transpose eliminations (matches the validated single_forward pattern).
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
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

p = argparse.ArgumentParser(description="E5-Mistral AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)

model_path = (
    os.environ.get("E5_MISTRAL_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "intfloat/e5-mistral-7b-instruct")
)
max_seq_len = int(spec.get("max_seq_len", 128))
os.makedirs(a.output_dir, exist_ok=True)

print("[import-e5-mistral] Loading e5-mistral-7b-instruct from:", model_path)
m = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).eval()
m.config.use_cache = False
print(
    f"  model class: {type(m).__name__}, "
    f"params: {sum(pp.numel() for pp in m.parameters()):,}"
)

dc = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)
dummy = torch.zeros((1, max_seq_len), dtype=torch.int64)
mask = torch.ones((1, max_seq_len), dtype=torch.int64)
with torch.no_grad():
    g = dc.importer(m, input_ids=dummy, attention_mask=mask)
print(f"[import-e5-mistral] {len(g)} graphs")
assert len(g) == 1
graph = g[0]
params = dc.imported_params[graph]
print(
    f"[import-e5-mistral] first graph: {len(params)} params, "
    f"{sum(p.numel() for p in params):,} elems"
)

graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU
dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()
subgraph0_text = str(dr.subgraphs[0]._imported_module)
# The causal+attention mask is lowered into `select(mask, 0.0, -inf)`.
# With a padded attention_mask the all-masked (pad) rows then hit
# `-inf - (-inf) = NaN` in the softmax's max-subtraction, and that NaN
# propagates into the valid positions through later layers. PyTorch instead
# masks with a large FINITE negative (torch.finfo.min), which keeps the
# all-masked rows finite (a uniform softmax). Replace every -inf mask
# constant with -1e30 to reproduce the reference numerics.
subgraph0_text = subgraph0_text.replace(
    "dense<0xFF800000> : tensor<f32>", "dense<-1.00000000e+30> : tensor<f32>")
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(subgraph0_text, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)

# Write the flattened f32 weights sequentially (append mode) so a 7B model
# (~28 GB arg0.data) never needs a second 28 GB in-RAM concatenated copy.
weights_path = os.path.join(a.output_dir, "arg0.data")
total_elems = 0
with open(weights_path, "wb") as f:
    for pp in params:
        arr = pp.detach().cpu().numpy().reshape([-1]).astype(numpy.float32,
                                                             copy=False)
        f.write(arr.tobytes())
        total_elems += arr.size
print(f"[import-e5-mistral] Wrote forward.mlir, subgraph0.mlir, arg0.data "
      f"({total_elems:,} f32 elems) to {a.output_dir}")
