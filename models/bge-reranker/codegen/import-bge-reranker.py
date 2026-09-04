#!/usr/bin/env python3
# ===- import-bge-reranker.py - BGE-Reranker-v2-M3 AOT importer -----------===//
# Adapted from the original PR import_model.py (XLMRobertaForSequenceClassification
# cross-encoder reranker) to the buddy-codegen single_forward interface:
# `--spec` + `--output-dir`, with subgraph0.mlir / forward.mlir / arg0.data
# written into the output-dir ROOT.
#
# Architecture: XLMRobertaForSequenceClassification
#   - 24 layers, hidden=1024, heads=16, vocab=250002, max_position_embeddings=8194
#   - Input: (input_ids, attention_mask) — single query+document pair
#   - Output: logit score (relevance)
#
# The traced forward ABI (mirrored by BgeRerankerRunner.cpp and
# gen_bge_reranker_manifest.py):
#   forward(weights: memref<params_size x f32>,
#           position_ids: memref<max_position_embeddings x i64>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           attention_mask: memref<1 x max_seq_len x i64>)
#     -> (logits: memref<1 x 1 x f32>)
#
# ===----------------------------------------------------------------------===//
import argparse
import json
import os
import sys
import types

import numpy
import torch

import torch._dynamo

torch._dynamo.config.suppress_errors = True

# The buddy Python frontend imports `tomli` only when loading trace configs,
# which this importer never does; provide a stub so an uninstalled tomli does
# not break the import (same pattern as the validated BGE-M3 importer).
if "tomli" not in sys.modules:
    tomli_stub = types.ModuleType("tomli")

    def _tomli_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "tomli is required only when loading trace configs")

    tomli_stub.load = _tomli_unavailable
    tomli_stub.loads = _tomli_unavailable
    sys.modules["tomli"] = tomli_stub

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForSequenceClassification


class BgeRerankerWrapper(torch.nn.Module):
    """Thin wrapper returning just the classification logits so the traced
    graph has a single tensor output (the relevance score)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


p = argparse.ArgumentParser(description="BGE-Reranker AOT importer")
p.add_argument("--spec", required=True, help="Variant spec JSON")
p.add_argument("--output-dir", required=True, help="Output directory")
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)

model_path = (os.environ.get("BGE_RERANKER_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "BAAI/bge-reranker-v2-m3"))
max_seq_len = int(spec.get("max_seq_len", 512))
os.makedirs(a.output_dir, exist_ok=True)

print(f"[import-bge-reranker] Loading BGE-Reranker-v2-M3 from: {model_path}")
# The model uses flash-attention in some published configurations; force eager
# attention so the trace works without a flash_attn install.
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, torch_dtype=torch.float32, attn_implementation="eager"
).eval()
model.config.use_cache = False
print(f"  model class: {type(model).__name__}, "
      f"hidden={model.config.hidden_size}, "
      f"layers={model.config.num_hidden_layers}, "
      f"heads={model.config.num_attention_heads}, "
      f"params: {sum(pp.numel() for pp in model.parameters()):,}")

wrapped = BgeRerankerWrapper(model).eval()

dc = DynamoCompiler(primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=inductor_decomp,
                    func_name="forward")

dummy_ids = torch.ones((1, max_seq_len), dtype=torch.int64)
dummy_mask = torch.ones((1, max_seq_len), dtype=torch.int64)
print(f"[import-bge-reranker] Dummy inputs: {dummy_ids.shape}")

with torch.no_grad():
    graphs = dc.importer(wrapped, input_ids=dummy_ids,
                         attention_mask=dummy_mask)

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]
params = dc.imported_params[graph]
print(f"[import-bge-reranker] 1 graph, {len(params)} params, "
      f"{sum(p.numel() for p in params):,} elems")

# The forward ABI's first memref is the flattened f32 weights
# (memref<params_size x f32>).  `imported_params` may also capture the
# model's non-persistent `position_ids` buffer as a separate i64 table that
# the graph never reads from the weight buffer (it is fed through the second
# forward argument instead).  Keep only the f32 tensors so arg0.data matches
# the `params_size` in the spec / forward.mlir exactly.
params = [p for p in params if p.dtype == torch.float32]
print(f"[import-bge-reranker] {len(params)} f32 params -> "
      f"{sum(p.numel() for p in params):,} weight elems")

graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()

with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(dr.subgraphs[0]._imported_module, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)

all_param = numpy.concatenate(
    [p.detach().cpu().numpy().reshape([-1]) for p in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))

print(f"[import-bge-reranker] Wrote forward.mlir, subgraph0.mlir, "
      f"arg0.data to {a.output_dir}")
