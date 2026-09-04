#!/usr/bin/env python3
# ===- import-molformer.py - MoLFormer AOT importer -----------------------===//
# Adapted from the original models/molformer/import_model.py PR to the
# buddy-codegen single_forward interface: `--spec` + `--output-dir`, with
# MLIR + weights written at the output-dir root (no layer_partitioned/).
#
# Model: ibm/MoLFormer-XL-both-10pct, a chemistry Transformer encoder
# (bert-like MolformerModel).  It is loaded via the snapshot's remote code
# (`trust_remote_code=True`; configuration_molformer.py + modeling_molformer.py
# ship in the HF snapshot, so no extra pip package is needed at build time).
#
# transformers >= 4.40 removed `masking_utils.create_bidirectional_mask`, which
# modeling_molformer.py imports at module scope.  We install a universal
# bidirectional mask *before* AutoModel.from_pretrained so the remote module
# import succeeds (this mirrors the original import_model.py).
# ===----------------------------------------------------------------------===//

import argparse
import os
import json
import sys
import numpy
import torch

# --- Install a universal bidirectional attention mask for MoLFormer. -------
# modeling_molformer.py does `from transformers.masking_utils import
# create_bidirectional_mask`; that symbol is absent in modern transformers, so
# we patch the attribute onto the module first.  The patched function derives
# the [batch, 1, 1, seq_len] all-ones mask from the actual traced tensors so
# that 128-token sequences get a correctly-sized full-attention mask.
import transformers.masking_utils as mask_utils


def universal_bidirectional_mask(*args, **kwargs):
    if args:
        first_arg = args[0]
        shape = first_arg.shape if isinstance(first_arg, torch.Tensor) else first_arg
    elif kwargs.get("inputs_embeds") is not None:
        shape = kwargs["inputs_embeds"].shape
    elif kwargs.get("attention_mask") is not None:
        shape = kwargs["attention_mask"].shape
    elif kwargs.get("input_shape") is not None:
        shape = kwargs["input_shape"]
    else:
        shape = torch.zeros(1, 128).shape
    batch_size, seq_len = shape[0], shape[1]
    return torch.ones((batch_size, 1, 1, seq_len), device=kwargs.get("device", "cpu"))


mask_utils.create_bidirectional_mask = universal_bidirectional_mask

import torch._dynamo

torch._dynamo.config.suppress_errors = True
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

p = argparse.ArgumentParser(description="MoLFormer AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()
with open(a.spec) as f:
    spec = json.load(f)
model_path = (os.environ.get("MOLFORMER_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "ibm/MoLFormer-XL-both-10pct"))
os.makedirs(a.output_dir, exist_ok=True)

# --- Deterministic random features + drop the attention shape check ----------
# MoLFormer's linear attention uses Generalized Random Fourier Features.  With
# `deterministic_eval` False (the checkpoint default) every forward re-rolls
# the projection via torch.randn/linalg.qr, which torch._dynamo cannot trace
# into a single AOT graph (linalg_qr has no TOSA mapping).  Flipping the flag
# *after* construction is not enough: MolformerFeatureMap reads
# config.deterministic_eval in __init__ into self.deterministic, so we force
# each feature map into deterministic mode explicitly.  The random projection
# weights are a persistent buffer, so they are part of the exported weights.
#
# MolformerSelfAttention.forward additionally performs
# `torch.equal(attention_mask, per_query_extended.expand_as(...))` and raises
# ValueError on arbitrary 3D masks.  torch.equal yields a scalar consumed by
# `if`, which is a graph break inside every attention layer.  We replace the
# forward with an equivalent one that skips the (always-true, given our
# universal all-ones bidirectional mask) equality check.
def _make_deterministic(model):
    model.config.deterministic_eval = True
    for layer in model.encoder.layer:
        layer.attention.self.feature_map.deterministic = True


def _patch_attention(model):
    mod = sys.modules[type(model.encoder.layer[0].attention.self).__module__]

    def patched_self_attn(self, hidden_states, attention_mask=None,
                          position_ids=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        kv_seq_len = key_layer.shape[-2]
        cos, sin = self.rotary_embeddings(value_layer, seq_len=kv_seq_len)
        query_layer, key_layer = mod.apply_rotary_pos_emb(
            query_layer, key_layer, cos, sin, position_ids)
        query_layer, key_layer = self.feature_map(query_layer, key_layer)
        if attention_mask is not None:
            attention_mask = (attention_mask == 0).to(attention_mask.dtype)
            # separate original mask from causal mask (always all-ones here)
            per_query_attn = attention_mask[:, 0, -1]
            key_layer = key_layer * per_query_attn[:, None, -kv_seq_len:, None]
        key_value = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        norm = torch.matmul(
            query_layer, key_layer.sum(dim=-2).unsqueeze(-1)).clamp(min=self.eps)
        context_layer = torch.matmul(query_layer, key_value) / norm
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return (context_layer,)

    for layer in model.encoder.layer:
        bound = patched_self_attn.__get__(
            layer.attention.self, type(layer.attention.self))
        layer.attention.self.forward = bound


print("[import-molformer] Loading ibm/MoLFormer-XL-both-10pct from:", model_path)
m = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                              dtype=torch.float32).eval()
m.config.use_cache = False
_make_deterministic(m)
_patch_attention(m)
print(f"  model class: {type(m).__name__}, params: {sum(pp.numel() for pp in m.parameters()):,}")

dc = DynamoCompiler(primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=inductor_decomp, func_name="forward")
dummy = torch.zeros((1, 128), dtype=torch.int64)
mask = torch.ones((1, 128), dtype=torch.int64)
with torch.no_grad():
    g = dc.importer(m, input_ids=dummy, attention_mask=mask)
print(f"[import-molformer] {len(g)} graphs")
graph = g[0]
params = dc.imported_params[graph]
print(f"[import-molformer] first graph: {len(params)} tensors, "
      f"{sum(p.numel() for p in params):,} elems")

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
print(f"[import-molformer] Wrote forward.mlir, subgraph0.mlir, arg0.data to {a.output_dir}")
