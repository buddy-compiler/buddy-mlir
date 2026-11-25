#!/usr/bin/env python3
# ===- import_embeddinggemma.py - embeddinggemma AOT importer ------------===//
# EmbeddingGemma AOT importer for Buddy-MLIR.
# Implements the SentenceTransformer pipeline using the buddy-codegen
# single_forward interface.: `--spec` + `--output-dir`, with
# subgraph0.mlir / forward.mlir / arg0.data written at the output-dir ROOT.
#
# Architecture (unchanged from the original PR):
#   Gemma3TextModel -> Mean Pooling -> Dense(768->3072) ->
#   Dense(3072->768) -> L2 Normalize -> 768-dim embedding
#
# Model source resolution (in order):
#   $EMBEDDINGGEMMA_MODEL_PATH  (set by buddy_add_model LOCAL_MODEL_ENV)
#   $BUDDY_LOCAL_MODEL_PATH     (generic fallback)
#   spec["hf_model_path"]       (e.g. google/embeddinggemma-300m)
#
# ===----------------------------------------------------------------------===//
import argparse
import json
import os
import sys

import numpy
import torch
import torch.nn as nn
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import eliminate_transpose, simply_fuse
from buddy.compiler.ops import tosa
from buddy_mlir import ir
from sentence_transformers import SentenceTransformer
from torch._inductor.decomposition import decompositions as inductor_decomp


# ==============================================================================
# The wrapper executes the SentenceTransformer pipeline as a single
# forward graph. It takes (input_ids, attention_mask) and returns the
# normalized sentence embedding.
# ==============================================================================
class EmbeddingGemmaWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        st_model = SentenceTransformer(model_path, device="cpu")
        st_model.eval()

        # Extract sub-modules
        self.transformer = st_model._first_module()
        self.dense1 = st_model._modules["2"]
        self.dense2 = st_model._modules["3"]
        self.normalize = st_model._modules["4"]

        # Force float32
        for p in self.parameters():
            p.data = p.data.float()

        # Single forward pass only: disable KV cache so Dynamo does not
        # materialize past-key/value tensors in the compiled graph.
        self.transformer.auto_model.config.use_cache = False
        # Trace the HF attention math directly: the frontend's SDPA lowering
        # currently assumes batch size 1 when reshaping the attention mask.
        self.transformer.auto_model.set_attn_implementation("eager")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 1. Transformer (Gemma3TextModel)
        auto_model = self.transformer.auto_model
        # Embed IDs first to preserve the runner's (IDs, mask) input order.
        inputs_embeds = auto_model.embed_tokens(input_ids)
        # Full attention masks padding; sliding attention also masks distance.
        additive_mask = (
            1.0 - attention_mask[:, None, None, :].float()
        ) * torch.finfo(inputs_embeds.dtype).min
        additive_mask = additive_mask.expand(
            -1, 1, input_ids.shape[1], -1
        ).contiguous()
        sliding_mask = additive_mask
        if input_ids.shape[1] > auto_model.config.sliding_window:
            positions = torch.arange(
                input_ids.shape[1], device=input_ids.device
            )
            outside_window = (
                positions[:, None] - positions[None, :]
            ).abs() >= auto_model.config.sliding_window
            sliding_mask = additive_mask.masked_fill(
                outside_window, torch.finfo(inputs_embeds.dtype).min
            )
        trans_out = self.transformer(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": {
                    "full_attention": additive_mask,
                    "sliding_attention": sliding_mask,
                },
            }
        )
        token_emb = trans_out["token_embeddings"]

        # 2. Mean Pooling (traceable: all tensor ops, no control flow)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        )
        sum_emb = torch.sum(token_emb * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / sum_mask

        # 3. Dense 768 -> 3072 (Linear + Identity activation)
        pooled = {"sentence_embedding": pooled}
        d1_out = self.dense1(pooled)["sentence_embedding"]

        # 4. Dense 3072 -> 768 (Linear + Identity activation)
        d2_out = self.dense2({"sentence_embedding": d1_out})[
            "sentence_embedding"
        ]

        # 5. L2 Normalize
        normed = self.normalize({"sentence_embedding": d2_out})[
            "sentence_embedding"
        ]

        return normed


# ==============================================================================
# 1. Argument parsing
# ==============================================================================
p = argparse.ArgumentParser(description="embeddinggemma-300m AOT importer")
p.add_argument("--spec", required=True, help="Variant spec JSON")
p.add_argument("--output-dir", required=True, help="Directory to save outputs")
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)
variant = spec.get("variant", "f32")
if variant not in ("f32", "fp16_comp3"):
    raise ValueError(f"Unsupported EmbeddingGemma variant: {variant}")

model_path = (
    os.environ.get("EMBEDDINGGEMMA_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "google/embeddinggemma-300m")
)
max_seq_len = int(spec.get("max_seq_len", 128))
batch_size = int(spec.get("batch_size", 1))
if min(batch_size, max_seq_len) < 1:
    raise ValueError("batch_size and max_seq_len must be positive")
os.makedirs(a.output_dir, exist_ok=True)

# ==============================================================================
# 2. Load model
# ==============================================================================
print("[import-embeddinggemma] Loading embeddinggemma-300m from:", model_path)
model = EmbeddingGemmaWrapper(model_path)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

# ==============================================================================
# 3. Initialize Dynamo Compiler
# ==============================================================================
dc = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# ==============================================================================
# 4. Dummy inputs (fixed shape; matches the original PR)
# ==============================================================================
dummy_input_ids = torch.ones((batch_size, max_seq_len), dtype=torch.int64)
dummy_attention_mask = torch.ones((batch_size, max_seq_len), dtype=torch.int64)
print("[import-embeddinggemma] Dummy inputs:")
print(f"   input_ids:      {tuple(dummy_input_ids.shape)}")
print(f"   attention_mask: {tuple(dummy_attention_mask.shape)}")

# ==============================================================================
# 5. Trace the model
# ==============================================================================
print("\n[import-embeddinggemma] Tracing forward graph...")

with torch.no_grad():
    graphs = dc.importer(
        model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
    )

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"


graph = graphs[0]
params = dc.imported_params[graph]
print(
    f"[import-embeddinggemma] {len(graphs)} graph(s), "
    f"{len(params)} params, "
    f"{sum(p.numel() for p in params):,} elems"
)

# ==============================================================================
# 6. Graph optimization
# ==============================================================================
graph.perform([eliminate_transpose])
kernel_path = os.path.join(a.output_dir, "embeddinggemma_ime.mlir")
if variant == "fp16_comp3":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embeddinggemma_ime import (
        PROJECTION_COUNTS,
        RMS_NORM_COUNT,
        gen_kernels,
        lower_calls,
        rewrite_graph,
    )

    kernels = rewrite_graph(graph, batch_size * max_seq_len)
    with open(kernel_path, "w") as f:
        f.write(gen_kernels(kernels))
    print(
        f"[import-embeddinggemma] {sum(PROJECTION_COUNTS.values())} Linear calls, "
        f"{RMS_NORM_COUNT} RMSNorm calls, {len(kernels)} generated wrappers"
    )
elif os.path.exists(kernel_path):
    os.remove(kernel_path)
graph.fuse_ops([simply_fuse])

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
subgraph_module = driver.subgraphs[0]._imported_module
if variant == "fp16_comp3":
    lower_calls(subgraph_module)
forward_module = driver.construct_main_graph(True)
if target_cpu := spec.get("target_cpu"):
    for module in (subgraph_module, forward_module):
        with module.context:
            for op in module.body.operations:
                if op.operation.name == "func.func":
                    op.attributes["llvm.target_cpu"] = ir.StringAttr.get(
                        target_cpu
                    )
subgraph_module.operation.verify()
forward_module.operation.verify()

# ==============================================================================
# 7. Save outputs at the output-dir ROOT (single_forward build convention)
# ==============================================================================
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as module_file:
    print(subgraph_module, file=module_file)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as module_file:
    print(forward_module, file=module_file)

all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
).astype(numpy.float32, copy=False)
if all_param.size != int(spec["params_size"]):
    raise ValueError("Exported parameter count differs from the model spec")
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))

print(
    f"[import-embeddinggemma] Wrote forward.mlir, subgraph0.mlir, "
    f"arg0.data ({all_param.size} f32 elems) to {a.output_dir}"
)
