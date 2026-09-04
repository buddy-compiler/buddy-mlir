#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
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
# embeddinggemma-300m Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: Gemma3TextModel → Mean Pooling → Dense(768→3072) →
#               Dense(3072→768) → L2 Normalize → 768-dim embedding
#
# ===---------------------------------------------------------------------------

import argparse
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from sentence_transformers import SentenceTransformer


# ==============================================================================
# 1. Build a clean wrapper module for Dynamo tracing
# ==============================================================================

class EmbeddingGemmaWrapper(nn.Module):
    """Wraps the SentenceTransformer pipeline as a clean nn.Module for tracing.

    Takes (input_ids, attention_mask) → returns 768-dim L2-normalized embedding.
    """

    def __init__(self, model_name="google/embeddinggemma-300m"):
        super().__init__()
        st_model = SentenceTransformer(model_name, device="cpu")
        st_model.eval()

        # Extract sub-modules
        self.transformer = st_model._first_module()
        self.pooling = st_model._modules["1"]
        self.dense1 = st_model._modules["2"]
        self.dense2 = st_model._modules["3"]
        self.normalize = st_model._modules["4"]

        # Force float32
        for p in self.parameters():
            p.data = p.data.float()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 1. Transformer
        trans_out = self.transformer(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        token_emb = trans_out["token_embeddings"]

        # 2. Mean Pooling (traceable: all tensor ops, no control flow)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / sum_mask

        # 3. Dense 768→3072 (Linear + Identity activation)
        pooled = {"sentence_embedding": pooled}
        d1_out = self.dense1(pooled)["sentence_embedding"]

        # 4. Dense 3072→768 (Linear + Identity activation)
        d2_out = self.dense2({"sentence_embedding": d1_out})["sentence_embedding"]

        # 5. L2 Normalize
        normed = self.normalize({"sentence_embedding": d2_out})["sentence_embedding"]

        return normed


# ==============================================================================
# 2. Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description="embeddinggemma-300m Model AOT Importer")
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
    help="Precision mode. Currently only 'f32' is supported.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 3. Load model
# ==============================================================================

print("[EmbeddingGemma-Import] Loading embeddinggemma-300m...")
model = EmbeddingGemmaWrapper("google/embeddinggemma-300m")
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# ==============================================================================
# 4. Initialize Dynamo Compiler
# ==============================================================================

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# ==============================================================================
# 5. Dummy inputs
# ==============================================================================

max_seq_len = 128
dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.int64)
dummy_attention_mask = torch.ones((1, max_seq_len), dtype=torch.int64)

print(f"[EmbeddingGemma-Import] Dummy inputs:")
print(f"   input_ids:       {dummy_input_ids.shape}")
print(f"   attention_mask:  {dummy_attention_mask.shape}")

# ==============================================================================
# 6. Trace the model
# ==============================================================================

print("\n[EmbeddingGemma-Import] Tracing forward graph...")
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
    )

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]

params = dynamo_compiler.imported_params[graph]
print(f"[EmbeddingGemma-Import] Graph captured. Params: {len(params)} tensors.")

# ==============================================================================
# 7. Graph optimizations
# ==============================================================================

print("[EmbeddingGemma-Import] Running graph transforms...")
graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

pattern_list = [
    simply_fuse,
    apply_classic_fusion,
]
graph.fuse_ops(pattern_list)

graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

# ==============================================================================
# 8. Save outputs
# ==============================================================================

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)
print(f"\n[EmbeddingGemma-Import] Writing MLIR files to: {layer_dir}")

with open(os.path.join(layer_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

print(f"[EmbeddingGemma-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[EmbeddingGemma-Import] Done!\n")
