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
# BGE-Reranker-v2-M3 Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: XLMRobertaForSequenceClassification (cross-encoder reranker)
#   - 24 layers, hidden=1024, heads=16, vocab=250002
#   - Input: (input_ids, attention_mask) — single text pair
#   - Output: logit score (relevance)
#
# ===---------------------------------------------------------------------------

import argparse
import os
import numpy
import torch
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
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description="BGE-Reranker Model AOT Importer")
parser.add_argument("--output-dir", type=str, default="./", help="Output directory")
parser.add_argument("--precision", type=str, default="f32", choices=["f32"])
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("[BGE-Import] Loading BGE-Reranker-v2-M3...")
model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-v2-m3", dtype=torch.float32
).eval()
print(f"   hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}, heads={model.config.num_attention_heads}")

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

max_seq_len = 512
dummy_ids = torch.ones((1, max_seq_len), dtype=torch.int64)
dummy_mask = torch.ones((1, max_seq_len), dtype=torch.int64)

print(f"[BGE-Import] Dummy inputs: {dummy_ids.shape}")

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, input_ids=dummy_ids, attention_mask=dummy_mask)

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
print(f"[BGE-Import] 1 graph, {len(params)} params")

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
print("[BGE-Import] Done!")
