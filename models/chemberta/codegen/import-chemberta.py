#!/usr/bin/env python3
# ===- import-chemberta.py - ChemBERTa AOT importer ----------------------===//
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
# ===----------------------------------------------------------------------===//
#
# ChemBERTa (DeepChem/ChemBERTa-77M-MLM, a RoBERTa-style MLM encoder) AOT
# importer adapted from the original PR import_model.py to the buddy-codegen
# single_forward interface: `--spec` + `--output-dir`, with MLIR + weights
# written at the output-dir ROOT (not layer_partitioned/).
#
# The produced forward ABI (mirrors the ColBERTv2 encoder import):
#   forward(weights: memref<params_size x f32>,
#           position: memref<position_buffer_size x i64>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           attention_mask: memref<1 x max_seq_len x i64>)
#     -> (logits: memref<1 x max_seq_len x vocab_size x f32>)
#
# Usage:
#   python import-chemberta.py --spec specs/f32.json --output-dir <dir>
#
# The local HuggingFace snapshot is read from the CHEMBERTA_MODEL_PATH
# environment variable (fallback: spec["hf_model_path"]).
#
# ===----------------------------------------------------------------------===//
import argparse, os, json, numpy, torch
import torch._dynamo; torch._dynamo.config.suppress_errors = True
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForMaskedLM

p = argparse.ArgumentParser(description="ChemBERTa AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()
with open(a.spec) as f:
    spec = json.load(f)
model_path = (os.environ.get("CHEMBERTA_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "DeepChem/ChemBERTa-77M-MLM"))
max_seq_len = int(spec.get("max_seq_len", 128))
os.makedirs(a.output_dir, exist_ok=True)

print(f"[import-chemberta] Loading ChemBERTa-77M-MLM from: {model_path}")
m = AutoModelForMaskedLM.from_pretrained(
    model_path, torch_dtype=torch.float32).eval()
m.config.use_cache = False
print(f"  model class: {type(m).__name__}, params: "
      f"{sum(pp.numel() for pp in m.parameters()):,}")

dc = DynamoCompiler(primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=inductor_decomp,
                    func_name="forward")
dummy = torch.zeros((1, max_seq_len), dtype=torch.int64)
mask = torch.ones((1, max_seq_len), dtype=torch.int64)
with torch.no_grad():
    g = dc.importer(m, input_ids=dummy, attention_mask=mask)
print(f"[import-chemberta] {len(g)} graphs")
assert len(g) == 1, f"expected 1 graph, got {len(g)}"
graph = g[0]
params = dc.imported_params[graph]
print(f"[import-chemberta] first graph: {len(params)} params, "
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
print(f"[import-chemberta] Wrote forward.mlir, subgraph0.mlir, arg0.data "
      f"to {a.output_dir}")
