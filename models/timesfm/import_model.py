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
# TimesFM 2.5 (200M) Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: Time Series Foundation Model
#   - 20 transformer layers, hidden=1280, heads=16, patch_length=32
#   - Input: (batch, num_patches, 32) time series + masks
#   - Output: (embeddings, point_forecast, quantile_forecast)
#
# ===---------------------------------------------------------------------------

import argparse
import os
import numpy
import torch
import torch.nn as nn
import timesfm
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


# ==============================================================================
# 1. Build a clean wrapper module for Dynamo tracing
# ==============================================================================

class TimesFMWrapper(nn.Module):
    """Wraps TimesFM 2.5 for single-graph Dynamo tracing.

    The raw model returns a tuple of 4 tensors + decode_caches.
    We simplify to return only the point forecast tensor.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        out, _caches = self.base_model(inputs=inputs, masks=masks)
        # out = (input_embeddings, output_embeddings, point_forecast, quantile_forecast)
        _, _, point_forecast, _ = out
        return point_forecast


# ==============================================================================
# 2. Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description="TimesFM 2.5 Model AOT Importer")
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
    help="Precision mode.",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 3. Load model
# ==============================================================================

print("[TimesFM-Import] Loading TimesFM 2.5 200M PyTorch...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", backend="cpu"
)
base_model = tfm.model
# Force all params to CPU explicitly
base_model = base_model.cpu()
for p in base_model.parameters():
    p.data = p.data.cpu()
base_model.eval()

model = TimesFMWrapper(base_model)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

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

num_patches = 16
patch_length = 32
dummy_inputs = torch.randn(1, num_patches, patch_length, dtype=torch.float32)
dummy_masks = torch.ones(1, num_patches, patch_length, dtype=torch.float32)

print(f"[TimesFM-Import] Dummy inputs:")
print(f"   inputs:  {dummy_inputs.shape}")
print(f"   masks:   {dummy_masks.shape}")

# ==============================================================================
# 6. Trace the model
# ==============================================================================

print("\n[TimesFM-Import] Tracing forward graph...")
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model,
        inputs=dummy_inputs,
        masks=dummy_masks,
    )

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]

params = dynamo_compiler.imported_params[graph]
print(f"[TimesFM-Import] Graph captured. Params: {len(params)} tensors.")

# ==============================================================================
# 7. Graph optimizations
# ==============================================================================

print("[TimesFM-Import] Running graph transforms...")
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
print(f"\n[TimesFM-Import] Writing MLIR files to: {layer_dir}")

with open(os.path.join(layer_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

print(f"[TimesFM-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[TimesFM-Import] Done!\n")
