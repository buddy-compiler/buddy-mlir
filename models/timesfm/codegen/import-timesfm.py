#!/usr/bin/env python3
# ===- import-timesfm.py - TimesFM 2.5 AOT importer ----------------------===//
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
# ===---------------------------------------------------------------------------//
#
# TimesFM 2.5 (200M) AOT importer for the buddy-codegen single_forward
# interface: `--spec <spec.json> --output-dir <dir>`.
#
# Architecture: Time Series Foundation Model
#   - 20 transformer layers, hidden=1280, heads=16, patch_length=32
#   - Input: (batch, num_patches, patch_length) time series + masks
#   - Trace target: forward(inputs, masks) -> point_forecast
#
# Adapted from the original PR import_model.py:
#   - CLI is now --spec/--output-dir (single_forward convention).
#   - The local HF snapshot is read from the TIMESFM_MODEL_PATH env var
#     (fallback to spec["hf_model_path"]).
#   - Only `graph.fuse_ops([simply_fuse])` is applied (no extra transforms).
#   - subgraph0.mlir, forward.mlir and arg0.data are written to the
#     output-dir ROOT (not a layer_partitioned/ sub-directory).
#
# ===---------------------------------------------------------------------------//

import argparse
import json
import os

import numpy
import torch
import torch.nn as nn
import torch._dynamo

import timesfm
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp

torch._dynamo.config.suppress_errors = True


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
# 2. Argument parsing (single_forward convention)
# ==============================================================================

parser = argparse.ArgumentParser(description="TimesFM 2.5 Model AOT Importer")
parser.add_argument("--spec", type=str, required=True,
                    help="Variant spec JSON (models/timesfm/specs/f32.json).")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory to save output files.")
args = parser.parse_args()

with open(args.spec) as spec_file:
    spec = json.load(spec_file)

model_path = (os.environ.get("TIMESFM_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "google/timesfm-2.5-200m-pytorch"))

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 3. Load model
# ==============================================================================

print(f"[TimesFM-Import] Loading TimesFM 2.5 200M PyTorch from {model_path} ...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path, backend="cpu")
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
# 5. Dummy inputs (fixed input shape from the original PR)
# ==============================================================================

num_patches = int(spec.get("num_patches", 16))
patch_length = int(spec.get("patch_length", 32))
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
print(f"[TimesFM-Import] Graph captured. Params: {len(params)} tensors, "
      f"{sum(p.numel() for p in params):,} elems.")

# ==============================================================================
# 7. Graph optimizations (only simply_fuse)
# ==============================================================================

print("[TimesFM-Import] Running graph transforms...")
graph.fuse_ops([simply_fuse])

graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()


def _repair_subgraph0(module_text):
    """Repair invalid `tosa.slice` start offsets the buddy-codegen trace emits.

    TimesFM's traced graph shifts the input along the time axis (e.g.
    `x[:, :, :-1]`), which Dynamo emits as `tosa.slice` with a NEGATIVE start
    offset (e.g. `dense<[0, 0, -1]>` on a `tensor<1x16x32xf32>`). TOSA requires
    offsets to be non-negative, so `tosa-to-tensor` fails with
    "expected offsets to be non-negative" and the subgraph silently compiles to
    nothing. For a static input the negative offset -k on axis i is exactly
    dim_i - k, so we rewrite the start constant in place.
    """
    import re

    # Collect start-const name -> input shape dims for every tosa.slice.
    slice_pat = re.compile(
        r'tosa\.slice (\S+), (\S+), (\S+) : \(tensor<([0-9x]+xf32)>')
    start_to_dims = {}
    for m in slice_pat.finditer(module_text):
        dims = [int(d) for d in m.group(4).split('x')[:-1]]
        start_to_dims.setdefault(m.group(2).lstrip('%'), dims)

    def _const_repl(mo):
        name, vals, n = mo.group(1), mo.group(2), mo.group(3)
        dims = start_to_dims.get(name)
        if dims is None:
            return mo.group(0)
        nums = [int(v) for v in vals.split(',')]
        fixed = [
            str(dims[i] + v) if (v < 0 and i < len(dims)) else str(v)
            for i, v in enumerate(nums)
        ]
        return (f'    %{name} = tosa.const_shape  '
                f'{{values = dense<[{", ".join(fixed)}]> : tensor<{n}xindex>')

    const_pat = re.compile(
        r'    %(\w+) = tosa\.const_shape  '
        r'\{values = dense<\[([^\]]*)\]> : tensor<(\d+)xindex>')
    return const_pat.sub(_const_repl, module_text)


# ==============================================================================
# 8. Save outputs (to the output-dir root)
# ==============================================================================

print(f"\n[TimesFM-Import] Writing MLIR files to: {output_dir}")

with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(_repair_subgraph0(str(driver.subgraphs[0]._imported_module)),
          file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

print("[TimesFM-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print(f"[TimesFM-Import] Wrote forward.mlir, subgraph0.mlir, arg0.data "
      f"({all_param.size:,} f32 elems) to {output_dir}")
print("[TimesFM-Import] Done!\n")
