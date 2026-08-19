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
# PaddleOCR-VL-0.9B Official Model Importer (Adapted for Buddy-MLIR Pipeline)
#
# ===---------------------------------------------------------------------------

import argparse
import os
import re
import shutil

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
from transformers import AutoModel

# ==============================================================================
# 0. Patch HF model code for Dynamo fullgraph compatibility
# ==============================================================================

snapshot_path = "/home/hanyuning/.cache/huggingface/hub/models--lvyufeng--PaddleOCR-VL-0.9B/snapshots/b68da00edeb02675e68282c6d2fee98e03a58213/modeling_paddleocr_vl.py"
hf_module_dir = "/home/hanyuning/.cache/huggingface/modules/transformers_modules/lvyufeng/PaddleOCR-VL-0.9B/b68da00edeb02675e68282c6d2fee98e03a58213"
hf_file_path = os.path.join(hf_module_dir, "modeling_paddleocr_vl.py")

print("[PaddleOCR-Import] Patching HF model for fullgraph tracing...")
os.makedirs(hf_module_dir, exist_ok=True)
shutil.copy(snapshot_path, hf_file_path)

with open(hf_file_path, "r", encoding="utf-8") as f:
    code = f.read()

print("   -> (F) ROPE_INIT_FUNCTIONS compat handled in-code (see below).")

# --- (A) thw loop: remove numpy/detach/cpu/numpy deps ---
old_thw_loop = """                pro = 0
                for idx, thw in enumerate(image_grid_thw):
                    thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                    numel = np.prod(thw_tuple)
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = torch.arange(numel) % int(np.prod(thw_tuple[1:]))
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)"""

new_thw_loop = """                pro = 0
                for idx, thw in enumerate(image_grid_thw):
                    thw_tuple = tuple(thw) if isinstance(thw, (list, tuple)) else tuple(thw.tolist())
                    t, h, w = int(thw_tuple[0]), int(thw_tuple[1]), int(thw_tuple[2])
                    numel = t * h * w
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = torch.arange(numel) % (h * w)
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)"""

if old_thw_loop in code:
    code = code.replace(old_thw_loop, new_thw_loop)
    print("   -> (A) thw loop replaced (numpy/detach removed).")
else:
    print("   -> (A) thw loop block not found, using fallback...")
    code = code.replace(
        "thw_tuple = tuple(thw.detach().cpu().numpy().tolist())",
        "thw_tuple = tuple(thw) if isinstance(thw, (list, tuple)) else tuple(thw.tolist())",
    )
    code = code.replace("numel = np.prod(thw_tuple)", "numel = int(np.prod(thw_tuple))")
    code = re.sub(
        r"int\(np\.prod\(thw_tuple\[1:\]\)\)",
        "(int(thw_tuple[1]) * int(thw_tuple[2]))",
        code,
    )

# --- (B) data-dependent assert -> True ---
code = re.sub(
    r"sum\(\[np\.prod\(x\) for x in flatten_image_grid_thw\]\)\s*==\s*embeddings\.shape\[1\]",
    "True",
    code,
)
code = re.sub(
    r"sum\(\[np\.prod\(x\) for x in flatten_image_grid_thw\]\)\s*==\s*hidden_states\.shape\[1\]",
    "True",
    code,
)
code = re.sub(r"assert batch_size == 1", "pass # assert batch_size == 1", code)
print("   -> (B) assert control-flow replaced.")

# --- (C) cu_seqlens dynamic slice -> static squeeze ---
old_slice = """        sample_hidden_state = list()
        assert cu_seqlens is not None
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)"""

new_slice = """        assert cu_seqlens is not None
        sample_hidden_state = [last_hidden_state.squeeze(0)]"""

if old_slice in code:
    code = code.replace(old_slice, new_slice)
    print("   -> (C) cu_seqlens dynamic slice replaced.")
else:
    print("   -> (C) cu_seqlens pattern not found (may already be patched).")

# --- (D) n_image_tokens data-dependent check -> removed ---
old_check = """                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                # image_embeds is a list of tensor, each tensor is a image feature,I want to concat them all into a tensor
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )"""

new_check = """                # image_embeds is a list of tensor, each tensor is a image feature
                image_embeds = torch.cat(image_embeds, dim=0)"""

if old_check in code:
    code = code.replace(old_check, new_check)
    print("   -> (D) n_image_tokens check removed.")
else:
    print("   -> (D) n_image_tokens block not found.")

# --- (E) SigLIPRotaryEmbedding.forward: fixed-size rope table ---
old_rope_forward = """    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs"""

new_rope_forward = """    def forward(self, seqlen=None) -> torch.Tensor:
        max_len = 256
        seq = torch.arange(
            max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs"""

if old_rope_forward in code:
    code = code.replace(old_rope_forward, new_rope_forward)
    print("   -> (E) SigLIPRotaryEmbedding.forward (fixed rope).")
else:
    print("   -> (E) SigLIPRotaryEmbedding.forward not found.")

with open(hf_file_path, "w", encoding="utf-8") as f:
    f.write(code)

# Clear Python bytecode cache to ensure patched file is loaded
import py_compile
pycache = os.path.join(os.path.dirname(hf_file_path), "__pycache__")
if os.path.exists(pycache):
    import shutil
    shutil.rmtree(pycache)
    print("   -> Cleared __pycache__ to force recompile.")

print("[PaddleOCR-Import] HF model patched successfully.\n")

# ==============================================================================
# 1. Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description="PaddleOCR-VL-0.9B Model AOT Importer")
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
# 2. Load model
# ==============================================================================

# --- (F) Monkey-patch ROPE_INIT_FUNCTIONS for transformers >=5.0 compat ---
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config, device, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    print("   -> (F) ROPE_INIT_FUNCTIONS 'default' monkey-patched.")

print("[PaddleOCR-Import] Loading PaddleOCR-VL-0.9B model...")
model = AutoModel.from_pretrained(
    "lvyufeng/PaddleOCR-VL-0.9B", trust_remote_code=True
).eval()
model.config.use_cache = False

image_token_id = model.config.image_token_id
print(f"   image_token_id = {image_token_id}")

# ==============================================================================
# 3. Initialize Dynamo Compiler (prefill + decode for pipeline compatibility)
# ==============================================================================

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

# ==============================================================================
# 4. Dummy inputs
# ==============================================================================

# image_grid_thw = [[1, 54, 72]] -> 3888 patches
# Projector merge_kernel_size=(2,2): output = 1 * 27 * 36 = 972 tokens
n_img_tokens = 972
total_len = 982

input_ids = torch.full((1, total_len), 1, dtype=torch.int64)
input_ids[0, :n_img_tokens] = image_token_id
attention_mask = torch.ones((1, total_len), dtype=torch.int64)
pixel_values = torch.zeros((3888, 3, 14, 14), dtype=torch.float32)
# position_ids shape: (3, batch_size, seq_len) — bypass get_rope_index
position_ids = torch.zeros((3, 1, total_len), dtype=torch.int64)

static_image_grid_thw = [[1, 54, 72]]

print(f"[PaddleOCR-Import] Dummy inputs:")
print(f"   input_ids:     {input_ids.shape}")
print(f"   attention_mask:{attention_mask.shape}")
print(f"   pixel_values:  {pixel_values.shape}")
print(f"   position_ids:  {position_ids.shape}")

# ==============================================================================
# 5. Trace the model (prefill + decode for pipeline compatibility)
# ==============================================================================

print("\n[PaddleOCR-Import] Tracing prefill graph...")
with torch.no_grad():
    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=static_image_grid_thw,
        position_ids=position_ids,
        return_dict=False,
    )

assert len(graphs_prefill) == 1, f"Expected 1 prefill graph, got {len(graphs_prefill)}"
graph_prefill = graphs_prefill[0]

print("[PaddleOCR-Import] Tracing decode graph...")
torch._dynamo.reset()
with torch.no_grad():
    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=static_image_grid_thw,
        position_ids=position_ids,
        return_dict=False,
    )

assert len(graphs_decode) == 1, f"Expected 1 decode graph, got {len(graphs_decode)}"
graph_decode = graphs_decode[0]

params = dynamo_compiler_prefill.imported_params[graph_prefill]
print(f"[PaddleOCR-Import] Graphs captured. Params: {len(params)} tensors.")

# ==============================================================================
# 6. Graph optimizations (prefill & decode)
# ==============================================================================

print("[PaddleOCR-Import] Running graph transforms...")
for g in [graph_prefill, graph_decode]:
    g.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

pattern_list = [
    simply_fuse,
    apply_classic_fusion,
]
graph_prefill.fuse_ops(pattern_list)
graph_decode.fuse_ops(pattern_list)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop("subgraph0")
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop("subgraph0")
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graph_prefill)
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graph_decode)
driver_decode.subgraphs[0].lower_to_top_level_ir()

# ==============================================================================
# 7. Save outputs (pipeline-compatible naming)
# ==============================================================================

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)
print(f"\n[PaddleOCR-Import] Writing MLIR files to: {layer_dir}")

with open(os.path.join(layer_dir, "subgraph0_prefill.mlir"), "w") as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_prefill.mlir"), "w") as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

with open(os.path.join(layer_dir, "subgraph0_decode.mlir"), "w") as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_decode.mlir"), "w") as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

print(f"[PaddleOCR-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[PaddleOCR-Import] Done!\n")
