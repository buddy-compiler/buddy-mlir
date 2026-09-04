#!/usr/bin/env python3
# ===- import-paddleocr.py - PaddleOCR-VL-0.9B AOT importer --------------===//
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
# PaddleOCR-VL-0.9B (lvyufeng/PaddleOCR-VL-0.9B) AOT importer for the buddy-codegen
# `single_forward` interface: `--spec <spec.json> --output-dir <dir>`.
#
# Adapted from the original PR's `import_model.py` (which traced prefill+decode
# graphs and wrote into `layer_partitioned/`) to the new single_forward format:
#   - ONE Dynamo trace of `model.forward` over the fixed-shape OCR input
#     (972 vision tokens + 10 text tokens, 982 total).
#   - Fusion uses ONLY `graph.fuse_ops([simply_fuse])`.
#   - Writes `subgraph0.mlir`, `forward.mlir` and `arg0.data` into the
#     output-directory ROOT (not layer_partitioned/).
#
# The HuggingFace remote modeling code (`modeling_paddleocr_vl.py`) is staged
# under /tmp and patched (same edits the original PR applied) so the whole
# vision+language path is Dynamo-fullgraph traceable:
#   (A) thw loop            : numpy/detach/cpu deps removed
#   (B) asserts             : data-dependent asserts disabled
#   (C) cu_seqlens slice    : dynamic per-sample slice -> static squeeze
#   (D) image token check   : data-dependent feature-count check removed
#   (E) SigLIP rope         : fixed-size 256-entry rope table
#
# The local model path is read from the PADDLEOCR_MODEL_PATH environment
# variable (set by buddy_add_model via LOCAL_MODEL_ENV), falling back to
# spec["hf_model_path"]. The snapshot must contain the remote-code files
# (modeling_paddleocr_vl.py, etc.) as shipped by lvyufeng/PaddleOCR-VL-0.9B.
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import re
import shutil
import tempfile

import numpy
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

# ==============================================================================
# 0. Argument parsing / spec / model path
# ==============================================================================

parser = argparse.ArgumentParser(description="PaddleOCR-VL-0.9B AOT importer")
parser.add_argument("--spec", required=True, help="Variant spec JSON")
parser.add_argument("--output-dir", required=True,
                    help="Directory to save subgraph0.mlir / forward.mlir / arg0.data")
args = parser.parse_args()

with open(args.spec) as spec_file:
    spec = json.load(spec_file)

model_path = (os.environ.get("PADDLEOCR_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "lvyufeng/PaddleOCR-VL-0.9B"))
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print(f"[PaddleOCR-Import] Spec: {args.spec}")
print(f"[PaddleOCR-Import] Model path: {model_path}")

# ==============================================================================
# 1. Patch HF remote modeling code for Dynamo fullgraph compatibility.
#    The patched copy is re-derived from the local snapshot each run.
# ==============================================================================

snapshot_path = os.path.join(model_path, "modeling_paddleocr_vl.py")
if not os.path.isfile(snapshot_path):
    raise RuntimeError(
        f"[PaddleOCR-Import] modeling_paddleocr_vl.py not found at {snapshot_path}. "
        "PASS the local HF snapshot via PADDLEOCR_MODEL_PATH (trust_remote_code "
        "remote files must be present in the snapshot)."
    )

# Stage a patched copy of the model directory.  When loading remote code from a
# LOCAL path, transformers re-syncs its transformers_modules cache with the
# source dir on every load (filecmp-based copy), so patching the cache copy
# directly is unreliable.  Instead we stage the snapshot under /tmp:
#   - symlink every snapshot file (config.json, model.safetensors, ...),
#   - replace modeling_paddleocr_vl.py with a real PATCHED copy,
#   - load from the staged dir.
# transformers then copies the (already-patched) modeling file into its
# `transformers_modules/<staged-basename>/` cache once and never re-copies it.
staged_dir = os.environ.get(
    "PADDLEOCR_IMPORT_STAGE_DIR"
) or os.path.join(tempfile.gettempdir(), "paddleocr_model_stage")
os.makedirs(staged_dir, exist_ok=True)
for _name in os.listdir(model_path):
    _src = os.path.join(model_path, _name)
    _dst = os.path.join(staged_dir, _name)
    if os.path.lexists(_dst):
        os.remove(_dst)
    os.symlink(_src, _dst)

hf_file_path = os.path.join(staged_dir, "modeling_paddleocr_vl.py")
if os.path.lexists(hf_file_path):
    os.remove(hf_file_path)
shutil.copy(snapshot_path, hf_file_path)
print(f"[PaddleOCR-Import] Staged model dir: {staged_dir}")

print("[PaddleOCR-Import] Patching HF model for fullgraph tracing...")
with open(hf_file_path, "r", encoding="utf-8") as f:
    code = f.read()

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

# --- (B) data-dependent assert -> True / disabled ---
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

# Clear Python bytecode caches (both the staged dir and the transformers
# modules cache) to force the patched file to be recompiled.
for _pyc in (
    os.path.join(os.path.dirname(hf_file_path), "__pycache__"),
    os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                 "modules", "transformers_modules",
                 os.path.basename(staged_dir), "__pycache__"),
):
    if os.path.exists(_pyc):
        shutil.rmtree(_pyc)
        print(f"   -> Cleared {_pyc} to force recompile.")

print("[PaddleOCR-Import] HF model patched successfully.\n")

# ==============================================================================
# 2. Load model
# ==============================================================================

# Monkey-patch ROPE_INIT_FUNCTIONS for transformers compat (no-op when a
# 'default' entry already exists, e.g. transformers >= 4.46).
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
model = AutoModel.from_pretrained(staged_dir, trust_remote_code=True).eval()
model.config.use_cache = False

image_token_id = model.config.image_token_id
print(f"   image_token_id = {image_token_id}")

# ==============================================================================
# 3. Initialize Dynamo Compiler (single forward graph)
# ==============================================================================

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# ==============================================================================
# 4. Dummy inputs (fixed shapes, replicated from the original PR)
# ==============================================================================
# image_grid_thw = [1, 54, 72] -> 3888 patches (t*h*w with merge kept upstream)
# Projector merge_kernel_size=(2,2) -> 3888/4 = 972 image tokens
# + 10 text tokens = 982 total sequence length.

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
# 5. Trace the model (single forward graph)
# ==============================================================================

print("\n[PaddleOCR-Import] Tracing forward graph...")
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=static_image_grid_thw,
        position_ids=position_ids,
        return_dict=False,
    )

assert len(graphs) == 1, f"Expected 1 forward graph, got {len(graphs)}"
graph = graphs[0]

params = dynamo_compiler.imported_params[graph]
n_param_elems = sum(p.numel() for p in params)
print(f"[PaddleOCR-Import] Graph captured. Params: {len(params)} tensors, "
      f"{n_param_elems:,} elements.")

# ==============================================================================
# 6. Graph optimization (simply_fuse ONLY)
# ==============================================================================

print("[PaddleOCR-Import] Running graph transforms...")
graph.fuse_ops([simply_fuse])

graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

# ==============================================================================
# 7. Save outputs (single_forward naming at output-dir ROOT)
# ==============================================================================

print(f"\n[PaddleOCR-Import] Writing MLIR files to: {output_dir}")

with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

print(f"[PaddleOCR-Import] Writing weight data...")
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print(f"[PaddleOCR-Import] Done! "
      f"arg0.data has {n_param_elems:,} f32 elements "
      f"({all_param.nbytes / 1e9:.2f} GB).")
