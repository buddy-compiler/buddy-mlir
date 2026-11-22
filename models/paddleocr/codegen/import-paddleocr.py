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
from buddy.compiler.frontend import DynamoCompiler  # noqa: E402
from buddy.compiler.graph import GraphDriver  # noqa: E402
from buddy.compiler.graph.operation import *  # noqa: E402, F403
from buddy.compiler.graph.transform import simply_fuse  # noqa: E402
from buddy.compiler.graph.type import DeviceType  # noqa: E402
from buddy.compiler.ops import tosa  # noqa: E402
from torch._inductor.decomposition import (  # noqa: E402
    decompositions as inductor_decomp,  # noqa: E402
)
from transformers import AutoModel  # noqa: E402

# ==============================================================================
# 0. Argument parsing / spec / model path
# ==============================================================================

parser = argparse.ArgumentParser(description="PaddleOCR-VL-0.9B AOT importer")
parser.add_argument("--spec", required=True, help="Variant spec JSON")
parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory to save subgraph0.mlir / forward.mlir / arg0.data",
)
args = parser.parse_args()

with open(args.spec) as spec_file:
    spec = json.load(spec_file)

model_path = (
    os.environ.get("PADDLEOCR_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "lvyufeng/PaddleOCR-VL-0.9B")
)
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
staged_dir = os.environ.get("PADDLEOCR_IMPORT_STAGE_DIR") or os.path.join(
    tempfile.gettempdir(), "paddleocr_model_stage"
)
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
with open(hf_file_path, encoding="utf-8") as f:
    code = f.read()

# --- (A) thw loop: remove numpy/detach/cpu/numpy deps ---
old_thw_loop = """                pro = 0
                for idx, thw in enumerate(image_grid_thw):
                    thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                    numel = np.prod(thw_tuple)
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)"""

new_thw_loop = """                pro = 0
                for idx, thw in enumerate(image_grid_thw):
                    thw_tuple = tuple(thw)
                    t, h, w = int(thw_tuple[0]), int(thw_tuple[1]), int(thw_tuple[2])
                    numel = t * h * w
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = torch.arange(numel) % (h * w)
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)"""

if old_thw_loop not in code:
    raise RuntimeError(
        "Unsupported PaddleOCR-VL image-grid source; update patch A"
    )
code = code.replace(old_thw_loop, new_thw_loop)
print("   -> (A) thw loop replaced (numpy/detach removed).")

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

# In inference only the final position predicts the next token. Avoid a
# [sequence, vocabulary] projection when the package requests one logit row.
if spec.get("last_token_only", False):
    old_head = "logits = self.lm_head(hidden_states)"
    if code.count(old_head) != 1:
        raise RuntimeError("Unsupported PaddleOCR-VL LM head source")
    code = code.replace(
        old_head, "logits = self.lm_head(hidden_states[:, -1:, :])"
    )
    print("   -> LM head restricted to the final token.")

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
    os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "modules",
        "transformers_modules",
        os.path.basename(staged_dir),
        "__pycache__",
    ),
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
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

if "default" not in ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(
        config, device, seq_len=None, **kwargs
    ):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).float().to(device)
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    print("   -> (F) ROPE_INIT_FUNCTIONS 'default' monkey-patched.")

print("[PaddleOCR-Import] Loading PaddleOCR-VL-0.9B model...")
model = AutoModel.from_pretrained(
    staged_dir,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float16 if spec.get("precision") == "f16" else torch.float32,
).eval()
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
# 4. Fixed-shape trace inputs. A sample_image spec uses the model's actual
#    OCR processor and records the tensors for an identical runtime smoke test.
# ==============================================================================
if spec.get("sample_image"):
    from PIL import Image
    from transformers import AutoProcessor

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
    image_path = spec["sample_image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(repo_root, image_path)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    image = Image.open(image_path).convert("RGB")
    prompt = spec.get("prompt", "OCR:")
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"].to(model.visual.dtype)
    image_grid_thw = inputs["image_grid_thw"]
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    static_image_grid_thw = image_grid_thw.tolist()
    if static_image_grid_thw != [spec["image_grid_thw"]]:
        raise RuntimeError("OCR image grid does not match spec")
    n_img_tokens = int((input_ids == image_token_id).sum())
    total_len = input_ids.shape[1]
    if total_len != int(spec["max_seq_len"]):
        raise RuntimeError(
            f"OCR input has {total_len} tokens; spec expects {spec['max_seq_len']}"
        )
    if pixel_values.shape[0] != int(spec["num_image_patches"]):
        raise RuntimeError("OCR patch count does not match spec")
    if n_img_tokens != int(spec["num_image_tokens"]):
        raise RuntimeError("OCR image token count does not match spec")
    for name, tensor, ext in (
        ("input_ids", input_ids, "i64"),
        (
            "pixel_values",
            pixel_values,
            "f16" if spec.get("precision") == "f16" else "f32",
        ),
        ("attention_mask", attention_mask, "i64"),
        ("position_ids", position_ids, "i64"),
    ):
        tensor.contiguous().numpy().tofile(
            os.path.join(output_dir, f"{name}.{ext}")
        )
else:
    n_img_tokens = 972
    total_len = 982
    input_ids = torch.full((1, total_len), 1, dtype=torch.int64)
    input_ids[0, :n_img_tokens] = image_token_id
    attention_mask = torch.ones((1, total_len), dtype=torch.int64)
    pixel_values = torch.zeros((3888, 3, 14, 14), dtype=model.visual.dtype)
    position_ids = torch.zeros((3, 1, total_len), dtype=torch.int64)
    static_image_grid_thw = [[1, 54, 72]]

print("[PaddleOCR-Import] Trace inputs:")
print(f"   input_ids:     {input_ids.shape}")
print(f"   attention_mask:{attention_mask.shape}")
print(f"   pixel_values:  {pixel_values.shape}")
print(f"   position_ids:  {position_ids.shape}")

# ==============================================================================
# 5. Trace the model (single forward or Prefill + static KV-cache Decode)
# ==============================================================================

kv_cache_enabled = bool(spec.get("kv_cache", False))
max_cache_len = int(spec.get("max_cache_len", 384))

if kv_cache_enabled:
    import sys as _sys

    _model_mod = _sys.modules[model.__class__.__module__]
    num_layers = int(model.config.num_hidden_layers)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    num_attn_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    max_past_len = max_cache_len - 1

    class StaticListCache:
        def __init__(self, n_layers):
            self.keys = [None] * n_layers
            self.values = [None] * n_layers

        def update(
            self, key_states, value_states, layer_idx, cache_kwargs=None
        ):
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
            return key_states, value_states

        def get_seq_length(self, layer_idx=0):
            return (
                0
                if self.keys[layer_idx] is None
                else self.keys[layer_idx].shape[2]
            )

        def get_mask_sizes(self, cache_position, layer_idx=0):
            q_len = (
                cache_position
                if isinstance(cache_position, int)
                else cache_position.shape[0]
            )
            return q_len, 0

    class PrefillKVWrapper(torch.nn.Module):
        def __init__(self, inner, grid_thw):
            super().__init__()
            self.inner = inner
            self.grid_thw = grid_thw

        def forward(
            self, input_ids, attention_mask, pixel_values, position_ids
        ):
            cache = StaticListCache(num_layers)
            out = self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=self.grid_thw,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            prefill_k = torch.cat(cache.keys, dim=0)
            prefill_v = torch.cat(cache.values, dim=0)
            return out.logits, prefill_k, prefill_v

    class DecodeKVWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, position_ids, cache_k, cache_v, attn_mask):
            h = self.inner.model.embed_tokens(input_ids)
            cos, sin = self.inner.model.rotary_emb(h, position_ids)
            new_ks = []
            new_vs = []
            for idx, layer in enumerate(self.inner.model.layers):
                residual = h
                h_norm = layer.input_layernorm(h)
                attn = layer.self_attn
                q = (
                    attn.q_proj(h_norm)
                    .view(1, 1, num_attn_heads, head_dim)
                    .transpose(1, 2)
                )
                k = (
                    attn.k_proj(h_norm)
                    .view(1, 1, num_kv_heads, head_dim)
                    .transpose(1, 2)
                )
                v = (
                    attn.v_proj(h_norm)
                    .view(1, 1, num_kv_heads, head_dim)
                    .transpose(1, 2)
                )
                q, k = _model_mod.apply_multimodal_rotary_pos_emb(
                    q, k, cos, sin, attn.rope_scaling["mrope_section"]
                )
                new_ks.append(k)
                new_vs.append(v)
                full_k = torch.cat([cache_k[idx : idx + 1], k], dim=2)
                full_v = torch.cat([cache_v[idx : idx + 1], v], dim=2)
                full_k_rep = _model_mod.repeat_kv(
                    full_k, attn.num_key_value_groups
                )
                full_v_rep = _model_mod.repeat_kv(
                    full_v, attn.num_key_value_groups
                )
                scores = (
                    torch.matmul(q, full_k_rep.transpose(2, 3)) * attn.scaling
                    + attn_mask
                )
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                    q.dtype
                )
                a_out = (
                    torch.matmul(probs, full_v_rep)
                    .transpose(1, 2)
                    .reshape(1, 1, num_attn_heads * head_dim)
                )
                h = residual + attn.o_proj(a_out)
                residual = h
                h = residual + layer.mlp(layer.post_attention_layernorm(h))
            h = self.inner.model.norm(h)
            logits = self.inner.lm_head(h)
            return logits, torch.cat(new_ks, dim=0), torch.cat(new_vs, dim=0)

    print("\n[PaddleOCR-Import] Tracing Prefill (with KV cache) graph...")
    prefill_wrapper = PrefillKVWrapper(model, static_image_grid_thw).eval()
    with torch.no_grad():
        graphs = dynamo_compiler.importer(
            prefill_wrapper,
            input_ids,
            attention_mask,
            pixel_values,
            position_ids,
        )
else:
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
print(
    f"[PaddleOCR-Import] Graph captured. Params: {len(params)} tensors, "
    f"{n_param_elems:,} elements."
)

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

print("[PaddleOCR-Import] Writing weight data...")
# GraphDriver packs parameters by dtype, sorted by the dtype name. Preserve
# that grouping: FP16 weights also contain a small FP32 RoPE parameter pack.
dtypes = sorted({param.dtype for param in params}, key=str)
if len(dtypes) > 2:
    raise RuntimeError(f"Unsupported parameter dtypes: {dtypes}")
prefill_param_map = {}
prefill_totals = {}
for index, dtype in enumerate(dtypes):
    if dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"Unsupported parameter dtype: {dtype}")
    dtype_params = [param for param in params if param.dtype == dtype]
    offset = 0
    for param in dtype_params:
        key = (param.data_ptr(), param.numel())
        prefill_param_map.setdefault(key, (index, offset))
        offset += param.numel()
    prefill_totals[index] = offset
    packed = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in dtype_params]
    )
    expected = spec["params_size" if index == 0 else "params_f32_size"]
    if packed.size != expected:
        raise RuntimeError(
            f"arg{index} has {packed.size} elements; spec expects {expected}"
        )
    packed.tofile(os.path.join(output_dir, f"arg{index}.data"))
    print(
        f"   arg{index}.data: {packed.size:,} {dtype} elements "
        f"({packed.nbytes / 1e9:.3f} GB)"
    )

if kv_cache_enabled:
    print("\n[PaddleOCR-Import] Tracing Decode (1-token KV cache) graph...")
    decode_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_decode",
    )
    decode_wrapper = DecodeKVWrapper(model).eval()
    dec_dtype = (
        torch.float16 if spec.get("precision") == "f16" else torch.float32
    )
    dec_input_ids = torch.zeros((1, 1), dtype=torch.int64)
    dec_position_ids = torch.zeros((3, 1, 1), dtype=torch.int64)
    dec_cache_k = torch.zeros(
        (num_layers, num_kv_heads, max_past_len, head_dim), dtype=dec_dtype
    )
    dec_cache_v = torch.zeros(
        (num_layers, num_kv_heads, max_past_len, head_dim), dtype=dec_dtype
    )
    dec_attn_mask = torch.zeros((1, 1, 1, max_cache_len), dtype=dec_dtype)

    with torch.no_grad():
        dec_graphs = decode_compiler.importer(
            decode_wrapper,
            dec_input_ids,
            dec_position_ids,
            dec_cache_k,
            dec_cache_v,
            dec_attn_mask,
        )
    assert len(dec_graphs) == 1, (
        f"Expected 1 decode graph, got {len(dec_graphs)}"
    )
    dec_graph = dec_graphs[0]
    dec_params = decode_compiler.imported_params[dec_graph]
    print(
        f"[PaddleOCR-Import] Decode graph captured. Params: {len(dec_params)} "
        f"tensors, {sum(p.numel() for p in dec_params):,} elements."
    )
    dec_graph.fuse_ops([simply_fuse])
    dec_graph.op_groups["subgraph0"] = dec_graph.op_groups.pop("subgraph0")
    dec_graph.group_map_device["subgraph0"] = DeviceType.CPU
    dec_driver = GraphDriver(dec_graph)
    dec_driver.subgraphs[0].lower_to_top_level_ir()

    dec_subgraph_ir = str(dec_driver.subgraphs[0]._imported_module).replace(
        "@subgraph0", "@subgraph_decode"
    )
    dec_forward_ir = str(dec_driver.construct_main_graph(True)).replace(
        "@subgraph0", "@subgraph_decode"
    )

    # Remap decode_forward.mlir parameter subview offsets to share Prefill's
    # arg0.data / arg1.data without duplicating weights on disk or in RAM.
    dec_dtypes = sorted({param.dtype for param in dec_params}, key=str)
    mlir_type_names = {torch.float16: "f16", torch.float32: "f32"}
    dec_lines = dec_forward_ir.splitlines()

    def _row_major_strides(dims):
        strides = [1] * len(dims)
        for k in range(len(dims) - 2, -1, -1):
            strides[k] = strides[k + 1] * dims[k + 1]
        return strides

    def _strided_type(dims_str, ty, off):
        if off == 0:
            return f"memref<{dims_str}x{ty}>"
        dims = [int(d) for d in dims_str.split("x")]
        st = ", ".join(str(s) for s in _row_major_strides(dims))
        return f"memref<{dims_str}x{ty}, strided<[{st}], offset: {off}>>"

    for dec_arg_idx, dtype in enumerate(dec_dtypes):
        dtype_params = [param for param in dec_params if param.dtype == dtype]
        dec_total = sum(p.numel() for p in dtype_params)
        prefill_arg_idx = dtypes.index(dtype)
        prefill_total = prefill_totals[prefill_arg_idx]
        mlir_ty = mlir_type_names[dtype]
        offset_map = {}
        old_off = 0
        for param in dtype_params:
            key = (param.data_ptr(), param.numel())
            if key not in prefill_param_map:
                raise RuntimeError(
                    f"Decode param ({dtype}, numel={param.numel()}) not found in Prefill params"
                )
            p_arg, p_off = prefill_param_map[key]
            assert p_arg == prefill_arg_idx
            offset_map[(old_off, param.numel())] = p_off
            old_off += param.numel()

        # Track SSA values whose static offset changed so expand_shape/cast match.
        ssa_remap = {}
        new_lines = []
        subview_pat = re.compile(
            rf"^(\s*)(%\S+)\s*=\s*memref\.subview\s+%arg{dec_arg_idx}\[(\d+)\]\s+\[(\d+)\]\s+\[1\]\s*:\s*memref<{dec_total}x{mlir_ty}>\s+to\s+(.+)$"
        )
        expand_pat = re.compile(
            rf"^(\s*)(%\S+)\s*=\s*memref\.expand_shape\s+(%\S+)\s+(\[.+?\]\s+output_shape\s+\[.+?\])\s*:\s*(memref<.+?>)\s+into\s+memref<([\dx]+)x{mlir_ty}(?:,\s*strided<\[[^\]]+\],\s*offset:\s*\d+>)?>\s*$"
        )
        cast_pat = re.compile(
            r"^(\s*)(%\S+)\s*=\s*memref\.cast\s+(%\S+)\s*:\s*(memref<.+?>)\s+to\s+(.+)$"
        )

        for line in dec_lines:
            if f"%arg{dec_arg_idx}: memref<{dec_total}x{mlir_ty}>" in line:
                line = line.replace(
                    f"%arg{dec_arg_idx}: memref<{dec_total}x{mlir_ty}>",
                    f"%arg{dec_arg_idx}: memref<{prefill_total}x{mlir_ty}>",
                )
                new_lines.append(line)
                continue
            m = subview_pat.match(line)
            if m:
                indent, res, o_str, s_str, _ = m.groups()
                o_val, s_val = int(o_str), int(s_str)
                if (o_val, s_val) not in offset_map:
                    raise RuntimeError(
                        f"Unmapped decode subview offset={o_val} size={s_val} on %arg{dec_arg_idx}"
                    )
                new_o = offset_map[(o_val, s_val)]
                new_ty = _strided_type(str(s_val), mlir_ty, new_o)
                ssa_remap[res] = (new_o, new_ty)
                new_lines.append(
                    f"{indent}{res} = memref.subview %arg{dec_arg_idx}[{new_o}] [{s_val}] [1] "
                    f": memref<{prefill_total}x{mlir_ty}> to {new_ty}"
                )
                continue
            m = expand_pat.match(line)
            if m and m.group(3) in ssa_remap:
                indent, res, src_ssa, middle, _, out_dims = m.groups()
                new_o, src_ty = ssa_remap[src_ssa]
                out_ty = _strided_type(out_dims, mlir_ty, new_o)
                ssa_remap[res] = (new_o, out_ty)
                new_lines.append(
                    f"{indent}{res} = memref.expand_shape {src_ssa} {middle} "
                    f": {src_ty} into {out_ty}"
                )
                continue
            m = cast_pat.match(line)
            if m and m.group(3) in ssa_remap:
                indent, res, src_ssa, _, dst_ty = m.groups()
                _, src_ty = ssa_remap[src_ssa]
                new_lines.append(
                    f"{indent}{res} = memref.cast {src_ssa} : {src_ty} to {dst_ty}"
                )
                continue
            new_lines.append(line)
        dec_lines = new_lines

    dec_forward_ir = "\n".join(dec_lines) + "\n"

    with open(os.path.join(output_dir, "decode_subgraph0.mlir"), "w") as f:
        f.write(dec_subgraph_ir)
    with open(os.path.join(output_dir, "decode_forward.mlir"), "w") as f:
        f.write(dec_forward_ir)
    print(
        "[PaddleOCR-Import] Written decode_subgraph0.mlir and decode_forward.mlir "
        "(sharing Prefill arg0.data/arg1.data)."
    )

print(f"[PaddleOCR-Import] Done! {n_param_elems:,} total elements.")
