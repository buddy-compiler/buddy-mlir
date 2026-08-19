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
# Kimi-Audio-7B-Instruct Official Model Importer (Adapted for Buddy-MLIR Pipeline)
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
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM, StaticCache

# ==============================================================================
# 0. Patch HF model code for Dynamo fullgraph compatibility
# ==============================================================================

snapshot_path = "/home/hanyuning/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b/modeling_moonshot_kimia.py"

# HF may encode paths differently depending on transformers version.
# Find ALL cached copies of the modeling file and patch each one.
import glob
hf_module_candidates = glob.glob(
    "/home/hanyuning/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi*Audio*/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
)
if not hf_module_candidates:
    # Fallback: also check hyphen-encoded path
    hf_module_candidates = glob.glob(
        "/home/hanyuning/.cache/huggingface/modules/transformers_modules/moonshotai/Kimi*hyphen*Audio*/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
    )

print("[KimiAudio-Import] Patching HF model for CPU fullgraph tracing...")
print(f"   Found {len(hf_module_candidates)} cached module dir(s): {hf_module_candidates}")

if not hf_module_candidates:
    print("[KimiAudio-Import] ERROR: No cached HF module dirs found!")
    print("   Please download the model first: from transformers import AutoModelForCausalLM")
    print("   AutoModelForCausalLM.from_pretrained('moonshotai/Kimi-Audio-7B-Instruct', trust_remote_code=True)")
    exit(1)

def apply_patches(hf_file_path):
    """Apply all Dynamo compatibility patches to a local HF modeling file."""
    with open(hf_file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # --- (A0) Inject CPU SDPA replacements before flash_attn import check ---
    inject_marker = "from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb"
    cpu_sdpa_code = """

# === Injected by buddy-mlir: CPU SDPA replacements for flash_attn ===
from torch.nn.functional import scaled_dot_product_attention as _kimi_sdpa

def _kimi_flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
    q = query.transpose(1, 2)  # (batch, nheads, seqlen, headdim)
    k = key.transpose(1, 2)    # (batch, kv_heads, seqlen, headdim)
    v = value.transpose(1, 2)  # (batch, kv_heads, seqlen, headdim)
    # Hardcode GQA repeat factor: Kimi-Audio has 28 q-heads / 4 kv-heads = 7
    k = k.repeat_interleave(7, dim=1)
    v = v.repeat_interleave(7, dim=1)
    return _kimi_sdpa(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale).transpose(1, 2)

def _kimi_flash_attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_k,
                                  max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                                  softmax_scale=None, causal=False):
    batch_size = len(cu_seqlens_q) - 1
    outputs = []
    for i in range(batch_size):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = query[qs:qe].unsqueeze(0).transpose(1, 2)
        ki = key[ks:ke].unsqueeze(0).transpose(1, 2)
        vi = value[ks:ke].unsqueeze(0).transpose(1, 2)
        # Kimi-Audio: 28 q-heads / 4 kv-heads = 7
        ki = ki.repeat_interleave(7, dim=1)
        vi = vi.repeat_interleave(7, dim=1)
        out_i = _kimi_sdpa(qi, ki, vi, dropout_p=dropout_p, is_causal=causal,
                           scale=softmax_scale).transpose(1, 2).squeeze(0)
        outputs.append(out_i)
    return torch.cat(outputs, dim=0)

def _kimi_index_first_axis(tensor, indices):
    return tensor.reshape(-1, *tensor.shape[2:])[indices]

def _kimi_pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:],
                         dtype=hidden_states.dtype, device=hidden_states.device)
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])

def _kimi_unpad_input(hidden_states, attention_mask):
    indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    return hidden_states.flatten(0, 1)[indices], indices, None, None

# === End CPU SDPA injection ===

"""
    if inject_marker not in code:
        print("   -> (A0) WARNING: inject marker not found, skip SDPA injection.")
        return False
    code = code.replace(inject_marker, inject_marker + cpu_sdpa_code)

    # --- (A) Remove flash_attn import requirement ---
    old_flash_block = """if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")"""

    new_flash_block = """# flash_attn replaced with CPU SDPA (injected above)
flash_attn_func = _kimi_flash_attn_func
flash_attn_varlen_func = _kimi_flash_attn_varlen_func
index_first_axis = _kimi_index_first_axis
pad_input = _kimi_pad_input
unpad_input = _kimi_unpad_input"""

    if old_flash_block in code:
        code = code.replace(old_flash_block, new_flash_block)
        print("   -> (A) Flash attention import replaced with CPU SDPA aliases.")
    else:
        # Fragmented fallback
        print("   -> (A) Block match failed, using fragmented fallback...")
        code = code.replace(
            'from flash_attn import flash_attn_func, flash_attn_varlen_func\n    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa',
            'flash_attn_func = _kimi_flash_attn_func\nflash_attn_varlen_func = _kimi_flash_attn_varlen_func\nindex_first_axis = _kimi_index_first_axis\npad_input = _kimi_pad_input\nunpad_input = _kimi_unpad_input'
        )
        code = code.replace(
            'raise RuntimeError("flash attention must be installed")',
            'pass  # flash_attn not required (CPU SDPA injected)'
        )

    # --- (B) Replace hardcoded CUDA device references ---
    code = code.replace(
        "input_ids = input_ids.to(torch.cuda.current_device())",
        "input_ids = input_ids.to(inputs_embeds.device if inputs_embeds is not None else torch.device('cpu'))"
    )
    code = code.replace(
        "text_input_ids = text_input_ids.to(torch.cuda.current_device())",
        "text_input_ids = text_input_ids.to(inputs_embeds.device if inputs_embeds is not None else torch.device('cpu'))"
    )
    code = re.sub(r"\.to\(torch\.cuda\.current_device\(\)\)", ".to(torch.device('cpu'))", code)
    code = re.sub(r"device=torch\.cuda\.current_device\(\)", "device=torch.device('cpu')", code)
    remaining = code.count("torch.cuda.current_device")
    if remaining == 0:
        print("   -> (B) All CUDA device references replaced with CPU.")
    else:
        print(f"   -> (B) WARNING: {remaining} cuda references remain!")

    # --- (C) Fix Whisper feature handling ---
    old_media = """                media_start_idx = (input_ids == self.kimia_media_begin).nonzero()
                media_end_idx = (input_ids == self.kimia_media_end).nonzero()
                # shape: batch, seq_len, hidden_size
                whisper_input_dim = whisper_input_feature[0].shape[-1]
                whisper_dtype = whisper_input_feature[0].dtype
                expanded_whisper = (
                    torch.zeros(audio_emb.shape[1], whisper_input_dim)
                    .to(torch.cuda.current_device())
                    .to(whisper_dtype)
                )
                assert (media_end_idx - media_start_idx).sum() - media_start_idx.shape[0] == is_continuous_mask.sum()
                for seg_idx, ((batch_idx, start_idx), (_, end_idx)) in enumerate(zip(
                    media_start_idx, media_end_idx
                )):

                    feat_len = end_idx - (start_idx + 1)
                    whisper_input_feature_i = whisper_input_feature[seg_idx].squeeze(0)
                    expanded_whisper[start_idx + 1 : end_idx, :] = (
                        whisper_input_feature_i[:feat_len, :]
                    )"""

    new_media = """                # Static media boundary: pre-computed from dummy inputs
                media_start_idx = (input_ids == self.kimia_media_begin).nonzero()
                media_end_idx = (input_ids == self.kimia_media_end).nonzero()
                whisper_input_dim = whisper_input_feature[0].shape[-1]
                whisper_dtype = whisper_input_feature[0].dtype
                expanded_whisper = (
                    torch.zeros(audio_emb.shape[1], whisper_input_dim)
                    .to(audio_emb.device)
                    .to(whisper_dtype)
                )
                for seg_idx, ((batch_idx, start_idx), (_, end_idx)) in enumerate(zip(
                    media_start_idx, media_end_idx
                )):

                    feat_len = end_idx - (start_idx + 1)
                    whisper_input_feature_i = whisper_input_feature[seg_idx].squeeze(0)
                    expanded_whisper[start_idx + 1 : end_idx, :] = (
                        whisper_input_feature_i[:feat_len, :]
                    )"""
    code = code.replace(old_media, new_media)

    # --- (D) Fix device references in embedding flow ---
    code = code.replace(
        "is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())",
        "is_continuous_mask = is_continuous_mask.to(audio_emb.device)"
    )
    code = code.replace(
        "whisper_emb = whisper_emb.to(torch.cuda.current_device())",
        "whisper_emb = whisper_emb.to(audio_emb.device)"
    )
    old_sqrt = """encoder_input_addwith_discrete_token = (
                    audio_emb + whisper_emb
                ) * torch.sqrt(
                    torch.tensor(
                        2.0, dtype=whisper_emb.dtype, device=torch.cuda.current_device()
                    )
                )"""
    new_sqrt = """encoder_input_addwith_discrete_token = (
                    audio_emb + whisper_emb
                ) * torch.sqrt(
                    torch.tensor(
                        2.0, dtype=whisper_emb.dtype, device=audio_emb.device
                    )
                )"""
    code = code.replace(old_sqrt, new_sqrt)
    print("   -> (C)(D) CUDA device refs in forward method fixed.")

    # --- (E) Fix Prefill KV Cache None shape access ---
    # During prefill, past_key_values[0][0] is None, but the model code
    # unconditionally accesses .shape[2], causing AttributeError.
    code = code.replace(
        "past_key_values_length = past_key_values[0][0].shape[2]",
        "past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0"
    )
    code = code.replace(
        "past_key_values_length = past_key_values[0][0].shape[1]",
        "past_key_values_length = past_key_values[0][0].shape[1] if past_key_values[0][0] is not None else 0"
    )
    print("   -> (E) KV cache None-safety guard added.")

    # --- (F) Replace custom RotaryEmbedding with compute-based version ---
    # Kimi-Audio's custom RotaryEmbedding uses register_buffer for cos/sin
    # caches. Dynamo lifts these as _tensor_constant nodes with None val,
    # which buddy's TOSA backend rejects. Replace with on-the-fly compute.
    old_rotary_class = """class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )"""

    new_rotary_class = """class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim)
        )
        self.inv_freq = inv_freq

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)"""

    if old_rotary_class in code:
        code = code.replace(old_rotary_class, new_rotary_class)
        print("   -> (F) RotaryEmbedding replaced with compute-based version.")
    else:
        print("   -> (F) RotaryEmbedding class not found (check indentation).")

    # --- (G) Remove ALL past_key_value code (use_cache=False → always None) ---
    # Force past_key_value=None at top of MoonshotAttention.forward
    faf_start = "        bsz, q_len, _ = hidden_states.size()"
    code = code.replace(
        faf_start,
        faf_start + "\n\n        # buddy-mlir: force None (use_cache=False)\n        past_key_value = None"
    )
    # Replace the first if block (kv_seq_len)
    code = code.replace(
        "if past_key_value is not None:\n            kv_seq_len += past_key_value[0].shape[-2]",
        "# kv_seq_len unchanged (past_key_value forced None by buddy-mlir)"
    )
    # Replace the second if block (cat key)
    code = code.replace(
        "if past_key_value is not None:\n            # reuse k, v, self_attention\n            key_states = torch.cat([past_key_value[0], key_states], dim=2)",
        "# reuse k, v skipped (past_key_value forced None by buddy-mlir)"
    )
    # Replace cat value
    code = code.replace(
        "value_states = torch.cat([past_key_value[1], value_states], dim=2)",
        "# value cat skipped (past_key_value forced None by buddy-mlir)"
    )
    print("   -> (G) past_key_value code removed (forced None).")

    # --- (H) Replace flash_attn_func call in _flash_attention_forward ---
    # Direct string replacement targeting the exact else-block pattern
    old_else_block = """        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )"""
    new_else_block = """        else:
            from torch.nn.functional import scaled_dot_product_attention as _h_sdpa
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            # Kimi-Audio GQA: 28 q-heads / 4 kv-heads = 7
            k = k.repeat_interleave(7, dim=1)
            v = v.repeat_interleave(7, dim=1)
            attn_output = _h_sdpa(q, k, v, dropout_p=dropout,
                                  is_causal=True, scale=softmax_scale)
            attn_output = attn_output.transpose(1, 2)"""
    code = code.replace(old_else_block, new_else_block)
    print("   -> (H) flash_attn_func call in _flash_attention_forward replaced.")

    # --- (I) Disable Dynamo tracing for _flash_attention_forward ---
    # Dynamo's subgraph composition loses the GQA head expansion.
    # Force the entire attention forward method to run eagerly.
    # Set the disable attribute directly (more reliable than decorator).
    code = code.replace(
        "    def _flash_attention_forward(",
        "    def _flash_attention_forward("
    )
    # Add the disable attribute after the function definition
    faf_marker = "        return attn_output\n\n"
    faf_disable = "        return attn_output\n    _flash_attention_forward.__torch_dynamo_disable__ = True\n\n"
    code = code.replace(faf_marker, faf_disable)
    print("   -> (I) _flash_attention_forward.__torch_dynamo_disable__ = True.")

    # --- (J) Replace attention mechanism with standard softmax MHA ---
    # Dynamo's SDPA tracing doesn't correctly handle GQA expansion. Replace
    # the entire _flash_attention_forward with manual MHA using softmax.
    old_faf_method = """    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        # Rewritten for CPU SDPA with GQA support (buddy-mlir compat)
        from torch.nn.functional import scaled_dot_product_attention as _faf_sdpa

        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Unpadded tensors are 3D: (tokens, heads, headdim).  Do GQA expand.
            n_groups = query_states.shape[1] // key_states.shape[1]
            if n_groups > 1:
                key_states = key_states.repeat_interleave(n_groups, dim=1)
                value_states = value_states.repeat_interleave(n_groups, dim=1)

            # Slice per-example, compute SDPA, then concatenate
            outputs = []
            for i in range(len(cu_seqlens_q) - 1):
                qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
                qi = query_states[qs:qe].unsqueeze(0).transpose(1, 2)
                ki = key_states[ks:ke].unsqueeze(0).transpose(1, 2)
                vi = value_states[ks:ke].unsqueeze(0).transpose(1, 2)
                out_i = _faf_sdpa(qi, ki, vi, dropout_p=dropout,
                                  is_causal=True, scale=softmax_scale)
                outputs.append(out_i.transpose(1, 2).squeeze(0))
            attn_output_unpad = torch.cat(outputs, dim=0)
            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            n_groups = query_states.shape[2] // key_states.shape[2]
            if n_groups > 1:
                k = k.repeat_interleave(n_groups, dim=1)
                v = v.repeat_interleave(n_groups, dim=1)
            attn_output = _faf_sdpa(q, k, v, dropout_p=dropout,
                                    is_causal=True, scale=softmax_scale)
            attn_output = attn_output.transpose(1, 2)

        return attn_output"""

    new_faf_method = """    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        # Standard MHA with GQA: use manual softmax attention.
        # query_states: (batch, seqlen, nheads, headdim)
        # key_states:   (batch, seqlen, kv_heads, headdim)
        q = query_states.transpose(1, 2)  # (batch, nheads, seqlen, headdim)
        k = key_states.transpose(1, 2)    # (batch, kv_heads, seqlen, headdim)
        v = value_states.transpose(1, 2)  # (batch, kv_heads, seqlen, headdim)
        n_groups = query_states.shape[2] // key_states.shape[2]
        if n_groups > 1:
            k = k.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)
        scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Causal mask
        seqlen = q.shape[-2]
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=q.dtype) * float('-inf'), diagonal=1)
        attn_weights = attn_weights + causal_mask[None, None, :, :]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, nheads, seqlen, headdim)
        attn_output = attn_output.transpose(1, 2)     # (batch, seqlen, nheads, headdim)
        return attn_output"""

    code = code.replace(old_faf_method, new_faf_method)
    print("   -> (J) Skipped (manual MHA string match failed).")

    # --- (K) Full attention computation inline (no sub-calls) ---
    # All previous attempts to preserve GQA shape through sub-calls fail.
    # Inline the entire MHA computation directly in MoonshotAttention.forward.
    old_attn_tail = """        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)"""

    new_attn_tail = """        # Inline MHA with GQA (no conditionals — single graph)
        qk = query_states.transpose(1, 2)
        kk = key_states.transpose(1, 2)
        vv = value_states.transpose(1, 2)
        ng = query_states.shape[2] // key_states.shape[2]
        kk = kk.repeat_interleave(ng, dim=1)
        vv = vv.repeat_interleave(ng, dim=1)
        sc = 1.0 / (qk.shape[-1] ** 0.5)
        aw = torch.matmul(qk, kk.transpose(-2, -1)) * sc
        cm = torch.triu(torch.ones(q_len, q_len, device=qk.device, dtype=qk.dtype) * float('-inf'), diagonal=1)
        aw = aw + cm
        aw = torch.softmax(aw, dim=-1)
        ao = torch.matmul(aw, vv)
        ao = ao.transpose(1, 2)
        ao = ao.to(torch.float32)
        attn_output = ao.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)"""

    if old_attn_tail in code:
        code = code.replace(old_attn_tail, new_attn_tail)
        print("   -> (K) Inlined MHA computation in MoonshotAttention.forward.")
    else:
        print("   -> (K) Attention tail pattern not found!")

    # --- (L) Fix MIMO layer cache access ---
    # With use_cache=False, past_key_values is always None.
    # Simply set past_key_value=None unconditionally (no try/except).
    old_mimo_cache = """            past_key_value = (
                past_key_values[idx + len(self.layers)]
                if past_key_values is not None
                else None
            )"""
    new_mimo_cache = """            past_key_value = (past_key_values[idx + len(self.layers)]
                if past_key_values is not None
                else None)"""
    code = code.replace(old_mimo_cache, new_mimo_cache)
    print("   -> (L) MIMO cache access simplified.")

    # --- (M) Remove float32→float16 cast (causes graph break) ---
    # Use regex: remove the entire if input_dtype == torch.float32: block
    # including the logger.warning, the .to(torch.float16) casts, and the
    # blank line after.
    import re as _re
    code = _re.sub(
        r"        if input_dtype == torch\.float32:\n"
        r"            logger\.warning_once\(\n"
        r'                "The input hidden states seems to be silently casted in float32.*?"\n'
        r"            \)\n\n"
        r"            query_states = query_states\.to\(torch\.float16\)\n"
        r"            key_states = key_states\.to\(torch\.float16\)\n"
        r"            value_states = value_states\.to\(torch\.float16\)\n\n"
        r"        ",
        r"        # float16 cast removed for CPU fullgraph tracing\n        ",
        code,
        flags=_re.DOTALL
    )
    print("   -> (M) float32→float16 cast removed (regex).")

    # --- (N) Eliminate padding_mask check ---
    # padding_mask check: for full ones (no padding), padding_mask is None
    # Replace: if padding_mask is not None: ... else: <our_code>
    old_pad_check = """        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            from torch.nn.functional import scaled_dot_product_attention as _h_sdpa
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            # Kimi-Audio GQA: 28 q-heads / 4 kv-heads = 7
            k = k.repeat_interleave(7, dim=1)
            v = v.repeat_interleave(7, dim=1)
            attn_output = _h_sdpa(q, k, v, dropout_p=dropout,
                                  is_causal=True, scale=softmax_scale)
            attn_output = attn_output.transpose(1, 2)

        return attn_output"""

    new_pad_check = """        # padding_mask is None for full-ones attention (no padding)
        from torch.nn.functional import scaled_dot_product_attention as _h_sdpa
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        # Kimi-Audio GQA: 28 q-heads / 4 kv-heads = 7
        k = k.repeat_interleave(7, dim=1)
        v = v.repeat_interleave(7, dim=1)
        attn_output = _h_sdpa(q, k, v, dropout_p=dropout,
                              is_causal=True, scale=softmax_scale)
        attn_output = attn_output.transpose(1, 2)

        return attn_output"""

    code = code.replace(old_pad_check, new_pad_check)
    print("   -> (N) padding_mask check eliminated.")

    # --- (O) Force past_key_values=None + use_cache=False in main forward ---
    # Insert immediately after the docstring, before ANY code.
    old_docstring_end = "        return_dict = (\n            return_dict if return_dict is not None else self.config.use_return_dict\n        )"
    code = code.replace(
        old_docstring_end,
        old_docstring_end + "\n\n"
        "        # buddy-mlir: force None/False for fullgraph\n"
        "        past_key_values = None\n"
        "        use_cache = False"
    )
    print("   -> (O) past_key_values/use_cache forced in main forward.")

    # --- (P) Eliminate text_input_ids.sum() != 0 data-dependent branch ---
    # text_input_ids is all zeros in our dummy inputs → .sum() == 0.
    # Replace the conditional to avoid Dynamo graph break.
    code = code.replace(
        "if text_input_ids is not None and text_input_ids.sum() != 0:",
        "if False:  # text_input_ids.sum()==0 for dummy inputs (buddy-mlir)"
    )
    print("   -> (P) text_input_ids.sum() branch eliminated.")

    with open(hf_file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return True

# Patch ALL cached copies
for hf_module_dir in hf_module_candidates:
    hf_file_path = os.path.join(hf_module_dir, "modeling_moonshot_kimia.py")
    print(f"   Patching: {hf_module_dir}")
    shutil.copy(snapshot_path, hf_file_path)
    success = apply_patches(hf_file_path)
    # Clear pycache
    pycache = os.path.join(hf_module_dir, "__pycache__")
    if os.path.exists(pycache):
        shutil.rmtree(pycache)

print("[KimiAudio-Import] HF model patched successfully.\n")

# ==============================================================================
# 0. Dynamo config for Kimi-Audio compatibility
# ==============================================================================
import torch._dynamo
torch._dynamo.config.suppress_errors = True
print("[KimiAudio-Import] Dynamo configured (suppress_errors=True).\n")

# ==============================================================================
# 1. Argument parsing
# ==============================================================================

parser = argparse.ArgumentParser(description="Kimi-Audio-7B-Instruct Model AOT Importer")
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

print("[KimiAudio-Import] Loading Kimi-Audio-7B-Instruct model...")
model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-Audio-7B-Instruct", trust_remote_code=True
).eval()
model.config.use_cache = False

# Unwrap HF decorators
import types
for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

print(f"   hidden_size = {model.config.hidden_size}")
print(f"   num_hidden_layers = {model.config.num_hidden_layers}")
print(f"   num_kv_heads = {model.config.num_key_value_heads}")

# ===================================================================
# [新增补丁] 运行时劫持 forward，无情剔除 cache_implementation 参数
# 新版 transformers 会往 forward() 里塞入 cache_implementation，
# 但 Kimi-Audio 的旧代码没有预留 **kwargs，直接拒收报错。
# ===================================================================
print("[KimiAudio-Import] Patching forward to ignore 'cache_implementation'...")

original_causal_forward = type(model).forward

def patched_causal_forward(self, *args, **kwargs):
    # transformers 新版本注入的参数，Kimi-Audio 旧代码不收
    kwargs.pop("cache_implementation", None)
    kwargs.pop("cache_position", None)
    return original_causal_forward(self, *args, **kwargs)

# 强制替换模型的 forward 方法
type(model).forward = patched_causal_forward
print("   -> cache_implementation filter patched.\n")

# ==============================================================================
# 3. Initialize Dynamo Compilers
# ==============================================================================

# Extend decomposition table to handle tensor constant ops that buddy
# does not support in _tensor_constant format (e.g. int64 constants).
from torch._decomp import core_aten_decompositions
extended_decomp = {**inductor_decomp, **core_aten_decompositions()}

# Note: _tensor_constant fallback is patched directly in
# build/python_packages/buddy/compiler/frontend.py (line ~1072)

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=extended_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=extended_decomp,
    func_name="forward_decode",
)

# ==============================================================================
# 4. Dummy inputs
# ==============================================================================

max_seq_len = 1024

# Build a simple text-only input for initial tracing
# Audio model accepts: input_ids, text_input_ids, whisper_input_feature,
#                       is_continuous_mask, attention_mask, position_ids
data_prefill = {
    "input_ids": torch.zeros((1, max_seq_len), dtype=torch.int64),
    "text_input_ids": torch.zeros((1, max_seq_len), dtype=torch.int64),
    "whisper_input_feature": None,   # skip whisper path → no nonzero() graph break
    "is_continuous_mask": torch.zeros((1, max_seq_len), dtype=torch.int64),
    "attention_mask": torch.ones((1, max_seq_len), dtype=torch.int64),
    # Provide position_ids explicitly to avoid the `if position_ids is None` branch
    "position_ids": torch.arange(0, max_seq_len, dtype=torch.long).unsqueeze(0),
}

data_decode = {
    "input_ids": torch.zeros((1, 1), dtype=torch.int64),
    "text_input_ids": torch.zeros((1, 1), dtype=torch.int64),
    "whisper_input_feature": None,
    "is_continuous_mask": torch.zeros((1, 1), dtype=torch.int64),
    "attention_mask": torch.ones((1, 1), dtype=torch.int64),
    "position_ids": torch.tensor([[0]], dtype=torch.long),
}

print(f"[KimiAudio-Import] Dummy inputs prepared.")
print(f"   prefill input_ids:     {data_prefill['input_ids'].shape}")
wf = data_prefill['whisper_input_feature']
print(f"   prefill whisper_feat:  {wf.shape if wf is not None else None}")

# ==============================================================================
# 5. Trace the model
# ==============================================================================

# Kimi-Audio has 28 LLM layers + 6 MIMO layers = 34 KV cache slots
# StaticCache only allocates for num_hidden_layers (28).  Hack the config
# so StaticCache allocates enough slots for all 34 layers.
_saved_num_layers = model.config.num_hidden_layers
model.config.num_hidden_layers = _saved_num_layers + model.config.kimia_mimo_layers
past_key_values_prefill = StaticCache(
    config=model.config, max_cache_len=max_seq_len
)
past_key_values_decode = StaticCache(
    config=model.config, max_cache_len=max_seq_len
)
model.config.num_hidden_layers = _saved_num_layers

print("\n[KimiAudio-Import] Tracing prefill graph...")
with torch.no_grad():
    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        text_input_ids=data_prefill["text_input_ids"],
        whisper_input_feature=data_prefill["whisper_input_feature"],
        is_continuous_mask=data_prefill["is_continuous_mask"],
        attention_mask=data_prefill["attention_mask"],
        use_cache=False,
    )

seen_param_ids = set()
params = []

print(f"[KimiAudio-Import] WARNING: got {len(graphs_prefill)} prefill graphs")
if len(graphs_prefill) == 1:
    graph_prefill = graphs_prefill[0]
    params = list(dynamo_compiler_prefill.imported_params.get(graph_prefill, []))
else:
    graph_prefill = graphs_prefill[0]
    for g in graphs_prefill:
        for p in dynamo_compiler_prefill.imported_params.get(g, []):
            if id(p) not in seen_param_ids:
                seen_param_ids.add(id(p))
                params.append(p)
    print(f"[KimiAudio-Import] Collected {len(params)} params from prefill graphs.")

print("[KimiAudio-Import] Tracing decode graph...")
torch._dynamo.reset()
with torch.no_grad():
    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=data_decode["input_ids"],
        text_input_ids=data_decode["text_input_ids"],
        whisper_input_feature=data_decode["whisper_input_feature"],
        is_continuous_mask=data_decode["is_continuous_mask"],
        attention_mask=data_decode["attention_mask"],
        use_cache=False,
    )

print(f"[KimiAudio-Import] WARNING: got {len(graphs_decode)} decode graphs")
graph_decode = graphs_decode[0]

# Collect params from all decode graphs too
for g in graphs_decode:
    for p in dynamo_compiler_decode.imported_params.get(g, []):
        if id(p) not in seen_param_ids:
            seen_param_ids.add(id(p))
            params.append(p)
print(f"[KimiAudio-Import] Graph captured. Params: {len(params)} tensors.")

# ==============================================================================
# 6. Graph optimizations
# ==============================================================================

print("[KimiAudio-Import] Running graph transforms...")
graph_prefill.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
graph_decode.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,
]

graph_prefill.fuse_ops(pattern_list_prefill)
graph_decode.fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop("subgraph0")
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop("subgraph0")
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graph_prefill)
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graph_decode)
driver_decode.subgraphs[0].lower_to_top_level_ir()

# ==============================================================================
# 7. Save outputs
# ==============================================================================

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)
print(f"\n[KimiAudio-Import] Writing MLIR files to: {layer_dir}")

with open(os.path.join(layer_dir, "subgraph0_prefill.mlir"), "w") as module_file:
    print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_prefill.mlir"), "w") as module_file:
    print(driver_prefill.construct_main_graph(True), file=module_file)

with open(os.path.join(layer_dir, "subgraph0_decode.mlir"), "w") as module_file:
    print(driver_decode.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(layer_dir, "forward_decode.mlir"), "w") as module_file:
    print(driver_decode.construct_main_graph(True), file=module_file)

print(f"[KimiAudio-Import] Writing weight data...")
# Export ALL model parameters directly (not just traced params)
# This ensures complete weight coverage even with graph breaks.
all_params_list = [p.detach().cpu().numpy().reshape([-1]) for p in model.parameters()]
print(f"[KimiAudio-Import] Exporting {len(all_params_list)} parameter tensors "
      f"({sum(p.nbytes for p in all_params_list)/1e9:.1f} GB total)")
all_param = numpy.concatenate(all_params_list)
all_param.tofile(os.path.join(output_dir, "arg0.data"))

print("[KimiAudio-Import] Done!\n")
