#!/usr/bin/env python3
# ===- qwen3_vl_codegen.py - Qwen3-VL import / pack / stage ---------------===//
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
# Codegen entry for the Qwen3-VL multimodal OCR package. Invoked by
# tools/buddy-codegen/import_model.py (when QWEN3_VL_KV_DECODE is set, the
# decoder path uses import-decoder-kv) and by CMake stage4 (`stage`).
#
# Subcommands:
#   import-vision      Import the pinned-grid vision encoder to MLIR.
#   import-decoder-rt  Legacy full-sequence decoder (no KV cache).
#   import-decoder-kv  Prefill + one-token decode graphs with GQA caches;
#                      pack decode linear weights for packed GEMV lowering.
#   preprocess         Emit per-query tensors for the runner (stage helper).
#   stage              Assemble the runnable package and pack qwen3_vl.rax.
#
# Run in the buddy Python environment:
#   conda activate buddy
#   export BUDDY_MLIR_BUILD_DIR=$PWD/build
#   export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
#   export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
#
# ===----------------------------------------------------------------------===//
"""Qwen3-VL codegen utilities.

Subcommands:
  import-vision      Import the pinned-grid vision encoder to MLIR.
  import-decoder-rt  Import the runtime-position decoder to MLIR.
  import-decoder-kv  Import prefill/decode graphs with fixed-length KV caches.
  preprocess         Emit per-query tensors for the runner.
  stage              Assemble the runnable package and pack qwen3_vl.rax.

Run in the buddy Python environment:
  conda activate buddy
  export BUDDY_MLIR_BUILD_DIR=$PWD/build
  export LLVM_MLIR_BUILD_DIR=$PWD/llvm/build
  export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL_DIR = os.environ.get("QWEN3_VL_MODEL_PATH")
ARTIFACT_DIR = os.path.abspath(
    os.environ.get(
        "QWEN3_VL_OUT_DIR",
        os.path.join(REPO, "build", "models", "qwen3_vl", "artifacts"),
    )
)
PKG_DIR = os.path.abspath(
    os.environ.get(
        "QWEN3_VL_PKG", os.path.join(REPO, "build", "models", "qwen3_vl")
    )
)
VISION_DIR = os.path.join(ARTIFACT_DIR, "vision")
DECODER_DIR = os.path.join(ARTIFACT_DIR, "decoder_rt")
TEST_IMAGE = os.path.join(REPO, "models", "qwen3_vl", "test_text.png")
PROMPT = "Read all the text in the image."

IMAGE_TOKEN_ID = 151655
MAX_SEQ_LEN = int(os.environ.get("QWEN3_VL_MAXLEN", "160"))
VOCAB_SIZE = 151936
CANON_WH = (
    448,
    224,
)  # W,H -> grid [1,14,28], matching the compiled vision graph.
PROCESSOR_ARCHIVE = "qwen3_vl_processor.tar"
PROCESSOR_DIRNAME = "qwen3_vl_processor"
PROCESSOR_EXCLUDE_SUFFIXES = (
    ".bin",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
)


def rmsnorm(x, w, eps=1e-6):
    # Match Qwen3VLTextRMSNorm: variance is accumulated in fp32, then
    # the result is cast back to the activation dtype.
    input_dtype = x.dtype
    x32 = x.float()
    v = x32.pow(2).mean(-1, keepdim=True)
    y = w.float() * (x32 * torch.rsqrt(v + eps))
    return y.to(dtype=input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Compiled shims and the runner speak raw IEEE fp16. Import, weights, and
# the bundled positional tables must use the same dtype.
COMPUTE_DTYPE = torch.float16


def load_processor_and_model(dtype=COMPUTE_DTYPE, eager_attn=True):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not MODEL_DIR:
        raise RuntimeError(
            "QWEN3_VL_MODEL_PATH is required. Build through "
            "tools/buddy-codegen/build_model.py --local-model /path/to/snapshot."
        )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    kwargs = {"dtype": dtype}
    if eager_attn:
        kwargs["attn_implementation"] = "eager"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, **kwargs
    ).eval()
    return processor, model


def write_f16(value, path):
    if torch.is_tensor(value):
        array = (
            value.detach().to(dtype=torch.float16).contiguous().cpu().numpy()
        )
    else:
        array = np.ascontiguousarray(value, dtype=np.float16)
    array.tofile(path)


def load_processor_and_config():
    from transformers import AutoConfig, AutoProcessor

    model_dir = os.environ.get("QWEN3_VL_MODEL_PATH") or MODEL_DIR
    if not model_dir:
        raise RuntimeError(
            "QWEN3_VL_MODEL_PATH is required or a packaged "
            f"{PROCESSOR_DIRNAME} directory must be present."
        )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    return processor, config


def encode_image_prompt(processor, image_path, prompt):
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize(CANON_WH)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def vision_position_ids(start_pos, grid_thw, spatial_merge_size):
    t, h, w = [int(x) for x in grid_thw]
    llm_t = t
    llm_h = h // spatial_merge_size
    llm_w = w // spatial_merge_size
    image_seq_length = llm_t * llm_h * llm_w
    position_width = torch.arange(start_pos, start_pos + llm_w).repeat(
        llm_h * llm_t
    )
    position_height = torch.arange(
        start_pos, start_pos + llm_h
    ).repeat_interleave(llm_w * llm_t)
    position_temporal = torch.full(
        (image_seq_length,), start_pos, dtype=torch.long
    )
    return torch.stack(
        [position_temporal, position_height, position_width], dim=0
    )


def compute_3d_position_ids(
    config, input_ids, image_grid_thw, mm_token_type_ids
):
    spatial_merge_size = config.vision_config.spatial_merge_size
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_iter = iter(image_grid_thw)

    for batch_idx, _ in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        groups = []
        for key, group in itertools.groupby(
            enumerate(input_token_type.tolist()), lambda x: x[1]
        ):
            group = list(group)
            groups.append((key, group[0][0], group[-1][0] + 1))

        current_pos = 0
        pos_chunks = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                pos_chunks.append(
                    torch.arange(text_len, device=input_ids.device)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_pos
                )
                current_pos += text_len
            elif modality_type == 1:
                grid_thw = next(image_iter)
                pos_chunks.append(
                    vision_position_ids(
                        current_pos, grid_thw, spatial_merge_size
                    )
                )
                current_pos += (
                    max(int(grid_thw[1]), int(grid_thw[2]))
                    // spatial_merge_size
                )
            else:
                raise RuntimeError(
                    "video inputs are not supported by qwen3_vl OCR"
                )
        position_ids[:, batch_idx] = torch.cat(pos_chunks, dim=1).reshape(3, -1)
    return position_ids


def text_rotary(config, seq_len, hidden, rope_pos_n):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    rope = Qwen3VLTextRotaryEmbedding(config.text_config)
    cos, sin = rope(torch.zeros(1, seq_len, hidden), rope_pos_n)
    return cos, sin


def capture_decoder_golden():
    processor, model = load_processor_and_model(COMPUTE_DTYPE, eager_attn=True)
    inputs = encode_image_prompt(processor, TEST_IMAGE, PROMPT)

    grab = {}
    lm = model.model.language_model
    orig = lm.forward

    def hook(*a, **kw):
        grab["inputs_embeds"] = kw["inputs_embeds"].detach()
        grab["position_ids"] = kw["position_ids"].detach()
        grab["visual_pos_masks"] = kw["visual_pos_masks"].detach()
        grab["deepstack"] = [d.detach() for d in kw["deepstack_visual_embeds"]]
        return orig(*a, **kw)

    lm.forward = hook
    with torch.no_grad():
        out = model(**inputs)
    lm.forward = orig
    grab["logits"] = out.logits.detach()
    grab["input_ids"] = inputs["input_ids"].detach()
    grab["model"] = model
    grab["lm"] = lm
    return grab


class VisionTrace(nn.Module):
    """Trace-friendly Qwen3-VL vision encoder for a single, fixed-grid image."""

    def __init__(self, vm, pos_embeds, cos, sin):
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )

        self._apply_rope = apply_rotary_pos_emb_vision
        self.blocks = vm.blocks
        self.merger = vm.merger
        self.deepstack_merger_list = vm.deepstack_merger_list
        self.deepstack_visual_indexes = list(vm.deepstack_visual_indexes)
        self.num_heads = vm.blocks[0].attn.num_heads
        self.scaling = vm.blocks[0].attn.scaling
        w = vm.patch_embed.proj.weight
        self.register_buffer("pe_w", w.reshape(w.shape[0], -1).clone())
        self.register_buffer("pe_b", vm.patch_embed.proj.bias.clone())
        self.register_buffer("pos_embeds", pos_embeds.clone())
        self.register_buffer("cos", cos.clone())
        self.register_buffer("sin", sin.clone())

    def _attn(self, blk, h):
        seq_len = h.shape[0]
        attn = blk.attn
        q, k, v = (
            attn.qkv(h)
            .reshape(seq_len, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = self._apply_rope(q, k, self.cos, self.sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        aw = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        out = torch.matmul(aw, v).transpose(1, 2).reshape(seq_len, -1)
        return attn.proj(out)

    def forward(self, pixel_values):
        h = pixel_values @ self.pe_w.t() + self.pe_b
        h = h + self.pos_embeds
        deepstack = []
        for i, blk in enumerate(self.blocks):
            h = h + self._attn(blk, blk.norm1(h))
            h = h + blk.mlp(blk.norm2(h))
            if i in self.deepstack_visual_indexes:
                j = self.deepstack_visual_indexes.index(i)
                deepstack.append(self.deepstack_merger_list[j](h))
        pooled = self.merger(h)
        return (pooled, *deepstack)


class DecoderTraceRT(nn.Module):
    """Qwen3-VL decoder with cos/sin/cmask as runtime forward inputs."""

    def __init__(self, lm, lm_head_w, deepstack_layers):
        super().__init__()
        self.layers = lm.layers
        self.norm = lm.norm
        self.n_heads = lm.config.num_attention_heads
        self.n_kv = lm.config.num_key_value_heads
        self.head_dim = lm.config.head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = lm.config.rms_norm_eps
        self.deepstack_layers = deepstack_layers
        self.register_buffer("lm_head_w", lm_head_w.clone())

    def _attn(self, attn, h, cos, sin, cmask):
        batch, seq_len, _ = h.shape
        q = rmsnorm(
            attn.q_proj(h).view(batch, seq_len, self.n_heads, self.head_dim),
            attn.q_norm.weight,
            self.eps,
        ).transpose(1, 2)
        k = rmsnorm(
            attn.k_proj(h).view(batch, seq_len, self.n_kv, self.head_dim),
            attn.k_norm.weight,
            self.eps,
        ).transpose(1, 2)
        v = (
            attn.v_proj(h)
            .view(batch, seq_len, self.n_kv, self.head_dim)
            .transpose(1, 2)
        )
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        aw = torch.matmul(q, k.transpose(2, 3)) * self.scaling + cmask
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        o = torch.matmul(aw, v).transpose(1, 2).reshape(batch, seq_len, -1)
        return attn.o_proj(o)

    def forward(self, inputs_embeds, cos, sin, cmask, ds0, ds1, ds2):
        ds = [ds0, ds1, ds2]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            residual = h
            h = self._attn(
                layer.self_attn,
                rmsnorm(h, layer.input_layernorm.weight, self.eps),
                c,
                s,
                cmask,
            )
            h = residual + h
            residual = h
            mlp = layer.mlp
            pn = rmsnorm(h, layer.post_attention_layernorm.weight, self.eps)
            h = residual + mlp.down_proj(
                torch.nn.functional.silu(mlp.gate_proj(pn)) * mlp.up_proj(pn)
            )
            if i < self.deepstack_layers:
                h = h + ds[i]
        h = rmsnorm(h, self.norm.weight, self.eps)
        return h @ self.lm_head_w.t()


class DecoderTracePrefillKV(DecoderTraceRT):
    """Prefill that materializes fixed-length GQA K/V caches (pre-repeat).

    Unlike the legacy full-forward decoder, each attention block returns
    (output, K, V). After all layers we interleave K/V so the MLIR multi-result
    ABI is logits + kv0..kv55 (28 layers × 2). Cache tensors keep the native
    GQA shape [1, n_kv, N, head_dim] before head repeat — decode will
    repeat_interleave when attending.
    """

    def _attn_prefill(self, attn, h, cos, sin, cmask):
        batch, seq_len, _ = h.shape
        q = rmsnorm(
            attn.q_proj(h).view(batch, seq_len, self.n_heads, self.head_dim),
            attn.q_norm.weight,
            self.eps,
        ).transpose(1, 2)
        k = rmsnorm(
            attn.k_proj(h).view(batch, seq_len, self.n_kv, self.head_dim),
            attn.k_norm.weight,
            self.eps,
        ).transpose(1, 2)
        v = (
            attn.v_proj(h)
            .view(batch, seq_len, self.n_kv, self.head_dim)
            .transpose(1, 2)
        )
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.n_heads // self.n_kv
        aw = (
            torch.matmul(q, k.repeat_interleave(rep, dim=1).transpose(2, 3))
            * self.scaling
            + cmask
        )
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        o = torch.matmul(aw, v.repeat_interleave(rep, dim=1))
        o = o.transpose(1, 2).reshape(batch, seq_len, -1)
        return attn.o_proj(o), k, v

    def forward(self, inputs_embeds, cos, sin, cmask, ds0, ds1, ds2):
        ds = [ds0, ds1, ds2]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        h = inputs_embeds
        keys = []
        values = []
        for i, layer in enumerate(self.layers):
            residual = h
            out, k, v = self._attn_prefill(
                layer.self_attn,
                rmsnorm(h, layer.input_layernorm.weight, self.eps),
                c,
                s,
                cmask,
            )
            h = residual + out
            residual = h
            mlp = layer.mlp
            pn = rmsnorm(h, layer.post_attention_layernorm.weight, self.eps)
            h = residual + mlp.down_proj(
                torch.nn.functional.silu(mlp.gate_proj(pn)) * mlp.up_proj(pn)
            )
            if i < self.deepstack_layers:
                h = h + ds[i]
            keys.append(k.contiguous())
            values.append(v.contiguous())
        h = rmsnorm(h, self.norm.weight, self.eps)
        logits = h @ self.lm_head_w.t()
        # Interleave K/V per layer so the C ABI matches DeepSeek-style kv0..kv55.
        kv = []
        for k, v in zip(keys, values):
            kv.append(k)
            kv.append(v)
        return (logits, *kv)


class DecoderTraceDecodeKV(DecoderTraceRT):
    """One-token decode against the fixed-length K/V caches from prefill.

    Writes the new K/V row into the cache at `pos` (index_copy) and attends
    over the full padded length N with a [1,1,1,N] causal mask (visible
    positions 0..pos). Deepstack inputs are zeros for decode steps — image
    features were already fused during prefill.
    """

    def _attn_decode(self, attn, h, cos, sin, cmask, k_cache, v_cache, pos):
        batch, seq_len, _ = h.shape
        assert seq_len == 1
        q = rmsnorm(
            attn.q_proj(h).view(batch, 1, self.n_heads, self.head_dim),
            attn.q_norm.weight,
            self.eps,
        ).transpose(1, 2)
        k = rmsnorm(
            attn.k_proj(h).view(batch, 1, self.n_kv, self.head_dim),
            attn.k_norm.weight,
            self.eps,
        ).transpose(1, 2)
        v = (
            attn.v_proj(h)
            .view(batch, 1, self.n_kv, self.head_dim)
            .transpose(1, 2)
        )
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # Static-cache style write: keep rank-4 buffers, update one slot.
        pos_idx = pos.reshape(-1)
        k_cache = k_cache.index_copy(2, pos_idx, k)
        v_cache = v_cache.index_copy(2, pos_idx, v)
        rep = self.n_heads // self.n_kv
        k_full = k_cache.repeat_interleave(rep, dim=1)
        v_full = v_cache.repeat_interleave(rep, dim=1)
        aw = torch.matmul(q, k_full.transpose(2, 3)) * self.scaling + cmask
        aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        o = torch.matmul(aw, v_full).transpose(1, 2).reshape(batch, 1, -1)
        return attn.o_proj(o), k_cache, v_cache

    def forward(
        self,
        inputs_embeds,
        cos,
        sin,
        cmask,
        ds0,
        ds1,
        ds2,
        cache_position,
        *past_kv,
    ):
        # past_kv: interleaved [k0,v0,k1,v1,...], each [1,n_kv,MAX,hd]
        ds = [ds0, ds1, ds2]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        h = inputs_embeds
        out_kv = []
        for i, layer in enumerate(self.layers):
            k_in = past_kv[2 * i]
            v_in = past_kv[2 * i + 1]
            residual = h
            out, k_out, v_out = self._attn_decode(
                layer.self_attn,
                rmsnorm(h, layer.input_layernorm.weight, self.eps),
                c,
                s,
                cmask,
                k_in,
                v_in,
                cache_position,
            )
            h = residual + out
            residual = h
            mlp = layer.mlp
            pn = rmsnorm(h, layer.post_attention_layernorm.weight, self.eps)
            h = residual + mlp.down_proj(
                torch.nn.functional.silu(mlp.gate_proj(pn)) * mlp.up_proj(pn)
            )
            if i < self.deepstack_layers:
                h = h + ds[i]
            out_kv.append(k_out.contiguous())
            out_kv.append(v_out.contiguous())
        h = rmsnorm(h, self.norm.weight, self.eps)
        logits = h @ self.lm_head_w.t()
        return (logits, *out_kv)


def pack_decode_linear_weights(graph, params, vecsize=16):
    """Panel-pack static matmul B weights for decode GEMV; skip activation mats.

    Decode is M=1 (GEMV). The packed layout used by
    `-matmul-vectorization-decode-packed=vector-size=V` stores B as panels of
    width V so the kernel can stream contiguous vector loads instead of
    striding across a huge N (e.g. lm_head N=151936).

    Layout transform on a [K, N] weight (N % vecsize == 0)::

        [K, N] -> reshape [K, N/V, V] -> permute [N/V, K, V] -> reshape [K, N]

    Unlike buddy's all-or-nothing ``pack_decode_matmul_weights``, Qwen3-VL
    decode also has attention matmuls whose B is a KV *activation* (not a
    Placeholder param). Those are left unpacked; only Placeholder B operands
    are rewritten. Returns packed param indices and (K, N) shapes so
    ``lower_to_obj.sh`` can detect the shapes file and enable the packed pass.
    """
    from buddy.compiler.graph.operation import AddMMOp, MatmulOp, PlaceholderOp

    weight_operand = {MatmulOp: 1, AddMMOp: 2}
    graph._params_ref = params
    name_to_index = {p.name: i for i, p in enumerate(graph.params)}
    packed_indices = []
    packed_shapes = []
    for node in graph.body:
        operand = weight_operand.get(type(node))
        if operand is None or len(node.args) <= operand:
            continue
        weight_name = str(node.args[operand])
        index = name_to_index.get(weight_name)
        # Skip non-params (activations) and already-packed duplicates.
        if index is None or not isinstance(
            graph.node_table.get(weight_name), PlaceholderOp
        ):
            continue
        if index in packed_indices:
            continue
        weight = params[index]
        if weight.dim() != 2:
            continue
        k, n = int(weight.shape[0]), int(weight.shape[1])
        if n % vecsize != 0:
            raise RuntimeError(
                f"decode pack: weight '{weight_name}' N={n} not divisible by "
                f"vecsize={vecsize}"
            )
        params[index] = (
            weight.detach()
            .reshape(k, n // vecsize, vecsize)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(k, n)
        )
        packed_indices.append(index)
        packed_shapes.append((k, n))
    return packed_indices, packed_shapes


def import_graph(
    module,
    out_dir,
    prefix,
    *example_inputs,
    template_partitioned=False,
    func_name="forward",
    fuse_patterns=None,
    pack_decode_vecsize=None,
):
    import numpy
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import (
        GraphDriver,
        TemplatePartitionedGraphDriver,
        build_transformer_partition_plan,
    )
    from buddy.compiler.graph.transform import simply_fuse
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as inductor_decomp

    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=func_name,
    )
    with torch.no_grad():
        graphs = dynamo.importer(module, *example_inputs)
    print(f"[import] graphs={len(graphs)} func={func_name}")
    graph = graphs[0]
    params = dynamo.imported_params[graph]
    patterns = fuse_patterns if fuse_patterns is not None else [simply_fuse]

    packed_shapes = []
    if pack_decode_vecsize:
        from buddy.compiler.graph.transform.eliminate_matmul_transpose_reshape import (
            eliminate_matmul_transpose_reshape,
        )
        from buddy.compiler.graph.transform.eliminate_weight_transpose import (
            eliminate_transpose,
        )

        # Fold `A @ W.T` style graph edges into a transposed Placeholder so
        # pack_decode sees a true weight B operand. Doing pack before this
        # leaves 0 packed weights (B is still a Transpose of a param).
        # Order matches DeepSeek's decode packing pipeline.
        graph._params_ref = params
        eliminate_transpose(graph)
        eliminate_matmul_transpose_reshape(graph)
        packed_idx, packed_shapes = pack_decode_linear_weights(
            graph, params, vecsize=pack_decode_vecsize
        )
        shape_str = ",".join(f"{k}x{n}" for k, n in sorted(set(packed_shapes)))
        print(
            f"[pack] decode vec={pack_decode_vecsize} "
            f"weights={len(packed_idx)} shapes={shape_str}"
        )
        # lower_to_obj.sh looks for decoder_decode_packed_shapes.txt next to
        # the decode MLIR and enables -matmul-vectorization-decode-packed.
        with open(
            os.path.join(out_dir, f"{prefix}_packed_shapes.txt"), "w"
        ) as f:
            f.write(shape_str + "\n")
            f.write(str(pack_decode_vecsize) + "\n")

    graph.fuse_ops(patterns)

    if template_partitioned:
        mlir_dir = os.path.join(out_dir, "layer_partitioned")
        os.makedirs(mlir_dir, exist_ok=True)

        for filename in os.listdir(mlir_dir):
            if filename.startswith(
                f"{prefix}_subgraph0_forward_"
            ) and filename.endswith(".mlir"):
                os.remove(os.path.join(mlir_dir, filename))

        plan = build_transformer_partition_plan(graph)
        driver = TemplatePartitionedGraphDriver(graph, plan)
        subgraphs = driver.build_template_subgraphs()

        if len(subgraphs) != len(plan.templates):
            raise ValueError(
                f"{prefix}: templates={len(plan.templates)}, "
                f"subgraphs={len(subgraphs)}"
            )

        template_files = []

        for unit, subgraph in zip(
            plan.templates,
            subgraphs,
            strict=True,
        ):
            subgraph.lower_to_top_level_ir()

            filename = (
                f"{prefix}_{driver.template_symbol(unit.template_id)}.mlir"
            )

            with open(
                os.path.join(mlir_dir, filename),
                "w",
            ) as module_file:
                print(subgraph._imported_module, file=module_file)

            template_files.append(filename)

        forward_file = f"{prefix}_forward.mlir"

        with open(
            os.path.join(mlir_dir, forward_file),
            "w",
        ) as module_file:
            print(
                driver.construct_template_combined_main_graph(True),
                file=module_file,
            )

        manifest = {
            "graph": prefix,
            "template_materialization": True,
            "forward": forward_file,
            "template_files": template_files,
        }

        with open(
            os.path.join(mlir_dir, "partition_manifest.json"),
            "w",
        ) as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")

    else:
        driver = GraphDriver(graph)
        driver.subgraphs[0].lower_to_top_level_ir()

        with open(
            os.path.join(
                out_dir,
                f"{prefix}_subgraph0.mlir",
            ),
            "w",
        ) as module_file:
            print(driver.subgraphs[0]._imported_module, file=module_file)

        with open(
            os.path.join(
                out_dir,
                f"{prefix}_forward.mlir",
            ),
            "w",
        ) as module_file:
            print(driver.construct_main_graph(True), file=module_file)
    flats = []
    for param in params:
        tensor = param.detach().cpu().contiguous()
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"qwen3_vl import expected float16 weights, got {tensor.dtype}"
            )
        flats.append(tensor.numpy().reshape([-1]))
    all_param = numpy.concatenate(flats)
    all_param.tofile(os.path.join(out_dir, f"{prefix}_arg0.data"))
    return all_param.size


def cmd_import_vision(args):
    os.makedirs(VISION_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    processor, model = load_processor_and_model(COMPUTE_DTYPE, eager_attn=True)
    enc = encode_image_prompt(processor, TEST_IMAGE, "ocr")
    pixel_values = enc["pixel_values"].to(dtype=COMPUTE_DTYPE)
    grid_thw = enc["image_grid_thw"].long()
    print(
        f"[in] pixel_values {tuple(pixel_values.shape)} grid {grid_thw.tolist()}"
    )

    vm = model.model.visual
    ref = vm(hidden_states=pixel_values, grid_thw=grid_thw)
    ref_pooled = ref.pooler_output
    ref_ds = list(ref.deepstack_features)
    print(
        f"[ref] pooled {tuple(ref_pooled.shape)} | "
        f"deepstack x{len(ref_ds)} each {tuple(ref_ds[0].shape)}"
    )

    pos_embeds = vm.fast_pos_embed_interpolate(grid_thw)
    rotary = vm.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary, rotary), dim=-1)
    trace = VisionTrace(vm, pos_embeds, emb.cos(), emb.sin()).eval()
    trace = trace.to(dtype=COMPUTE_DTYPE)
    out = trace(pixel_values)
    pooled, ds = out[0], list(out[1:])
    dp = (pooled - ref_pooled).abs().max().item()
    dd = max((a - b).abs().max().item() for a, b in zip(ds, ref_ds))
    pooled_scale = max(ref_pooled.abs().max().item(), 1e-6)
    ds_scale = max(max(t.abs().max().item() for t in ref_ds), 1e-6)
    print(
        f"[equiv] pooled max|delta|={dp:.3e} ({dp / pooled_scale:.3e} rel)  "
        f"deepstack max|delta|={dd:.3e} ({dd / ds_scale:.3e} rel)"
    )
    # fp16 accumulation through the ViT is looser than the f32 wrapper.
    # A structural mismatch shows up as a large relative error, not ~1e-2.
    assert dp / pooled_scale < 5e-2 and dd / ds_scale < 5e-2, (
        "trace wrapper diverges from HF vision model "
        f"(pooled {dp:.3e}, deepstack {dd:.3e})"
    )
    print("[equiv] OK: trace-friendly wrapper matches HF vision model")

    if args.no_import:
        return
    print("[import] running buddy DynamoCompiler on the vision wrapper ...")
    weight_count = import_graph(
        trace,
        VISION_DIR,
        "vision",
        pixel_values,
        template_partitioned=getattr(
            args,
            "experimental_template_partitioned",
            False,
        ),
    )
    print(
        f"[import] OK -> {VISION_DIR}/vision_forward.mlir weights={weight_count}"
    )


def make_decoder_inputs(seq_len):
    golden = capture_decoder_golden()
    lm, model = golden["lm"], golden["model"]
    inputs_embeds = golden["inputs_embeds"]
    pos = golden["position_ids"]
    vmask = golden["visual_pos_masks"]
    deepstack = golden["deepstack"]
    _, prompt_len, hidden = inputs_embeds.shape

    rope_pos = pos[1:] if pos.shape[0] == 4 else pos
    max_pos = int(rope_pos.max())
    tail = torch.arange(max_pos + 1, max_pos + 1 + (seq_len - prompt_len))
    tail = tail.view(1, 1, -1).expand(3, 1, seq_len - prompt_len)
    rope_pos_n = torch.cat([rope_pos, tail], dim=2)
    cos, sin = lm.rotary_emb(
        torch.zeros(
            1,
            seq_len,
            hidden,
            dtype=COMPUTE_DTYPE,
            device=inputs_embeds.device,
        ),
        rope_pos_n.to(device=inputs_embeds.device),
    )
    cos, sin = cos[0], sin[0]
    cmask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=COMPUTE_DTYPE), 1
    )
    cmask = cmask.view(1, 1, seq_len, seq_len)
    padded_embeds = torch.zeros(1, seq_len, hidden, dtype=COMPUTE_DTYPE)
    padded_embeds[:, :prompt_len] = inputs_embeds.to(dtype=COMPUTE_DTYPE)
    img_pos = vmask[0].nonzero(as_tuple=True)[0]
    padded_deepstack = []
    for d in deepstack:
        f = torch.zeros(1, seq_len, hidden, dtype=COMPUTE_DTYPE)
        f[0, img_pos] = d.to(dtype=COMPUTE_DTYPE)
        padded_deepstack.append(f)
    cos = cos.to(dtype=COMPUTE_DTYPE)
    sin = sin.to(dtype=COMPUTE_DTYPE)
    trace = DecoderTraceRT(
        lm, model.lm_head.weight, deepstack_layers=len(deepstack)
    ).eval()
    trace = trace.to(dtype=COMPUTE_DTYPE)
    ref_tok = int(golden["logits"][0, prompt_len - 1].argmax())
    return (
        trace,
        model,
        prompt_len,
        padded_embeds,
        cos,
        sin,
        cmask,
        padded_deepstack,
        ref_tok,
    )


def cmd_import_decoder_rt(args):
    os.makedirs(DECODER_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    trace, model, prompt_len, inputs_embeds, cos, sin, cmask, ds, ref_tok = (
        make_decoder_inputs(args.seq_len)
    )
    logits = trace(inputs_embeds, cos, sin, cmask, ds[0], ds[1], ds[2])
    tok = int(logits[0, prompt_len - 1].argmax())
    print(
        f"[rt] N={args.seq_len} next token at prompt end = {tok} "
        f"(hf fp16 {ref_tok})"
    )
    assert tok == ref_tok, f"fp16 trace token {tok} != HF fp16 token {ref_tok}"

    write_f16(
        model.lm_head.weight, os.path.join(DECODER_DIR, "embed_table.bin")
    )
    if args.no_import:
        return
    weight_count = import_graph(
        trace,
        DECODER_DIR,
        "decoder",
        inputs_embeds,
        cos,
        sin,
        cmask,
        ds[0],
        ds[1],
        ds[2],
        template_partitioned=getattr(
            args,
            "experimental_template_partitioned",
            False,
        ),
    )
    print(
        f"[rt] imported -> {DECODER_DIR}/decoder_forward.mlir weights={weight_count}"
    )


def cmd_import_decoder_kv(args):
    """Import prefill + decode graphs with fixed-length GQA KV caches.

    Emits under DECODER_DIR (default artifacts/decoder_rt):

      decoder_prefill_{forward,subgraph0}.mlir + decoder_prefill_arg0.data
      decoder_decode_{forward,subgraph0}.mlir  + decoder_decode_arg0.data
      decoder_decode_packed_shapes.txt         (triggers packed GEMV lower)
      embed_table.bin

    Prefill weights stay row-major for BLIS; decode weights are panel-packed
    (vecsize=16 for +zvl256b f16). CMake links both via link_decoder_kv_shim.sh.
    """
    os.makedirs(DECODER_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    _, model, prompt_len, embeds, cos, sin, cmask, ds, ref_tok = (
        make_decoder_inputs(args.seq_len)
    )
    lm = model.model.language_model
    pre = (
        DecoderTracePrefillKV(
            lm, model.lm_head.weight, deepstack_layers=len(ds)
        )
        .eval()
        .to(dtype=COMPUTE_DTYPE)
    )
    dec = (
        DecoderTraceDecodeKV(lm, model.lm_head.weight, deepstack_layers=len(ds))
        .eval()
        .to(dtype=COMPUTE_DTYPE)
    )

    # Correctness gate: prefill next-token must match HF fp16 reference.
    pre_out = pre(embeds, cos, sin, cmask, ds[0], ds[1], ds[2])
    pre_logits, *kvs = pre_out
    tok = int(pre_logits[0, prompt_len - 1].argmax())
    print(
        f"[kv] prefill next token = {tok} (hf fp16 {ref_tok}) "
        f"kv_tensors={len(kvs)} shape={tuple(kvs[0].shape)}"
    )
    assert tok == ref_tok, f"prefill token {tok} != HF {ref_tok}"
    assert len(kvs) == 2 * len(lm.layers)

    # Example shapes for Dynamo import: one decode step at the prompt end.
    pos = prompt_len
    N = args.seq_len
    HID = embeds.shape[-1]
    emb = model.lm_head.weight[tok].view(1, 1, -1).to(dtype=COMPUTE_DTYPE)
    cos1 = cos[pos : pos + 1]
    sin1 = sin[pos : pos + 1]
    cm1 = torch.full((1, 1, 1, N), float("-inf"), dtype=COMPUTE_DTYPE)
    cm1[0, 0, 0, : pos + 1] = 0
    # Distinct zero tensors: Dynamo aliases identical Python objects into one
    # memref, which collapses three deepstack inputs into a single ABI slot.
    z0 = torch.zeros(1, 1, HID, dtype=COMPUTE_DTYPE)
    z1 = torch.zeros(1, 1, HID, dtype=COMPUTE_DTYPE)
    z2 = torch.zeros(1, 1, HID, dtype=COMPUTE_DTYPE)
    pos_t = torch.tensor([pos], dtype=torch.int64)
    dec_out = dec(emb, cos1, sin1, cm1, z0, z1, z2, pos_t, *kvs)
    print(
        f"[kv] decode example logits {tuple(dec_out[0].shape)} "
        f"token={int(dec_out[0][0, 0].argmax())}"
    )

    write_f16(
        model.lm_head.weight, os.path.join(DECODER_DIR, "embed_table.bin")
    )
    if args.no_import:
        return

    # Prefill graph: same operand roles as the legacy full decoder (BLIS).
    w_pre = import_graph(
        pre,
        DECODER_DIR,
        "decoder_prefill",
        embeds,
        cos,
        sin,
        cmask,
        ds[0],
        ds[1],
        ds[2],
        func_name="forward_prefill",
    )
    # Decode graph: pack linear B weights (attn B = activation → skipped).
    w_dec = import_graph(
        dec,
        DECODER_DIR,
        "decoder_decode",
        emb,
        cos1,
        sin1,
        cm1,
        z0,
        z1,
        z2,
        pos_t,
        *kvs,
        func_name="forward_decode",
        pack_decode_vecsize=16,
    )
    print(
        f"[kv] imported prefill weights={w_pre} decode weights={w_dec} -> {DECODER_DIR}"
    )


def cmd_preprocess(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    processor, config = load_processor_and_config()
    inputs = encode_image_prompt(processor, args.image_path, args.prompt)

    input_ids = inputs["input_ids"]
    grid = inputs["image_grid_thw"]
    prompt_len = input_ids.shape[1]
    if grid.tolist() != [[1, 14, 28]]:
        raise RuntimeError(f"unexpected grid {grid.tolist()}")
    if prompt_len >= MAX_SEQ_LEN:
        raise RuntimeError(
            f"prompt too long: S0={prompt_len} >= N={MAX_SEQ_LEN}"
        )

    pos = compute_3d_position_ids(
        config,
        input_ids=input_ids,
        image_grid_thw=grid,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    rope_pos = pos[-3:]
    max_pos = int(rope_pos.max())
    tail = torch.arange(max_pos + 1, max_pos + 1 + (MAX_SEQ_LEN - prompt_len))
    tail = tail.view(1, 1, -1).expand(3, 1, MAX_SEQ_LEN - prompt_len)
    rope_pos_n = torch.cat([rope_pos, tail], dim=2)
    hidden = config.text_config.hidden_size
    cos, sin = text_rotary(config, MAX_SEQ_LEN, hidden, rope_pos_n)
    cmask = np.triu(np.full((MAX_SEQ_LEN, MAX_SEQ_LEN), -np.inf, np.float16), 1)
    cmask = cmask.reshape(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN)
    img_pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].numpy()

    write_f16(inputs["pixel_values"], os.path.join(out, "pixel_values.bin"))
    input_ids[0].numpy().astype(np.int64).tofile(
        os.path.join(out, "input_ids.i64")
    )
    img_pos.astype(np.int64).tofile(os.path.join(out, "img_pos.i64"))
    write_f16(cos[0], os.path.join(out, "cos.bin"))
    write_f16(sin[0], os.path.join(out, "sin.bin"))
    write_f16(cmask, os.path.join(out, "cmask.bin"))
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(
            f"{prompt_len} {MAX_SEQ_LEN} {len(img_pos)} {hidden} "
            f"{config.text_config.vocab_size}\n"
        )
    print(
        f"[preprocess] S0={prompt_len} N={MAX_SEQ_LEN} "
        f"img_tokens={len(img_pos)} -> {out}"
    )


def stage_file(src, dst):
    if os.path.lexists(dst):
        if not os.path.islink(dst) and os.path.realpath(
            src
        ) == os.path.realpath(dst):
            return
        os.remove(dst)
    shutil.copy2(src, dst)


def stage_processor_archive(model_dir, dst):
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("QWEN3_VL_MODEL_PATH must point to a model snapshot")

    def include(path):
        rel = os.path.relpath(path, model_dir)
        parts = rel.split(os.sep)
        if any(p in {".git", "__pycache__"} for p in parts):
            return False
        if os.path.isdir(path):
            return True
        return not path.endswith(PROCESSOR_EXCLUDE_SUFFIXES)

    with tarfile.open(dst, "w") as tar:
        for root, dirs, files in os.walk(model_dir):
            dirs[:] = [d for d in dirs if include(os.path.join(root, d))]
            for filename in files:
                path = os.path.join(root, filename)
                if include(path):
                    tar.add(path, arcname=os.path.relpath(path, model_dir))


def file_bytes(path):
    return os.path.getsize(path)


def rhal_file_constant(idx, name, path):
    return (
        f'  rhal.constant @{name} {{id = {idx} : i32, storage = "external",\n'
        f"                                type = tensor<{file_bytes(path)}xi8>,\n"
        f'                                uri = "file:{os.path.basename(path)}"}}\n'
    )


def cmd_stage(args):
    os.makedirs(PKG_DIR, exist_ok=True)
    rax_pack = os.environ.get(
        "RAX_PACK", os.path.join(REPO, "build", "bin", "rax-pack")
    )
    runner_so = os.environ.get(
        "QWEN3_VL_RUNNER_SO", os.path.join(PKG_DIR, "qwen3_vl_runner.so")
    )
    vocab = os.path.join(REPO, "examples", "BuddyQwen3", "vocab.txt")
    serving_so = os.environ.get("QWEN3_VL_SERVING_SO", "")
    if serving_so:
        stage_file(serving_so, os.path.join(PKG_DIR, "qwen3_vl_serving.so"))

    stage_file(
        os.path.join(VISION_DIR, "vision_shim.so"),
        os.path.join(PKG_DIR, "vision_shim.so"),
    )
    stage_file(
        os.path.join(DECODER_DIR, "decoder_shim.so"),
        os.path.join(PKG_DIR, "decoder_shim.so"),
    )
    stage_file(
        os.path.join(VISION_DIR, "vision_arg0.data"),
        os.path.join(PKG_DIR, "vision_weights.data"),
    )
    # KV builds emit separate prefill (row-major) and decode (panel-packed)
    # weight blobs. Legacy full-forward builds only have decoder_arg0.data.
    # Prefill weights are staged as decoder_weights.data so existing runner
    # lookups keep working; packed decode weights become an extra rax resource.
    prefill_weights = os.path.join(DECODER_DIR, "decoder_prefill_arg0.data")
    decode_weights = os.path.join(DECODER_DIR, "decoder_decode_arg0.data")
    legacy_weights = os.path.join(DECODER_DIR, "decoder_arg0.data")
    if os.path.isfile(prefill_weights):
        stage_file(
            prefill_weights, os.path.join(PKG_DIR, "decoder_weights.data")
        )
        if not os.path.isfile(decode_weights):
            raise RuntimeError(
                "KV import produced decoder_prefill_arg0.data but missing "
                f"{decode_weights}"
            )
        stage_file(
            decode_weights, os.path.join(PKG_DIR, "decoder_decode_weights.data")
        )
    else:
        stage_file(
            legacy_weights, os.path.join(PKG_DIR, "decoder_weights.data")
        )
    stage_file(
        os.path.join(DECODER_DIR, "embed_table.bin"),
        os.path.join(PKG_DIR, "embed_table.bin"),
    )
    stage_file(runner_so, os.path.join(PKG_DIR, "qwen3_vl_runner.so"))
    stage_file(__file__, os.path.join(PKG_DIR, "qwen3_vl_codegen.py"))
    stage_processor_archive(MODEL_DIR, os.path.join(PKG_DIR, PROCESSOR_ARCHIVE))
    shutil.copy(vocab, os.path.join(PKG_DIR, "vocab.txt"))

    sh = os.path.join(PKG_DIR, "preprocess.sh")
    with open(sh, "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'export BUDDY_MLIR_BUILD_DIR="${BUDDY_MLIR_BUILD_DIR:-${SCRIPT_DIR}}"\n'
            'export LLVM_MLIR_BUILD_DIR="${LLVM_MLIR_BUILD_DIR:-${SCRIPT_DIR}}"\n'
            'export PYTHONPATH="${SCRIPT_DIR}/python_packages:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH:-}"\n'
            'export CUDA_VISIBLE_DEVICES=""\n'
            f'PROCESSOR_DIR="${{QWEN3_VL_PROCESSOR_DIR:-$3/{PROCESSOR_DIRNAME}}}"\n'
            f'if [[ -z "${{QWEN3_VL_MODEL_PATH:-}}" && ! -d "${{PROCESSOR_DIR}}" && -f "${{SCRIPT_DIR}}/{PROCESSOR_ARCHIVE}" ]]; then\n'
            f'  mkdir -p "${{PROCESSOR_DIR}}"\n'
            f'  tar -xf "${{SCRIPT_DIR}}/{PROCESSOR_ARCHIVE}" -C "${{PROCESSOR_DIR}}"\n'
            "fi\n"
            'if [[ -z "${QWEN3_VL_MODEL_PATH:-}" && -d "${PROCESSOR_DIR}" ]]; then\n'
            '  export QWEN3_VL_MODEL_PATH="${PROCESSOR_DIR}"\n'
            "fi\n"
            'PREPROCESS_PY="${QWEN3_VL_PREPROCESS_PY:-${SCRIPT_DIR}/qwen3_vl_codegen.py}"\n'
            'exec "${BUDDY_PYTHON:-python3}" "${PREPROCESS_PY}" preprocess "$1" "$2" "$3"\n'
        )
    os.chmod(sh, 0o755)

    print("[stage] pre-processing bundled test image ...")
    preprocess_env = os.environ.copy()
    preprocess_env.setdefault("BUDDY_PYTHON", sys.executable)
    subprocess.run(
        ["bash", sh, TEST_IMAGE, PROMPT, PKG_DIR],
        check=True,
        env=preprocess_env,
    )

    # input_ids/img_pos/cos/sin/cmask/meta and pixel_values are all produced
    # in pure C++ at runtime now (see Qwen3VLRunner.cpp / ImagePreprocess.h),
    # so nothing here still needs the Python preprocess.sh helper or the HF
    # processor archive bundled into the .rax; img_pos/cos/sin/cmask/meta are
    # still baked in as query-independent constants (see cmd_preprocess) and
    # pixel_values.bin is only the fallback for --image bundled/omitted.
    resources = [
        ("vision_weights", "vision_weights.data"),
        ("decoder_weights", "decoder_weights.data"),
        ("embed_table", "embed_table.bin"),
        ("pixel_values", "pixel_values.bin"),
        ("img_pos", "img_pos.i64"),
        ("cos", "cos.bin"),
        ("sin", "sin.bin"),
        ("cmask", "cmask.bin"),
        ("meta", "meta.txt"),
    ]
    decode_w_pkg = os.path.join(PKG_DIR, "decoder_decode_weights.data")
    if os.path.isfile(decode_w_pkg):
        # Bundle packed decode weights in the rax so the runner finds them via
        # pkg.file("decoder_decode_weights", ...) without env overrides.
        resources.insert(
            2, ("decoder_decode_weights", "decoder_decode_weights.data")
        )
    serving_attr = (
        ',\n    serving_library = "file:qwen3_vl_serving.so"'
        if serving_so
        else ""
    )
    constants = "".join(
        rhal_file_constant(idx, name, os.path.join(PKG_DIR, filename))
        for idx, (name, filename) in enumerate(resources, start=1)
    )
    manifest = f"""rhal.module @qwen3_vl attributes {{
    version = "0.1.0",
    model_name = "qwen3_vl",
    vocab_uri = "file:vocab.txt",
    runner_library = "file:qwen3_vl_runner.so"{serving_attr}}} {{
{constants}  rhal.codeobj @vision_kernels {{id = 1 : i32, kind = "host_shared_lib",
                                backend = "cpu", uri = "file:vision_shim.so"}}
  rhal.codeobj @decoder_kernels {{id = 2 : i32, kind = "host_shared_lib",
                                backend = "cpu", uri = "file:decoder_shim.so"}}
  rhal.buffer @pixel  {{space = "host", type = tensor<392x1536xf16>}}
  rhal.buffer @logits {{space = "host", type = tensor<1x{MAX_SEQ_LEN}x{VOCAB_SIZE}xf16>}}
  rhal.func @forward_vision {{inputs = ["pixel"], outputs = ["logits"],
                      dispatch = "vision_kernels", args = ["pixel", "logits"]}}
  rhal.func @forward_decoder {{inputs = ["pixel"], outputs = ["logits"],
                      dispatch = "decoder_kernels", args = ["pixel", "logits"]}}
}}
"""
    mpath = os.path.join(PKG_DIR, "qwen3_vl.mlir")
    with open(mpath, "w") as f:
        f.write(manifest)
    rax = os.path.join(PKG_DIR, "qwen3_vl.rax")
    cmd = [rax_pack, mpath, "-o", rax]
    if os.environ.get("BUDDY_RAX_EMBED_PAYLOAD", "ON").upper() not in {
        "0",
        "FALSE",
        "OFF",
        "NO",
    }:
        cmd.append("--embed-payload")
    subprocess.run(cmd, check=True)
    print(f"[stage] package ready: {PKG_DIR}")
    print(
        f"[run]  {REPO}/build/bin/buddy-cli --model {rax} \\\n"
        f"         --image {TEST_IMAGE} --prompt '{PROMPT}'"
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import-vision")
    p.add_argument(
        "--no-import",
        action="store_true",
        help="only run the PyTorch equivalence check",
    )
    p.set_defaults(func=cmd_import_vision)

    p = sub.add_parser("import-decoder-rt")
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--no-import", action="store_true")
    p.set_defaults(func=cmd_import_decoder_rt)

    p = sub.add_parser("import-decoder-kv")
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--no-import", action="store_true")
    p.set_defaults(func=cmd_import_decoder_kv)

    p = sub.add_parser("preprocess")
    p.add_argument("image_path")
    p.add_argument("prompt")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("stage")
    p.set_defaults(func=cmd_stage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
