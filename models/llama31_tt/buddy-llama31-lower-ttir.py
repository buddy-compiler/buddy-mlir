# ===- buddy-llama31-lower-ttir.py ---------------------------------------------
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
# Llama-3.1-8B-Instruct (or any Llama3-shaped model) to Buddy Graph to
# ``lower_to_ttir()`` (bf16 TTIR). Mirrors ``buddy-deepseek-r1-lower-ttir.py``
# so the downstream ``ttmlir-opt`` / ``ttmlir-translate`` / ``ttrt`` pipeline
# stays identical. Requires ``ttmlir`` on ``PYTHONPATH`` and a HuggingFace
# model checkout (or set ``LLAMA31_MODEL_PATH`` to a local directory).
#
# Prefill vs decode:
#   - ``--mode prefill`` traces one forward with a prompt-shaped input_ids.
#   - ``--mode decode`` runs a prefill pass to build ``past_key_values``,
#     then traces **one decode step** with ``past_key_values`` +
#     ``use_cache=True`` (same recipe as the DeepSeek reference).
#
# Usage::
#
#   export PYTHONPATH=$BUDDY_BUILD/python_packages:$PYTHONPATH
#   export PYTHONPATH=$TTMLIR_BUILD/python_packages:$PYTHONPATH
#   python buddy-llama31-lower-ttir.py --mode prefill --seq 32
#   python buddy-llama31-lower-ttir.py --mode prefill --static-cache \
#       --max-cache-len 1024 --ttmlir-opt "$(command -v ttmlir-opt)"
#   python buddy-llama31-lower-ttir.py --mode decode  --static-cache \
#       --max-cache-len 1024
#
# If HuggingFace access is unavailable, point ``LLAMA31_MODEL_PATH`` at a local
# checkout/cache directory before running this script.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import GQAAttentionFusedOp, PlaceholderOp
from buddy.compiler.graph.transform import (
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from buddy.compiler.graph.ttir_import import (
    append_ttir_forward_bf16_f32_packed_i64_runtime,
)
from buddy.compiler.ops import tosa
from transformers.models.llama import modeling_llama

_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Lower Llama-3.1-8B subgraph to TTIR MLIR (bf16)."
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("LLAMA31_MODEL_PATH", _DEFAULT_MODEL),
        help="HF model id or local path.",
    )
    p.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="prefill",
        help="Graph to trace: prefill (prompt) or single decode step.",
    )
    p.add_argument(
        "--scope",
        choices=(
            "full",
            "layer",
            "block",
            "embed",
            "embed_roundtrip",
            "embed_weightmul",
            "rmsstat",
            "rmsscale",
            "rmsunweighted",
            "rmsnorm",
            "kproj",
            "rmsnorm_kproj",
            "qkvproj_input",
            "rope_input",
            "layer_stages",
            "attn_split",
            "sdpa_input",
            "sdpa_packed_input",
            "mlp_input",
            "mlp_split",
            "mlp_down_input",
            "post_attn_norm_input",
        ),
        default="full",
        help=(
            "Graph scope to trace. 'full' keeps the original full CausalLM "
            "path; 'layer' traces embed -> layer0 -> norm -> lm_head; "
            "'block' traces only layer0 from hidden states; 'embed' traces "
            "token embedding only; 'embed_roundtrip' traces embed bf16->f32"
            "->bf16; 'embed_weightmul' traces embedding times the RMSNorm "
            "weight; 'rmsstat' traces the RMSNorm mean-square statistic; "
            "'rmsscale' traces RMSNorm rsqrt scale; 'rmsunweighted' traces "
            "RMSNorm before weight multiply; "
            "'rmsnorm' traces embed -> layer0 input RMSNorm; 'kproj' traces "
            "embed -> layer0 input RMSNorm -> K projection; "
            "'rmsnorm_kproj' returns both layer0 input RMSNorm and the "
            "K projection from the same graph; "
            "'qkvproj_input' traces Q/K/V projections from hidden states "
            "provided as the graph input; "
            "'rope_input' traces RoPE from packed Q/K plus cos/sin graph "
            "inputs; "
            "'layer_stages' returns layer0 input norm, attention, residual, "
            "post-attention norm, MLP, and layer output; "
            "'attn_split' returns layer0 attention internals before "
            "post-attention RMSNorm; "
            "'sdpa_input' traces fused masked SDPA from Q/K/V graph inputs; "
            "'sdpa_packed_input' traces masked SDPA from packed 3D Q/K/V "
            "graph inputs; "
            "'mlp_input' traces layer0 MLP from hidden states provided as "
            "the graph input; "
            "'mlp_split' returns layer MLP silu(gate), up, product, and down "
            "from hidden states provided as the graph input; "
            "'mlp_down_input' traces layer MLP down projection from product "
            "states provided as the graph input; "
            "'post_attn_norm_input' traces layer0 post-attention RMSNorm "
            "from hidden states provided as the graph input."
        ),
    )
    p.add_argument(
        "--attn-implementation",
        choices=("default", "eager", "sdpa"),
        default="default",
        help=(
            "Override the HuggingFace attention implementation before tracing. "
            "'eager' is useful for official TTIR alignment because it can avoid "
            "capturing fused torch SDPA."
        ),
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for traced inputs (default 1; official TTIR examples use 32).",
    )
    p.add_argument(
        "--seq",
        type=int,
        default=32,
        help="Sequence length for non-static-cache prefill trace (default 32).",
    )
    p.add_argument(
        "--cur-pos-placeholder",
        type=str,
        default=None,
        help=(
            "For --mode decode: Buddy name of the cache position input "
            "(e.g. arg for current index). Patched into "
            "GQAAttentionFusedOp.kwargs['cur_pos_tensor']."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=here / "ttir_out",
        help="Output directory for MLIR.",
    )
    p.add_argument(
        "--output-stem-prefix",
        default=os.environ.get("BUDDY_LLAMA_TTIR_STEM", "llama31_ttir"),
        help=(
            "Prefix for emitted TTIR files. The default preserves the "
            "existing llama31_ttir_<phase>.mlir names; llama32 packaging uses "
            "llama32_ttir."
        ),
    )
    p.add_argument(
        "--element-dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help="TTIR element type for lower_to_ttir (default bf16).",
    )
    p.add_argument(
        "--ttmlir-opt",
        type=str,
        default=None,
        help="Path to ttmlir-opt for a quick parse check.",
    )
    p.add_argument(
        "--use-proxy",
        action="store_true",
        help=(
            "Set http(s)_proxy to the same URL as interactive ``proxy_on`` "
            "(http://192.168.15.159:7890) for HuggingFace downloads."
        ),
    )
    p.add_argument(
        "--packed-forward",
        action="store_true",
        help=(
            "After lowering, append ``@forward`` that packs bf16/f32 weights "
            "into two 1-D tensors and passes i64 inputs (e.g. token ids) "
            "through, then calls ``@subgraph0``."
        ),
    )
    p.add_argument(
        "--prefill-use-cache",
        action="store_true",
        help=(
            "For --mode prefill (non-static), trace the model with "
            "use_cache=True so the flatbuffer also returns past_key_values."
        ),
    )
    p.add_argument(
        "--static-cache",
        action="store_true",
        help=(
            "Use HuggingFace StaticCache with fixed max_cache_len "
            "(see --max-cache-len). Prefill traces "
            "input_ids.shape=[1, max_cache_len] (pad short prompts at host "
            "side), decode traces past_key_values=[1, n_kv_heads, "
            "max_cache_len, head_dim] + cache_position."
        ),
    )
    p.add_argument(
        "--prune-unused-ttir-args",
        action="store_true",
        help=(
            "After TTIR lowering, remove unused function arguments from the "
            "single generated subgraph. Useful for exact layer/block inventory "
            "when Dynamo leaves dead placeholders in the signature."
        ),
    )
    p.add_argument(
        "--annotate-official-arg-attrs",
        action="store_true",
        help=(
            "After TTIR lowering, annotate aligned layer/block function arguments with "
            "official tt-mlir style ttcore.argument_type/shard_status/ttir.name "
            "metadata. This is metadata-only and lets TTIR->TTNN hoist parameter "
            "const-eval work such as weight transposes."
        ),
    )
    p.add_argument(
        "--runtime-attention-mask",
        action="store_true",
        help=(
            "Trace attention_mask as a runtime input for left-padded batch "
            "prompts, matching a reference Llama generation setup "
            "semantics."
        ),
    )
    p.add_argument(
        "--matmul-as-dot-general",
        action="store_true",
        help=(
            "Lower Buddy MatmulOp/BatchMatmulOp as ttir.dot_general for the "
            "official-alignment path. The default still emits ttir.matmul."
        ),
    )
    p.add_argument(
        "--rmsnorm-mean-as-sum",
        action="store_true",
        help=(
            "Lower MeanOp as ttir.sum + 1/N scale + reshape for the "
            "official-alignment RMSNorm path, and preserve the f32 scalar add "
            "before rsqrt. The default still emits ttir.mean."
        ),
    )
    p.add_argument(
        "--rsqrt-as-sqrt-reciprocal",
        action="store_true",
        help="Lower torch.rsqrt as ttir.sqrt followed by ttir.reciprocal.",
    )
    p.add_argument(
        "--explicit-mul-broadcast",
        action="store_true",
        help="Insert ttir.broadcast before multiply when operand shapes differ.",
    )
    p.add_argument(
        "--mul-broadcast-as-repeat",
        action="store_true",
        help=(
            "With --explicit-mul-broadcast, materialize broadcasted multiply "
            "operands with ttir.repeat_interleave where possible."
        ),
    )
    p.add_argument(
        "--mul-broadcast-as-repeat-op",
        action="store_true",
        help=(
            "With --explicit-mul-broadcast, materialize broadcasted multiply "
            "operands with ttir.repeat."
        ),
    )
    p.add_argument(
        "--softmax-as-explicit",
        action="store_true",
        help=(
            "Lower SoftmaxOp as max/subtract/exp/sum/div for official TTIR "
            "alignment. The default still emits ttir.softmax."
        ),
    )
    p.add_argument(
        "--safe-softmax-mask",
        action="store_true",
        help=(
            "With --softmax-as-explicit, add the official all-masked-row guard "
            "around softmax using eq/logical_not/reduce_or/where."
        ),
    )
    p.add_argument(
        "--mask-as-where",
        action="store_true",
        help=(
            "Build the static causal mask as torch.where(mask, 0, -inf) for "
            "official TTIR alignment. The default keeps the arithmetic mask."
        ),
    )
    p.add_argument(
        "--official-attention-f32",
        action="store_true",
        help=(
            "For eager Llama attention, keep the score/prob/value attention "
            "path in f32 and split the attention scale across Q and K to match "
            "the official tt-mlir layer examples. Cache writes stay bf16."
        ),
    )
    p.add_argument(
        "--silu-as-sigmoid-mul",
        action="store_true",
        help=(
            "Lower SiLU as f32 sigmoid(x) * x plus bf16 cast for official TTIR "
            "alignment. The default still emits ttir.silu."
        ),
    )
    p.add_argument(
        "--embedding-as-gather",
        action="store_true",
        help=(
            "Lower EmbeddingOp as reshape + ui32 ttir.gather + reshape for "
            "official TTIR alignment. The default still emits ttir.embedding."
        ),
    )
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static cache length (default 1024; pairs with --static-cache).",
    )
    p.add_argument(
        "--cache-position",
        type=int,
        default=None,
        help="Concrete cache position value used during decode tracing.",
    )
    p.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help=(
            "Target decoder layer for layer-local probes such as "
            "--scope attn_split (default: 0)."
        ),
    )
    p.add_argument(
        "--synthetic-weights",
        action="store_true",
        help=(
            "Instantiate a bf16 model from config instead of loading HF "
            "weights. Useful for structural layer/block TTIR inventory."
        ),
    )
    p.add_argument(
        "--skip-flash-attn",
        action="store_true",
        help=(
            "Skip the ``flash_attention_prefill`` fusion pass. Useful when "
            "debugging a lowering failure inside SDPA; the resulting TTIR "
            "keeps an explicit softmax+matmul chain."
        ),
    )
    p.add_argument(
        "--device-argmax",
        action="store_true",
        help=(
            "Wrap the model so the graph emits argmax(logits, dim=-1) (i64) "
            "instead of the full bf16 logits. Cuts decode logits.to_host from "
            "~256.5 KB to 8 B; for prefill, from ~256 MB to 8 KB. Requires "
            "ttir.argmax lowering (registered in ttir_llm.py)."
        ),
    )
    p.add_argument(
        "--full-align-wrapper",
        action="store_true",
        help=(
            "For --scope full, bypass HuggingFace's full CausalLM forward and "
            "trace a manual embed -> decoder layers -> norm -> lm_head wrapper "
            "with explicit StaticCache inputs. This is the full-model extension "
            "of the already aligned layer wrapper and avoids HF masking/output "
            "helpers that currently split the graph."
        ),
    )
    return p.parse_args()


def _patch_static_cache_for_buddy() -> None:
    """Replace ``transformers.cache_utils.StaticLayer.update`` so it does not
    emit ``aten.index_copy_`` (which the Buddy TOSA op map does not register).

    Equivalent scatter-via-``where`` implementation, **compatible with both
    Transformers 4.x and 5.5+**:

    - Prefill (``key_states.shape[-2] == max_cache_len``): directly assign
      key/value states as the new cache contents.
    - Decode (``key_states.shape[-2] == 1``): build a boolean mask
      ``arange(max_cache_len) == cache_position[0]`` and ``torch.where``
      the new 1-token state into the corresponding slot.

    Position resolution:
      * Transformers 4.x: ``cache_kwargs={'cache_position': ...}``.
      * Transformers 5.5+: drop ``cache_kwargs``; read ``cumulative_length``
        from the layer instead (kept as a Dynamo placeholder).
    """
    import transformers.cache_utils as cu

    def update_cache_tensors(
        keys, values, key_states, value_states, cache_position
    ):
        L = keys.shape[-2]
        S = key_states.shape[-2]

        if S == L:
            return key_states, value_states

        out_keys = keys
        out_values = values
        for i in range(S):
            pos = cache_position[i : i + 1]
            out_keys = torch.ops.aten.index_put.default(
                out_keys,
                [None, None, pos, None],
                key_states[:, :, i : i + 1, :],
                False,
            )
            out_values = torch.ops.aten.index_put.default(
                out_values,
                [None, None, pos, None],
                value_states[:, :, i : i + 1, :],
                False,
            )
        return out_keys, out_values

    def update(self, key_states, value_states, *args, **kwargs):
        if (
            not getattr(self, "is_initialized", False)
            and getattr(self, "keys", None) is None
        ):
            try:
                self.lazy_initialization(key_states, value_states)
            except TypeError:
                self.lazy_initialization(key_states)

        cache_position = None
        ck = None
        if args:
            if isinstance(args[0], dict):
                ck = args[0]
            elif len(args) > 1 and isinstance(args[1], dict):
                ck = args[1]
        if ck is None and isinstance(kwargs.get("cache_kwargs"), dict):
            ck = kwargs["cache_kwargs"]
        if ck is not None:
            cache_position = ck.get("cache_position")
        if cache_position is None:
            S_ = key_states.shape[-2]
            base = getattr(self, "cumulative_length", None)
            if base is not None:
                if len(base.shape) == 1 and base.shape[0] == S_:
                    cache_position = base
                else:
                    cache_position = torch.arange(S_, device=self.device) + base
            else:
                cache_position = torch.arange(S_, device=self.device)

        keys, values = update_cache_tensors(
            self.keys, self.values, key_states, value_states, cache_position
        )
        self.keys = keys
        self.values = values
        return self.keys, self.values

    def static_cache_update(
        k_cache, v_cache, key_states, value_states, cache_position
    ):
        if cache_position is None:
            S = key_states.shape[-2]
            cache_position = torch.arange(S, device=key_states.device)
        return update_cache_tensors(
            k_cache, v_cache, key_states, value_states, cache_position
        )

    def static_cache_update_method(
        self, key_states, value_states, layer_idx, cache_kwargs=None
    ):
        if cache_kwargs is None:
            cache_kwargs = {}
        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)
        keys, values = static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_kwargs.get("cache_position"),
        )
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values
        return keys, values

    if hasattr(cu, "StaticLayer"):
        cu.StaticLayer.update = update
    elif hasattr(cu, "_static_cache_update"):
        cu._static_cache_update = static_cache_update
        if hasattr(cu, "StaticCache"):
            cu.StaticCache.update = static_cache_update_method
    else:
        raise RuntimeError(
            "unsupported transformers StaticCache implementation"
        )


def _load_model_and_tokenizer(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    m.eval()
    return m, tok


def _load_synthetic_model(model_id: str, num_layers: int | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if num_layers is not None:
        cfg.num_hidden_layers = int(num_layers)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    finally:
        torch.set_default_dtype(old_dtype)
    m.to(dtype=torch.bfloat16)
    m.eval()
    return m, None


def _n_layer_config(cfg, num_layers: int):
    one = copy.deepcopy(cfg)
    one.num_hidden_layers = int(num_layers)
    return one


def _one_layer_config(cfg):
    return _n_layer_config(cfg, 1)


def _early_initialize_static_cache(past_kv, config, batch: int, device) -> None:
    if not hasattr(past_kv, "early_initialization"):
        return
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    num_kv_heads = getattr(
        config,
        "num_key_value_heads",
        config.num_attention_heads,
    )
    past_kv.early_initialization(
        batch,
        num_kv_heads,
        head_dim,
        torch.bfloat16,
        device,
    )


def _make_static_cache(
    static_cache_cls, config, max_cache_len: int, batch: int, device
):
    try:
        return static_cache_cls(
            config=config,
            max_batch_size=batch,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.bfloat16,
        )
    except TypeError:
        return static_cache_cls(config=config, max_cache_len=max_cache_len)


def _position_ids_for(
    inputs_embeds: torch.Tensor, cache_position: torch.Tensor | None
):
    if cache_position is not None:
        return cache_position.view(1, -1)
    bsz, seq_len = inputs_embeds.shape[:2]
    return (
        torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
        .view(1, -1)
        .expand(bsz, -1)
    )


def _manual_static_causal_mask(
    inputs_embeds: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if past_key_values is None:
        return None
    bsz, seq_len = inputs_embeds.shape[:2]
    kv_len = int(past_key_values.get_max_cache_shape())
    position_ids_i64 = position_ids.to(torch.long)
    key_pos = torch.arange(
        kv_len, dtype=torch.long, device=inputs_embeds.device
    ).view(1, 1, kv_len)
    query_pos = position_ids_i64.view(position_ids_i64.shape[0], seq_len, 1)
    allowed = key_pos <= query_pos
    if attention_mask is not None:
        key_allowed = attention_mask.to(torch.bool).view(bsz, 1, kv_len)
        allowed = allowed & key_allowed
    allowed = allowed.unsqueeze(1)
    if allowed.shape[0] != bsz:
        allowed = allowed.expand(bsz, 1, seq_len, kv_len)
    if os.environ.get("BUDDY_TTIR_MASK_AS_WHERE") == "1":
        shape = (bsz, 1, seq_len, kv_len)
        zero = torch.full(
            shape,
            0.0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        neg_inf = torch.full(
            shape,
            float("-inf"),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return torch.where(allowed, zero, neg_inf)
    allowed_f = allowed.to(inputs_embeds.dtype)
    return (1.0 - allowed_f) * torch.finfo(inputs_embeds.dtype).min


def _patch_llama_official_attention_f32() -> None:
    """Patch HF eager Llama attention to mirror official TTIR f32 semantics."""

    import math

    import torch.nn as nn
    from transformers.models.llama import modeling_llama

    def _buddy_official_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        del kwargs
        scale = math.sqrt(float(scaling))
        key_states = modeling_llama.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(
            value, module.num_key_value_groups
        )

        query_f32 = query.to(torch.float32) * scale
        key_f32 = key_states.to(torch.float32).transpose(2, 3) * scale
        attn_weights = torch.matmul(query_f32, key_f32)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(torch.float32)

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        )
        if dropout and module.training:
            attn_weights = nn.functional.dropout(
                attn_weights, p=dropout, training=module.training
            )

        value_f32 = value_states.to(torch.float32)
        attn_output = torch.matmul(attn_weights, value_f32).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_llama.eager_attention_forward = _buddy_official_attention_forward


class _OneLayerLMWrapper(torch.nn.Module):
    """Trace embed -> layer0 -> norm -> lm_head with a one-layer cache."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        position_ids = _position_ids_for(hidden_states, cache_position)
        if past_key_values is not None and cache_position is not None:
            for layer in getattr(past_key_values, "layers", []):
                layer.cumulative_length = cache_position
        causal_mask = _manual_static_causal_mask(
            hidden_states, past_key_values, position_ids, attention_mask
        )
        position_embeddings = model.rotary_emb(
            hidden_states, position_ids=position_ids
        )
        hidden_states = model.layers[0](
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = model.norm(hidden_states)
        logits = self.inner.lm_head(hidden_states)
        if use_cache:
            return logits, past_key_values
        return logits


class _Layer0StageProbeWrapper(torch.nn.Module):
    """Trace one layer and return major intermediate stage tensors."""

    def __init__(self, inner: torch.nn.Module, layer_index: int = 0) -> None:
        super().__init__()
        self.inner = inner
        self.layer_index = int(layer_index)
        self.config = _n_layer_config(inner.config, self.layer_index + 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        del kwargs
        model = self.inner.model
        layer = model.layers[self.layer_index]
        hidden_states = model.embed_tokens(input_ids)
        position_ids = _position_ids_for(hidden_states, cache_position)
        if past_key_values is not None and cache_position is not None:
            for cache_layer in getattr(past_key_values, "layers", []):
                cache_layer.cumulative_length = cache_position
        causal_mask = _manual_static_causal_mask(
            hidden_states, past_key_values, position_ids, attention_mask
        )
        position_embeddings = model.rotary_emb(
            hidden_states, position_ids=position_ids
        )

        for layer_id in range(self.layer_index):
            hidden_states = model.layers[layer_id](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        normed = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        after_attn = hidden_states + attn_out
        post_norm = layer.post_attention_layernorm(after_attn)
        mlp_out = layer.mlp(post_norm)
        layer_out = after_attn + mlp_out
        return normed, attn_out, after_attn, post_norm, mlp_out, layer_out


class _AttentionSplitProbeWrapper(torch.nn.Module):
    """Trace one layer's attention internals up to the residual add."""

    def __init__(self, inner: torch.nn.Module, layer_index: int = 0) -> None:
        super().__init__()
        self.inner = inner
        self.layer_index = int(layer_index)
        self.config = _n_layer_config(inner.config, self.layer_index + 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        del kwargs
        model = self.inner.model
        layer = model.layers[self.layer_index]
        attn = layer.self_attn
        hidden_states = model.embed_tokens(input_ids)
        position_ids = _position_ids_for(hidden_states, cache_position)
        if past_key_values is not None and cache_position is not None:
            for cache_layer in getattr(past_key_values, "layers", []):
                cache_layer.cumulative_length = cache_position
        causal_mask = _manual_static_causal_mask(
            hidden_states, past_key_values, position_ids, attention_mask
        )
        position_embeddings = model.rotary_emb(
            hidden_states, position_ids=position_ids
        )

        for layer_id in range(self.layer_index):
            hidden_states = model.layers[layer_id](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        normed = layer.input_layernorm(hidden_states)
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_linear = attn.q_proj(normed)
        key_linear = attn.k_proj(normed)
        value_linear = attn.v_proj(normed)
        query_states = query_linear.view(hidden_shape).transpose(1, 2)
        key_states = key_linear.view(hidden_shape).transpose(1, 2)
        value_states = value_linear.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        query_rope = (
            query_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx
            )
        key_rope = (
            key_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )
        value_cache = (
            value_states.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )

        key_repeated = modeling_llama.repeat_kv(
            key_states, attn.num_key_value_groups
        )
        value_repeated = modeling_llama.repeat_kv(
            value_states, attn.num_key_value_groups
        )
        attn_heads = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_repeated,
            value_repeated,
            attn_mask=causal_mask,
            dropout_p=0.0 if not attn.training else attn.attention_dropout,
            scale=attn.scaling,
            is_causal=False,
        )
        attn_heads = attn_heads.transpose(1, 2).contiguous()
        attn_core = attn_heads.reshape(*input_shape, -1).contiguous()
        attn_out = attn.o_proj(attn_core)
        after_attn = hidden_states + attn_out
        return (
            normed,
            query_linear,
            key_linear,
            value_linear,
            query_rope,
            key_rope,
            value_cache,
            attn_core,
            attn_out,
            after_attn,
        )


class _OneBlockWrapper(torch.nn.Module):
    """Trace only decoder layer0 from hidden states."""

    def __init__(
        self, inner: torch.nn.Module, zero_position_ids: bool = False
    ) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)
        self.zero_position_ids = zero_position_ids

    def _zero_position_embeddings(self, hidden_states: torch.Tensor):
        model = self.inner.model
        bsz, seq_len = hidden_states.shape[:2]
        inv_freq = model.rotary_emb.inv_freq
        inv_freq_expanded = inv_freq.reshape(1, inv_freq.shape[0], 1)
        inv_freq_expanded = inv_freq_expanded.to(torch.float32).expand(
            bsz, -1, 1
        )
        position = torch.full(
            (bsz, 1, seq_len),
            0.0,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        freqs = torch.matmul(inv_freq_expanded, position)
        freqs = freqs.reshape(bsz, seq_len, inv_freq.shape[0])
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(dtype=hidden_states.dtype)
        sin = torch.sin(emb).to(dtype=hidden_states.dtype)
        return cos, sin

    def forward(self, hidden_states: torch.Tensor):
        model = self.inner.model
        if self.zero_position_ids:
            position_ids = None
            position_embeddings = self._zero_position_embeddings(hidden_states)
        else:
            position_ids = _position_ids_for(hidden_states, None)
            position_embeddings = model.rotary_emb(
                hidden_states, position_ids=position_ids
            )
        return model.layers[0](
            hidden_states,
            attention_mask=None,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )


class _KProjProbeWrapper(torch.nn.Module):
    """Trace embed -> layer0 input RMSNorm -> K projection."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        layer = model.layers[0]
        hidden_states = model.embed_tokens(input_ids)
        hidden_states = layer.input_layernorm(hidden_states)
        if os.environ.get("BUDDY_LLAMA31_KPROJ_CLONE_NORMED") == "1":
            hidden_states = hidden_states.clone()
        return layer.self_attn.k_proj(hidden_states)


class _RMSNormKProjProbeWrapper(torch.nn.Module):
    """Trace layer0 input RMSNorm and its downstream K projection."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        layer = model.layers[0]
        hidden_states = model.embed_tokens(input_ids)
        hidden_states = layer.input_layernorm(hidden_states)
        return hidden_states, layer.self_attn.k_proj(hidden_states)


class _QKVProjInputProbeWrapper(torch.nn.Module):
    """Trace layer0 Q/K/V projections from graph-input hidden states."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        attn = self.inner.model.layers[0].self_attn
        return (
            attn.q_proj(hidden_states),
            attn.k_proj(hidden_states),
            attn.v_proj(hidden_states),
        )


class _RoPEInputProbeWrapper(torch.nn.Module):
    """Trace RoPE from runtime packed Q/K and cos/sin inputs."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(
        self,
        query_packed: torch.Tensor,
        key_packed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        attn = self.inner.model.layers[0].self_attn
        bsz, seq_len, _ = query_packed.shape
        query_states = query_packed.view(
            bsz, seq_len, self.config.num_attention_heads, attn.head_dim
        ).transpose(1, 2)
        key_states = key_packed.view(
            bsz, seq_len, self.config.num_key_value_heads, attn.head_dim
        ).transpose(1, 2)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        query_out = (
            query_states.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        )
        key_out = (
            key_states.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        )
        return query_out, key_out


class _MLPInputProbeWrapper(torch.nn.Module):
    """Trace one layer's MLP from graph-input hidden states."""

    def __init__(self, inner: torch.nn.Module, layer_index: int = 0) -> None:
        super().__init__()
        self.inner = inner
        self.layer_index = int(layer_index)
        self.config = _n_layer_config(inner.config, self.layer_index + 1)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        return self.inner.model.layers[self.layer_index].mlp(hidden_states)


class _MLPSplitProbeWrapper(torch.nn.Module):
    """Trace one layer's MLP internals from graph-input hidden states."""

    def __init__(self, inner: torch.nn.Module, layer_index: int = 0) -> None:
        super().__init__()
        self.inner = inner
        self.layer_index = int(layer_index)
        self.config = _n_layer_config(inner.config, self.layer_index + 1)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        mlp = self.inner.model.layers[self.layer_index].mlp
        gate = mlp.gate_proj(hidden_states)
        silu_gate = mlp.act_fn(gate)
        up = mlp.up_proj(hidden_states)
        product = silu_gate * up
        down = mlp.down_proj(product)
        return silu_gate, up, product, down


class _MLPDownInputProbeWrapper(torch.nn.Module):
    """Trace one layer's MLP down projection from graph-input product states."""

    def __init__(self, inner: torch.nn.Module, layer_index: int = 0) -> None:
        super().__init__()
        self.inner = inner
        self.layer_index = int(layer_index)
        self.config = _n_layer_config(inner.config, self.layer_index + 1)

    def forward(self, product_states: torch.Tensor, **kwargs):
        del kwargs
        return self.inner.model.layers[self.layer_index].mlp.down_proj(
            product_states
        )


class _PostAttnNormInputProbeWrapper(torch.nn.Module):
    """Trace layer0 post-attention RMSNorm from graph-input hidden states."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        return self.inner.model.layers[0].post_attention_layernorm(
            hidden_states
        )


class _SDPAInputProbeWrapper(torch.nn.Module):
    """Trace fused masked SDPA from runtime Q/K/V inputs."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)
        mask = torch.zeros((1, 1, 1024, 1024), dtype=torch.bfloat16)
        upper = torch.triu(
            torch.ones((1024, 1024), dtype=torch.bool), diagonal=1
        )
        mask = mask.masked_fill(
            upper.reshape(1, 1, 1024, 1024), torch.finfo(torch.bfloat16).min
        )
        self.register_buffer("attention_mask", mask, persistent=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        attn = self.inner.model.layers[0].self_attn
        attention_mask = self.attention_mask[
            :, :, : query_states.shape[2], : key_states.shape[2]
        ]
        attn_heads = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            is_causal=False,
        )
        attn_heads = attn_heads.transpose(1, 2).contiguous()
        return attn_heads.reshape(
            query_states.shape[0], query_states.shape[2], -1
        )


class _SDPAPackedInputProbeWrapper(torch.nn.Module):
    """Trace masked SDPA from runtime packed 3D Q/K/V inputs."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)
        mask = torch.zeros((1, 1, 1024, 1024), dtype=torch.bfloat16)
        upper = torch.triu(
            torch.ones((1024, 1024), dtype=torch.bool), diagonal=1
        )
        mask = mask.masked_fill(
            upper.reshape(1, 1, 1024, 1024), torch.finfo(torch.bfloat16).min
        )
        self.register_buffer("attention_mask", mask, persistent=False)

    def forward(
        self,
        query_packed: torch.Tensor,
        key_packed: torch.Tensor,
        value_packed: torch.Tensor,
    ):
        attn = self.inner.model.layers[0].self_attn
        bsz, seq_len, _ = query_packed.shape
        num_heads = self.config.num_attention_heads
        query_states = query_packed.view(
            bsz, seq_len, num_heads, attn.head_dim
        ).transpose(1, 2)
        key_states = key_packed.view(
            bsz, seq_len, num_heads, attn.head_dim
        ).transpose(1, 2)
        value_states = value_packed.view(
            bsz, seq_len, num_heads, attn.head_dim
        ).transpose(1, 2)
        attention_mask = self.attention_mask[:, :, :seq_len, :seq_len]
        attn_heads = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=attn.scaling,
            is_causal=False,
        )
        attn_heads = attn_heads.transpose(1, 2).contiguous()
        return attn_heads.reshape(bsz, seq_len, -1)


class _EmbedProbeWrapper(torch.nn.Module):
    """Trace token embedding only."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        return self.inner.model.embed_tokens(input_ids)


class _EmbedRoundTripProbeWrapper(torch.nn.Module):
    """Trace embedding followed by the bf16 -> f32 -> bf16 path."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        hidden_states = self.inner.model.embed_tokens(input_ids)
        return hidden_states.to(torch.float32).to(hidden_states.dtype)


class _EmbedWeightMulProbeWrapper(torch.nn.Module):
    """Trace embedding multiplied by layer0 RMSNorm weight only."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        return hidden_states * model.layers[0].input_layernorm.weight


class _RMSStatProbeWrapper(torch.nn.Module):
    """Trace the layer0 RMSNorm mean-square statistic."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        hidden_states = self.inner.model.embed_tokens(input_ids)
        hidden_f32 = hidden_states.to(torch.float32)
        return hidden_f32.pow(2).mean(-1).to(hidden_states.dtype)


class _RMSScaleProbeWrapper(torch.nn.Module):
    """Trace the layer0 RMSNorm rsqrt scale."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        hidden_f32 = hidden_states.to(torch.float32)
        var = hidden_f32.pow(2).mean(-1)
        scale = torch.rsqrt(
            var + model.layers[0].input_layernorm.variance_epsilon
        )
        return scale.to(hidden_states.dtype)


class _RMSUnweightedProbeWrapper(torch.nn.Module):
    """Trace layer0 RMSNorm before multiplying by the learned weight."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        hidden_f32 = hidden_states.to(torch.float32)
        var = hidden_f32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(
            var + model.layers[0].input_layernorm.variance_epsilon
        )
        return (hidden_f32 * scale).to(hidden_states.dtype)


class _RMSNormProbeWrapper(torch.nn.Module):
    """Trace token embedding followed by layer0 input RMSNorm."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = _one_layer_config(inner.config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        return model.layers[0].input_layernorm(hidden_states)


class _FullLMAlignedWrapper(torch.nn.Module):
    """Trace the full model with the same explicit pieces as the layer wrapper."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = inner.config

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        del kwargs
        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        position_ids = _position_ids_for(hidden_states, cache_position)
        if past_key_values is not None and cache_position is not None:
            for layer in getattr(past_key_values, "layers", []):
                layer.cumulative_length = cache_position
        causal_mask = _manual_static_causal_mask(
            hidden_states, past_key_values, position_ids, attention_mask
        )
        position_embeddings = model.rotary_emb(
            hidden_states, position_ids=position_ids
        )
        for decoder_layer in model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        hidden_states = model.norm(hidden_states)
        logits = self.inner.lm_head(hidden_states)
        if use_cache:
            return logits, past_key_values
        return logits


class _ArgmaxHeadWrapper(torch.nn.Module):
    """nn.Module that runs ``self.inner(...)`` then emits argmax(logits, -1).

    Returns the int64 token id tensor of shape ``[B, S]`` as the first output;
    everything else (``past_key_values`` slots) is yielded by the underlying
    HF model and Dynamo flattens it into the rest of the graph outputs. The
    full bf16 logits (~256 KB/decode step, ~256 MB/prefill step) never leaves
    the device this way.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = inner.config

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        logits = out.logits
        token_ids = torch.argmax(logits, dim=-1)
        return token_ids, out.past_key_values


def _patch_gqa_cur_pos(graph, name: str) -> None:
    if not name:
        return
    for op in graph.body:
        if isinstance(op, GQAAttentionFusedOp):
            op.kwargs["cur_pos_tensor"] = name


def _split_mlir_args(args_text: str) -> list[str]:
    args: list[str] = []
    cur: list[str] = []
    angle_depth = 0
    brace_depth = 0
    for ch in args_text:
        if ch == "<":
            angle_depth += 1
        elif ch == ">" and angle_depth:
            angle_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth:
            brace_depth -= 1
        if ch == "," and angle_depth == 0 and brace_depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _prune_unused_ttir_args(mlir_text: str) -> str:
    """Remove unused `%argN` entries from the generated single-function TTIR."""
    m = re.search(r"(func\.func @\w+\()(?P<args>.*?)(\) -> )", mlir_text, re.S)
    if not m:
        return mlir_text
    body_start = mlir_text.find("{", m.end())
    if body_start < 0:
        return mlir_text
    sig_args = _split_mlir_args(m.group("args"))
    if not sig_args:
        return mlir_text

    body = mlir_text[body_start:]
    used = {int(x) for x in re.findall(r"%arg(\d+)(?!\d)", body)}
    parsed: list[tuple[int, str]] = []
    for arg in sig_args:
        am = re.match(r"%arg(\d+):\s*(.*)", arg, re.S)
        if am is None:
            return mlir_text
        parsed.append((int(am.group(1)), am.group(2).strip()))
    keep = [(old, ty) for old, ty in parsed if old in used]
    if len(keep) == len(parsed):
        return mlir_text

    remap = {old: new for new, (old, _) in enumerate(keep)}
    new_args = ", ".join(f"%arg{new}: {ty}" for new, (_, ty) in enumerate(keep))

    def repl(match):
        old = int(match.group(1))
        if old not in remap:
            return match.group(0)
        return f"%arg{remap[old]}"

    prefix = (
        mlir_text[: m.start("args")]
        + new_args
        + mlir_text[m.end("args") : body_start]
    )
    body = re.sub(r"%arg(\d+)(?!\d)", repl, mlir_text[body_start:])
    return prefix + body


def _keep_decode_cache_mask_compares_f32(
    mlir_text: str, max_cache_len: int
) -> tuple[str, int]:
    """Keep decode KV-cache position masks out of bf16.

    Dynamo lowers the scatter-style static cache update as:

      arange(max_cache_len) == cache_position

    and also derives causal attention masks as:

      arange(max_cache_len) <= cache_position

    These comparisons must stay exact. If both sides are converted to bf16,
    positions beyond 256 can alias and long decoding corrupts the KV cache or
    attention mask. Rewrite the older where-mask ladder here; the broader
    position-index pass below handles the current causal-mask shape.
    """

    typecast_arange_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.typecast"'
        r"\((?P<src>%\d+)\) <\{conservative_folding = false\}> : "
        r"\(tensor<(?P<len>\d+)xf32>\) -> tensor<(?P=len)xbf16>$"
    )
    reshape_arange_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.reshape"'
        r"\((?P<src>%\d+)\) <\{shape = \[1 : i32, (?P<len>\d+) : i32\]\}> : "
        r"\(tensor<(?P=len)xbf16>\) -> tensor<1x(?P=len)xbf16>$"
    )
    typecast_pos0_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.typecast"'
        r"\((?P<src>%arg\d+)\) <\{conservative_folding = false\}> : "
        r"\(tensor<1xi64>\) -> tensor<1xbf16>$"
    )
    slice_pos_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.slice_static"'
        r"\((?P<src>%\d+)\) <\{begins = \[0 : i32\], "
        r"ends = \[1 : i32\], step = \[1 : i32\]\}> : "
        r"\(tensor<1xbf16>\) -> tensor<1xbf16>$"
    )
    typecast_pos1_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.typecast"'
        r"\((?P<src>%\d+)\) <\{conservative_folding = false\}> : "
        r"\(tensor<1xbf16>\) -> tensor<1xf32>$"
    )
    typecast_pos2_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.typecast"'
        r"\((?P<src>%\d+)\) <\{conservative_folding = false\}> : "
        r"\(tensor<1xf32>\) -> tensor<1xbf16>$"
    )
    reshape_pos_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.reshape"'
        r"\((?P<src>%\d+)\) <\{shape = \[1 : i32, 1 : i32\]\}> : "
        r"\(tensor<1xbf16>\) -> tensor<1x1xbf16>$"
    )
    eq_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%\d+) = "ttir\.eq"'
        r"\((?P<lhs>%\d+), (?P<rhs>%\d+)\) : "
        r"\(tensor<1x(?P<len>\d+)xbf16>, tensor<1x1xbf16>\) "
        r"-> tensor<1x(?P=len)xi1>$"
    )

    lines = mlir_text.splitlines()
    patched = 0
    i = 0
    while i + 7 < len(lines):
        m0 = typecast_arange_re.match(lines[i])
        if m0 is None or int(m0.group("len")) != max_cache_len:
            i += 1
            continue
        m1 = reshape_arange_re.match(lines[i + 1])
        m2 = typecast_pos0_re.match(lines[i + 2])
        m3 = slice_pos_re.match(lines[i + 3])
        m4 = typecast_pos1_re.match(lines[i + 4])
        m5 = typecast_pos2_re.match(lines[i + 5])
        m6 = reshape_pos_re.match(lines[i + 6])
        m7 = eq_re.match(lines[i + 7])
        if not all((m1, m2, m3, m4, m5, m6, m7)):
            i += 1
            continue
        if (
            int(m1.group("len")) != max_cache_len
            or int(m7.group("len")) != max_cache_len
        ):
            i += 1
            continue
        if (
            m1.group("src") != m0.group("dst")
            or m3.group("src") != m2.group("dst")
            or m4.group("src") != m3.group("dst")
            or m5.group("src") != m4.group("dst")
            or m6.group("src") != m5.group("dst")
            or m7.group("lhs") != m1.group("dst")
            or m7.group("rhs") != m6.group("dst")
        ):
            i += 1
            continue

        indent = m0.group("indent")
        arange = m0.group("src")
        tc_arange = m0.group("dst")
        reshape_arange = m1.group("dst")
        pos_arg = m2.group("src")
        tc_pos0 = m2.group("dst")
        slice_pos = m3.group("dst")
        tc_pos1 = m4.group("dst")
        tc_pos2 = m5.group("dst")
        reshape_pos = m6.group("dst")
        eq = m7.group("dst")
        dim = str(max_cache_len)

        lines[i] = (
            f'{indent}{tc_arange} = "ttir.typecast"({arange}) '
            f"<{{conservative_folding = false}}> : (tensor<{dim}xf32>) "
            f"-> tensor<{dim}xbf16>"
        )
        lines[i + 1] = (
            f'{indent}{reshape_arange} = "ttir.reshape"({arange}) '
            f"<{{shape = [1 : i32, {dim} : i32]}}> : "
            f"(tensor<{dim}xf32>) -> tensor<1x{dim}xf32>"
        )
        lines[i + 2] = (
            f'{indent}{tc_pos0} = "ttir.typecast"({pos_arg}) '
            "<{conservative_folding = false}> : (tensor<1xi64>) "
            "-> tensor<1xf32>"
        )
        lines[i + 3] = (
            f'{indent}{slice_pos} = "ttir.slice_static"({tc_pos0}) '
            "<{begins = [0 : i32], ends = [1 : i32], "
            "step = [1 : i32]}> : (tensor<1xf32>) -> tensor<1xf32>"
        )
        lines[i + 4] = (
            f'{indent}{tc_pos1} = "ttir.typecast"({slice_pos}) '
            "<{conservative_folding = false}> : (tensor<1xf32>) "
            "-> tensor<1xf32>"
        )
        lines[i + 5] = (
            f'{indent}{tc_pos2} = "ttir.typecast"({tc_pos1}) '
            "<{conservative_folding = false}> : (tensor<1xf32>) "
            "-> tensor<1xf32>"
        )
        lines[i + 6] = (
            f'{indent}{reshape_pos} = "ttir.reshape"({tc_pos2}) '
            "<{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>) "
            "-> tensor<1x1xf32>"
        )
        lines[i + 7] = (
            f'{indent}{eq} = "ttir.eq"({reshape_arange}, {reshape_pos}) : '
            f"(tensor<1x{dim}xf32>, tensor<1x1xf32>) "
            f"-> tensor<1x{dim}xi1>"
        )
        patched += 1
        i += 8

    return "\n".join(lines), patched


def _keep_position_index_math_f32(mlir_text: str) -> tuple[str, int]:
    """Keep positional index math out of bf16.

    The full static-cache graphs derive RoPE positions and causal masks from
    arange/cache_position tensors. Lowering everything to bf16 makes integer
    positions beyond 256 lossy, which corrupts RoPE angles and long-context
    attention masks. This pass keeps only that small index/position subgraph
    in f32; model activations still lower to bf16.
    """

    def _bf16_to_f32_types(line: str) -> str:
        return re.sub(r"tensor<([^>]*)xbf16>", r"tensor<\1xf32>", line)

    def _def_name(line: str) -> str | None:
        m = re.match(r"^\s+(%\d+) = ", line)
        return m.group(1) if m else None

    def _operands(line: str) -> list[str]:
        m = re.search(r'=\s+"ttir\.[^"]+"\(([^)]*)\)', line)
        if not m:
            return []
        return re.findall(r"%arg\d+|%\d+", m.group(1))

    def _op_name(line: str) -> str | None:
        m = re.search(r'=\s+"ttir\.([^"]+)"', line)
        return m.group(1) if m else None

    seed_i64_to_bf16_re = re.compile(
        r"(\(tensor<[^>]+xi64>\) -> tensor<[^>]+)xbf16"
    )
    seed_inv_freq_re = re.compile(r"(\(tensor<64xf32>\) -> tensor<64x)bf16")

    f32_values: set[str] = set()
    patched = 0
    lines: list[str] = []
    propagate_ops = {
        "add",
        "broadcast",
        "concat",
        "matmul",
        "permute",
        "reshape",
        "slice_static",
        "typecast",
    }
    signature_only_ops = {"le", "eq"}

    for raw in mlir_text.splitlines():
        line = raw
        dst = _def_name(line)
        op = _op_name(line)
        operands = _operands(line)
        seeded = False

        if dst and ' = "ttir.typecast"' in line:
            updated = seed_i64_to_bf16_re.sub(r"\1xf32", line)
            updated = seed_inv_freq_re.sub(r"\1f32", updated)
            if updated != line:
                line = updated
                f32_values.add(dst)
                patched += 1
                seeded = True

        if dst and not seeded and any(o in f32_values for o in operands):
            if op in propagate_ops:
                updated = _bf16_to_f32_types(line)
                if updated != line:
                    line = updated
                    patched += 1
                f32_values.add(dst)
            elif op in signature_only_ops:
                updated = _bf16_to_f32_types(line)
                if updated != line:
                    line = updated
                    patched += 1

        lines.append(line)

    return "\n".join(lines), patched


def _legalize_rank_expanding_broadcasts(mlir_text: str) -> tuple[str, int]:
    """Insert leading-1 reshapes before rank-expanding ``ttir.broadcast`` ops."""

    def _parse_tensor_type(text: str) -> tuple[list[int], str] | None:
        parts = text.split("x")
        if not parts:
            return None
        dtype = parts[-1]
        dims: list[int] = []
        for part in parts[:-1]:
            if not part.isdigit():
                return None
            dims.append(int(part))
        return dims, dtype

    def _tensor_type(dims: list[int], dtype: str) -> str:
        if dims:
            return (
                "tensor<" + "x".join(str(d) for d in dims) + "x" + dtype + ">"
            )
        return f"tensor<{dtype}>"

    broadcast_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%[\w$._-]+) = "ttir\.broadcast"'
        r"\((?P<src>%[\w$._-]+)\) "
        r"<\{broadcast_dimensions = array<i64: (?P<dims>[^>]*)>\}> : "
        r"\(tensor<(?P<input>[^>]*)>\) -> tensor<(?P<output>[^>]*)>$"
    )

    lines: list[str] = []
    patched = 0
    for raw in mlir_text.splitlines():
        m = broadcast_re.match(raw)
        if m is None:
            lines.append(raw)
            continue

        input_parsed = _parse_tensor_type(m.group("input"))
        output_parsed = _parse_tensor_type(m.group("output"))
        if input_parsed is None or output_parsed is None:
            lines.append(raw)
            continue
        input_dims, input_dtype = input_parsed
        output_dims, output_dtype = output_parsed
        if input_dtype != output_dtype or len(input_dims) >= len(output_dims):
            lines.append(raw)
            continue

        padded_input_dims = [1] * (
            len(output_dims) - len(input_dims)
        ) + input_dims
        padded_type = _tensor_type(padded_input_dims, input_dtype)
        input_type = _tensor_type(input_dims, input_dtype)
        shape_attr = ", ".join(f"{dim} : i32" for dim in padded_input_dims)
        reshape_name = f"%buddy_broadcast_rank_{patched}"
        indent = m.group("indent")

        lines.append(
            f'{indent}{reshape_name} = "ttir.reshape"({m.group("src")}) '
            f"<{{shape = [{shape_attr}]}}> : ({input_type}) -> {padded_type}"
        )
        lines.append(
            f'{indent}{m.group("dst")} = "ttir.broadcast"({reshape_name}) '
            f"<{{broadcast_dimensions = array<i64: {m.group('dims')}>}}> : "
            f"({padded_type}) -> tensor<{m.group('output')}>"
        )
        patched += 1

    return "\n".join(lines), patched


def _keep_update_cache_indices_i32(mlir_text: str) -> tuple[str, int]:
    """Keep ``ttir.update_cache`` update indices as integer tensors.

    Full-model decode uses the same cache position value for RoPE/mask math
    and KV cache updates. The former may be deliberately cast to bf16/f32, but
    TTNN's ``paged_update_cache`` lowering requires the update index operand to
    be integer. Rebuild just the update-cache index chain as i32.
    """

    def _def_name(line: str) -> str | None:
        m = re.match(r"^\s+(%\d+) = ", line)
        return m.group(1) if m else None

    def _operands(line: str) -> list[str]:
        m = re.search(r'=\s+"ttir\.[^"]+"\(([^)]*)\)', line)
        if not m:
            return []
        return re.findall(r"%arg\d+|%\d+", m.group(1))

    update_re = re.compile(
        r'=\s+"ttir\.update_cache"\([^,]+,\s*[^,]+,\s*(%\d+)\)'
    )
    lines = mlir_text.splitlines()
    needed: set[str] = set()
    patched = 0

    for line in lines:
        m = update_re.search(line)
        if m and ("tensor<1xbf16>) ->" in line or "tensor<1xf32>) ->" in line):
            needed.add(m.group(1))

    if not needed:
        return mlir_text, 0

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        dst = _def_name(line)
        if dst not in needed:
            continue

        if ' = "ttir.slice_static"' in line:
            operands = _operands(line)
            if operands:
                needed.add(operands[0])
            updated = re.sub(r"tensor<1x(?:bf16|f32)>", "tensor<1xi32>", line)
        elif ' = "ttir.typecast"' in line and (
            "(tensor<1xi64>) -> tensor<1xbf16>" in line
            or "(tensor<1xi64>) -> tensor<1xf32>" in line
        ):
            updated = re.sub(
                r"\(tensor<1xi64>\) -> tensor<1x(?:bf16|f32)>",
                "(tensor<1xi64>) -> tensor<1xi32>",
                line,
            )
        else:
            continue

        if updated != line:
            lines[i] = updated
            patched += 1

    for i, line in enumerate(lines):
        if ' = "ttir.update_cache"' not in line:
            continue
        m = update_re.search(line)
        if not m or m.group(1) not in needed:
            continue
        updated = re.sub(
            r"tensor<1x(?:bf16|f32)>\) ->", "tensor<1xi32>) ->", line
        )
        if updated != line:
            lines[i] = updated
            patched += 1

    return "\n".join(lines), patched


def _legalize_batched_update_cache_inputs(mlir_text: str) -> tuple[str, int]:
    """Adapt batch-major update tensors to TTIR UpdateCacheOp's verifier shape.

    ``ttir.update_cache`` wants the update value in [1, heads, users, dim]
    form before canonicalization to paged_update_cache. HuggingFace static
    cache traces naturally produce [users, heads, 1, dim].
    """

    def _parse_tensor_type(text: str) -> tuple[list[int], str] | None:
        parts = text.split("x")
        if not parts:
            return None
        dtype = parts[-1]
        dims: list[int] = []
        for part in parts[:-1]:
            if not part.isdigit():
                return None
            dims.append(int(part))
        return dims, dtype

    def _tensor_type(dims: list[int], dtype: str) -> str:
        return "tensor<" + "x".join(str(d) for d in dims) + "x" + dtype + ">"

    update_re = re.compile(
        r'^(?P<indent>\s+)(?P<dst>%[\w$._-]+) = "ttir\.update_cache"'
        r"\((?P<cache>%[\w$._-]+), (?P<input>%[\w$._-]+), "
        r"(?P<index>%[\w$._-]+)\) (?P<attrs><\{[^}]*\}>) : "
        r"\(tensor<(?P<cache_ty>[^>]*)>, tensor<(?P<input_ty>[^>]*)>, "
        r"tensor<(?P<index_ty>[^>]*)>\) -> tensor<(?P<out_ty>[^>]*)>$"
    )

    lines: list[str] = []
    patched = 0
    for raw in mlir_text.splitlines():
        m = update_re.match(raw)
        if m is None:
            lines.append(raw)
            continue

        cache_parsed = _parse_tensor_type(m.group("cache_ty"))
        input_parsed = _parse_tensor_type(m.group("input_ty"))
        if cache_parsed is None or input_parsed is None:
            lines.append(raw)
            continue
        cache_dims, cache_dtype = cache_parsed
        input_dims, input_dtype = input_parsed
        input_name = m.group("input")
        input_type = _tensor_type(input_dims, input_dtype)
        indent = m.group("indent")

        if (
            cache_dtype == input_dtype
            and len(cache_dims) == 4
            and len(input_dims) == 3
            and input_dims[0] > 1
            and cache_dims[0] == input_dims[0]
            and cache_dims[1] == input_dims[1]
            and cache_dims[3] == input_dims[2]
        ):
            users, heads, head_dim = input_dims
            expanded_dims = [users, heads, 1, head_dim]
            expanded_type = _tensor_type(expanded_dims, input_dtype)
            expanded_name = f"%buddy_update_cache_step_{patched}"
            lines.append(
                f'{indent}{expanded_name} = "ttir.reshape"({input_name}) '
                f"<{{shape = [{users} : i32, {heads} : i32, "
                f"1 : i32, {head_dim} : i32]}}> : "
                f"({input_type}) -> {expanded_type}"
            )
            input_dims = expanded_dims
            input_name = expanded_name
            input_type = expanded_type

        if (
            cache_dtype != input_dtype
            or len(cache_dims) != 4
            or len(input_dims) != 4
            or input_dims[0] <= 1
            or input_dims[2] != 1
            or cache_dims[0] != input_dims[0]
            or cache_dims[1] != input_dims[1]
            or cache_dims[3] != input_dims[3]
        ):
            lines.append(raw)
            continue

        users, heads, _, head_dim = input_dims
        permuted_dims = [1, heads, users, head_dim]
        permuted_type = _tensor_type(permuted_dims, input_dtype)
        permuted_name = f"%buddy_update_cache_batch_{patched}"

        lines.append(
            f'{indent}{permuted_name} = "ttir.permute"({input_name}) '
            f"<{{permutation = array<i64: 2, 1, 0, 3>}}> : "
            f"({input_type}) -> {permuted_type}"
        )
        lines.append(
            f'{indent}{m.group("dst")} = "ttir.update_cache"'
            f"({m.group('cache')}, {permuted_name}, {m.group('index')}) "
            f"{m.group('attrs')} : "
            f"(tensor<{m.group('cache_ty')}>, {permuted_type}, "
            f"tensor<{m.group('index_ty')}>) -> tensor<{m.group('out_ty')}>"
        )
        patched += 1

    return "\n".join(lines), patched


def _legalize_batched_sdpa_decode_inputs(mlir_text: str) -> tuple[str, int]:
    """Adapt batch-major decode SDPA tensors to TTIR's decode layout."""

    def _parse_tensor_type(text: str) -> tuple[list[int], str] | None:
        parts = text.split("x")
        if not parts:
            return None
        dtype = parts[-1]
        dims: list[int] = []
        for part in parts[:-1]:
            if not part.isdigit():
                return None
            dims.append(int(part))
        return dims, dtype

    def _tensor_type(dims: list[int], dtype: str) -> str:
        return "tensor<" + "x".join(str(d) for d in dims) + "x" + dtype + ">"

    sdpa_re = re.compile(
        r"^(?P<indent>\s+)(?P<dst>%[\w$._-]+) = "
        r'"ttir\.scaled_dot_product_attention_decode"'
        r"\((?P<query>%[\w$._-]+), (?P<key>%[\w$._-]+), "
        r"(?P<value>%[\w$._-]+), (?P<mask>%[\w$._-]+), "
        r"(?P<pos>%[\w$._-]+)\) (?P<attrs><\{.*\}>) : "
        r"\(tensor<(?P<query_ty>[^>]*)>, tensor<(?P<key_ty>[^>]*)>, "
        r"tensor<(?P<value_ty>[^>]*)>, tensor<(?P<mask_ty>[^>]*)>, "
        r"tensor<(?P<pos_ty>[^>]*)>\) -> tensor<(?P<out_ty>[^>]*)>$"
    )

    lines: list[str] = []
    patched = 0
    for raw in mlir_text.splitlines():
        m = sdpa_re.match(raw)
        if m is None:
            lines.append(raw)
            continue

        query_parsed = _parse_tensor_type(m.group("query_ty"))
        out_parsed = _parse_tensor_type(m.group("out_ty"))
        pos_parsed = _parse_tensor_type(m.group("pos_ty"))
        if query_parsed is None or out_parsed is None or pos_parsed is None:
            lines.append(raw)
            continue

        query_dims, query_dtype = query_parsed
        out_dims, out_dtype = out_parsed
        pos_dims, pos_dtype = pos_parsed
        if (
            len(query_dims) != 4
            or query_dims[0] <= 1
            or query_dims[2] != 1
            or query_dims != out_dims
            or query_dtype != out_dtype
            or pos_dims != [1]
        ):
            lines.append(raw)
            continue

        batch, heads, _, head_dim = query_dims
        legal_dims = [1, batch, heads, head_dim]
        query_type = _tensor_type(query_dims, query_dtype)
        legal_type = _tensor_type(legal_dims, query_dtype)
        pos_type = _tensor_type(pos_dims, pos_dtype)
        batch_pos_type = _tensor_type([batch], pos_dtype)
        indent = m.group("indent")
        query_name = f"%buddy_sdpa_decode_query_{patched}"
        pos_name = f"%buddy_sdpa_decode_pos_{patched}"
        sdpa_name = f"%buddy_sdpa_decode_out_{patched}"

        lines.append(
            f'{indent}{query_name} = "ttir.permute"({m.group("query")}) '
            f"<{{permutation = array<i64: 2, 0, 1, 3>}}> : "
            f"({query_type}) -> {legal_type}"
        )
        lines.append(
            f'{indent}{pos_name} = "ttir.broadcast"({m.group("pos")}) '
            f"<{{broadcast_dimensions = array<i64: {batch}>}}> : "
            f"({pos_type}) -> {batch_pos_type}"
        )
        lines.append(
            f'{indent}{sdpa_name} = "ttir.scaled_dot_product_attention_decode"'
            f"({query_name}, {m.group('key')}, {m.group('value')}, "
            f"{m.group('mask')}, {pos_name}) {m.group('attrs')} : "
            f"({legal_type}, tensor<{m.group('key_ty')}>, "
            f"tensor<{m.group('value_ty')}>, tensor<{m.group('mask_ty')}>, "
            f"{batch_pos_type}) -> {legal_type}"
        )
        lines.append(
            f'{indent}{m.group("dst")} = "ttir.permute"({sdpa_name}) '
            f"<{{permutation = array<i64: 1, 2, 0, 3>}}> : "
            f"({legal_type}) -> tensor<{m.group('out_ty')}>"
        )
        patched += 1

    return "\n".join(lines), patched


def _full_layer_param_attrs(layer_idx: int) -> list[tuple[str, str]]:
    prefix = f"l__self___model_layers__modules__{layer_idx}___"
    return [
        ("parameter", prefix + "input_layernorm_weight"),
        ("parameter", prefix + "self_attn_q_proj_weight"),
        ("parameter", prefix + "self_attn_k_proj_weight"),
        ("parameter", prefix + "self_attn_v_proj_weight"),
        ("parameter", prefix + "self_attn_o_proj_weight"),
        ("parameter", prefix + "post_attention_layernorm_weight"),
        ("parameter", prefix + "mlp_gate_proj_weight"),
        ("parameter", prefix + "mlp_up_proj_weight"),
        ("parameter", prefix + "mlp_down_proj_weight"),
    ]


def _full_arg_attrs(mode: str, arg_count: int) -> list[tuple[str, str]]:
    """Official-style metadata for full static-cache Llama scopes.

    There is no single official full-model fixture in the local tt-mlir tree,
    so this table follows the same role semantics proven by the layer/block
    gates: model weights are parameters, RoPE inv_freq is a constant, and token
    ids/cache tensors/cache positions are runtime inputs.
    """

    attrs: list[tuple[str, str]] = [
        ("parameter", "l__self___model_embed_tokens_weight"),
        ("input", "args_0"),
    ]

    if mode == "prefill":
        layers_num, rem = divmod(arg_count - 6, 11)
        if rem == 0 and layers_num > 0:
            attrs.extend(
                [
                    ("input", "args_1_cache_position"),
                    ("constant", "l__self___model_rotary_emb_inv_freq"),
                ]
            )
            for layer_idx in range(layers_num):
                prefix = f"l__self___model_layers__modules__{layer_idx}___"
                attrs.extend(
                    [
                        ("parameter", prefix + "input_layernorm_weight"),
                        ("parameter", prefix + "self_attn_q_proj_weight"),
                        ("parameter", prefix + "self_attn_k_proj_weight"),
                        ("parameter", prefix + "self_attn_v_proj_weight"),
                        ("input", f"args_key_cache_{layer_idx}"),
                        ("input", f"args_value_cache_{layer_idx}"),
                        ("parameter", prefix + "self_attn_o_proj_weight"),
                        (
                            "parameter",
                            prefix + "post_attention_layernorm_weight",
                        ),
                        ("parameter", prefix + "mlp_gate_proj_weight"),
                        ("parameter", prefix + "mlp_up_proj_weight"),
                        ("parameter", prefix + "mlp_down_proj_weight"),
                    ]
                )
            attrs.extend(
                [
                    ("parameter", "l__self___model_norm_weight"),
                    ("parameter", "l__self___lm_head_weight"),
                ]
            )
            if len(attrs) != arg_count:
                raise ValueError(
                    f"internal full-align prefill arg annotation mismatch: "
                    f"expected {arg_count}, built {len(attrs)}"
                )
            return attrs

        layers_num, rem = divmod(arg_count - 6, 9)
        if rem == 0 and layers_num > 0:
            attrs.append(("input", "args_1_attention_mask"))
            attrs.append(("constant", "l__self___model_rotary_emb_inv_freq"))
            for layer_idx in range(layers_num):
                attrs.extend(_full_layer_param_attrs(layer_idx))
            attrs.extend(
                [
                    ("parameter", "l__self___model_norm_weight"),
                    ("parameter", "l__self___lm_head_weight"),
                ]
            )
            if len(attrs) != arg_count:
                raise ValueError(
                    f"internal attention-mask prefill arg annotation mismatch: "
                    f"expected {arg_count}, built {len(attrs)}"
                )
            return attrs

        layers_num, rem = divmod(arg_count - 5, 9)
        if rem != 0 or layers_num <= 0:
            raise ValueError(
                f"unsupported full prefill arg count {arg_count}; expected "
                "3 prefix args + 9 args/layer + 2 suffix args, "
                "4 prefix args with attention_mask + 9 args/layer + 2 suffix args, or "
                "4 full-align prefix args + 11 args/layer + 2 suffix args"
            )
        attrs.append(("constant", "l__self___model_rotary_emb_inv_freq"))
        for layer_idx in range(layers_num):
            attrs.extend(_full_layer_param_attrs(layer_idx))
    elif mode == "decode":
        layers_num, rem = divmod(arg_count - 6, 13)
        if rem == 0 and layers_num > 0:
            attrs.extend(
                [
                    ("input", "args_1_cache_position"),
                    ("constant", "l__self___model_rotary_emb_inv_freq"),
                ]
            )
            for layer_idx in range(layers_num):
                prefix = f"l__self___model_layers__modules__{layer_idx}___"
                attrs.extend(
                    [
                        ("parameter", prefix + "input_layernorm_weight"),
                        ("parameter", prefix + "self_attn_q_proj_weight"),
                        ("parameter", prefix + "self_attn_k_proj_weight"),
                        ("parameter", prefix + "self_attn_v_proj_weight"),
                        ("input", f"args_key_cache_{layer_idx}"),
                        ("input", f"args_key_cache_position_{layer_idx}"),
                        ("input", f"args_value_cache_{layer_idx}"),
                        ("input", f"args_value_cache_position_{layer_idx}"),
                        ("parameter", prefix + "self_attn_o_proj_weight"),
                        (
                            "parameter",
                            prefix + "post_attention_layernorm_weight",
                        ),
                        ("parameter", prefix + "mlp_gate_proj_weight"),
                        ("parameter", prefix + "mlp_up_proj_weight"),
                        ("parameter", prefix + "mlp_down_proj_weight"),
                    ]
                )
            attrs.extend(
                [
                    ("parameter", "l__self___model_norm_weight"),
                    ("parameter", "l__self___lm_head_weight"),
                ]
            )
            if len(attrs) != arg_count:
                raise ValueError(
                    f"internal full-align decode arg annotation mismatch: "
                    f"expected {arg_count}, built {len(attrs)}"
                )
            return attrs

        layers_num, rem = divmod(arg_count - 6, 12)
        if rem == 0 and layers_num > 0:
            attrs.extend(
                [
                    ("input", "args_1_cache_position"),
                    ("constant", "l__self___model_rotary_emb_inv_freq"),
                ]
            )
            for layer_idx in range(layers_num):
                prefix = f"l__self___model_layers__modules__{layer_idx}___"
                attrs.extend(
                    [
                        ("parameter", prefix + "input_layernorm_weight"),
                        ("parameter", prefix + "self_attn_q_proj_weight"),
                        ("parameter", prefix + "self_attn_k_proj_weight"),
                        ("parameter", prefix + "self_attn_v_proj_weight"),
                        ("input", f"args_cache_position_{layer_idx}"),
                        ("input", f"args_key_cache_{layer_idx}"),
                        ("input", f"args_value_cache_{layer_idx}"),
                        ("parameter", prefix + "self_attn_o_proj_weight"),
                        (
                            "parameter",
                            prefix + "post_attention_layernorm_weight",
                        ),
                        ("parameter", prefix + "mlp_gate_proj_weight"),
                        ("parameter", prefix + "mlp_up_proj_weight"),
                        ("parameter", prefix + "mlp_down_proj_weight"),
                    ]
                )
            attrs.extend(
                [
                    ("parameter", "l__self___model_norm_weight"),
                    ("parameter", "l__self___lm_head_weight"),
                ]
            )
            if len(attrs) != arg_count:
                raise ValueError(
                    f"internal full-align decode arg annotation mismatch: "
                    f"expected {arg_count}, built {len(attrs)}"
                )
            return attrs

        layers_num, rem = divmod(arg_count - 8, 12)
        if rem == 0 and layers_num > 0:
            attrs.extend(
                [
                    ("input", "args_1_attention_mask"),
                    ("input", "args_2_cache_position"),
                    ("input", "args_3_cache_position"),
                    ("constant", "l__self___model_rotary_emb_inv_freq"),
                ]
            )
            for layer_idx in range(layers_num):
                prefix = f"l__self___model_layers__modules__{layer_idx}___"
                attrs.extend(
                    [
                        ("parameter", prefix + "input_layernorm_weight"),
                        ("parameter", prefix + "self_attn_q_proj_weight"),
                        ("parameter", prefix + "self_attn_k_proj_weight"),
                        ("parameter", prefix + "self_attn_v_proj_weight"),
                        ("input", f"args_cache_position_{layer_idx}"),
                        ("input", f"args_key_cache_{layer_idx}"),
                        ("input", f"args_value_cache_{layer_idx}"),
                        ("parameter", prefix + "self_attn_o_proj_weight"),
                        (
                            "parameter",
                            prefix + "post_attention_layernorm_weight",
                        ),
                        ("parameter", prefix + "mlp_gate_proj_weight"),
                        ("parameter", prefix + "mlp_up_proj_weight"),
                        ("parameter", prefix + "mlp_down_proj_weight"),
                    ]
                )
            attrs.extend(
                [
                    ("parameter", "l__self___model_norm_weight"),
                    ("parameter", "l__self___lm_head_weight"),
                ]
            )
            if len(attrs) != arg_count:
                raise ValueError(
                    f"internal attention-mask decode arg annotation mismatch: "
                    f"expected {arg_count}, built {len(attrs)}"
                )
            return attrs

        layers_num, rem = divmod(arg_count - 7, 12)
        if rem != 0 or layers_num <= 0:
            raise ValueError(
                f"unsupported full decode arg count {arg_count}; expected "
                "5 prefix args + 12 args/layer + 2 suffix args, "
                "6 prefix args with attention_mask + 12 args/layer + 2 suffix args, or "
                "4 full-align prefix args + 12/13 args/layer + 2 suffix args"
            )
        attrs.extend(
            [
                ("input", "args_1_cache_position"),
                ("input", "args_2_cache_position"),
                ("constant", "l__self___model_rotary_emb_inv_freq"),
            ]
        )
        for layer_idx in range(layers_num):
            prefix = f"l__self___model_layers__modules__{layer_idx}___"
            attrs.extend(
                [
                    ("parameter", prefix + "input_layernorm_weight"),
                    ("parameter", prefix + "self_attn_q_proj_weight"),
                    ("parameter", prefix + "self_attn_k_proj_weight"),
                    ("parameter", prefix + "self_attn_v_proj_weight"),
                    ("input", f"args_cache_position_{layer_idx}"),
                    ("input", f"args_key_cache_{layer_idx}"),
                    ("input", f"args_value_cache_{layer_idx}"),
                    ("parameter", prefix + "self_attn_o_proj_weight"),
                    ("parameter", prefix + "post_attention_layernorm_weight"),
                    ("parameter", prefix + "mlp_gate_proj_weight"),
                    ("parameter", prefix + "mlp_up_proj_weight"),
                    ("parameter", prefix + "mlp_down_proj_weight"),
                ]
            )
    else:
        raise ValueError(f"unsupported full arg annotation mode: {mode!r}")

    attrs.extend(
        [
            ("parameter", "l__self___model_norm_weight"),
            ("parameter", "l__self___lm_head_weight"),
        ]
    )
    if len(attrs) != arg_count:
        raise ValueError(
            f"internal full arg annotation mismatch: expected {arg_count}, "
            f"built {len(attrs)}"
        )
    return attrs


def _official_arg_attrs(
    scope: str, mode: str, arg_count: int | None = None
) -> list[tuple[str, str]]:
    """Official-style argument metadata for aligned scopes.

    The Buddy layer argument order is intentionally different from the official
    tt-mlir examples; this table follows the semantic map used by the golden
    harness. The attributes are consumed by the TTIR->TTNN lowering pipeline for
    parameter/constant hoisting and do not change tensor values.
    """

    if scope == "full":
        if arg_count is None:
            raise ValueError("full arg annotation requires arg_count")
        return _full_arg_attrs(mode, arg_count)

    if scope == "block" and mode == "decode":
        return [
            ("constant", "l__self___rotary_emb_inv_freq"),
            ("parameter", "l__self___block_input_layernorm_weight"),
            ("parameter", "l__self___block_self_attn_q_proj_weight"),
            ("parameter", "l__self___block_self_attn_k_proj_weight"),
            ("parameter", "l__self___block_self_attn_v_proj_weight"),
            ("parameter", "l__self___block_self_attn_o_proj_weight"),
            ("input", "args_0"),
            ("parameter", "l__self___block_post_attention_layernorm_weight"),
            ("parameter", "l__self___block_mlp_gate_proj_weight"),
            ("parameter", "l__self___block_mlp_up_proj_weight"),
            ("parameter", "l__self___block_mlp_down_proj_weight"),
        ]
    if scope == "block":
        raise ValueError(
            f"unsupported block arg annotation mode: {mode!r}; only decode has "
            "an official tt-mlir block fixture"
        )

    if scope != "layer":
        raise ValueError(f"unsupported arg annotation scope: {scope!r}")

    if mode == "decode":
        return [
            ("parameter", "l__self___model_embed_tokens_weight"),
            ("input", "args_0"),
            ("constant", "l__self___model_rotary_emb_inv_freq"),
            (
                "parameter",
                "l__self___model_layers__modules__0___input_layernorm_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_q_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_k_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_v_proj_weight",
            ),
            ("input", "args_2"),
            ("input", "args_3"),
            ("input", "args_1"),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_o_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___post_attention_layernorm_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_gate_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_up_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_down_proj_weight",
            ),
            ("parameter", "l__self___model_norm_weight"),
            ("parameter", "l__self___lm_head_weight"),
        ]
    if mode == "prefill":
        return [
            ("parameter", "l__self___model_embed_tokens_weight"),
            ("input", "args_0"),
            ("input", "args_1"),
            ("constant", "l__self___model_rotary_emb_inv_freq"),
            (
                "parameter",
                "l__self___model_layers__modules__0___input_layernorm_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_q_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_k_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_v_proj_weight",
            ),
            ("input", "args_2"),
            ("input", "args_3"),
            (
                "parameter",
                "l__self___model_layers__modules__0___self_attn_o_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___post_attention_layernorm_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_gate_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_up_proj_weight",
            ),
            (
                "parameter",
                "l__self___model_layers__modules__0___mlp_down_proj_weight",
            ),
            ("parameter", "l__self___model_norm_weight"),
            ("parameter", "l__self___lm_head_weight"),
        ]
    raise ValueError(f"unsupported layer arg annotation mode: {mode!r}")


def _annotate_result(result: str) -> str:
    rm = re.match(r"(tensor<[^>]+>)(?:\s+\{.*\})?$", result, re.S)
    if rm is None:
        raise ValueError(
            f"cannot parse function result for annotation: {result}"
        )
    return f"{rm.group(1)} {{ttcore.shard_status = #ttcore.shard_status<unsharded>}}"


def _annotate_official_arg_attrs(mlir_text: str, scope: str, mode: str) -> str:
    """Attach official-style argument/result attributes to a single TTIR func."""

    m = re.search(
        r"(func\.func @\w+\()(?P<args>.*?)(\) -> \()(?P<results>.*?)(\) \{)",
        mlir_text,
        re.S,
    )
    tuple_results = m is not None
    if m is None:
        m = re.search(
            r"(func\.func @\w+\()(?P<args>.*?)(\) -> )(?P<results>tensor<[^>]+>(?:\s+\{.*?\})?)( \{)",
            mlir_text,
            re.S,
        )
        if not m:
            return mlir_text

    sig_args = _split_mlir_args(m.group("args"))
    arg_attrs = _official_arg_attrs(scope, mode, len(sig_args))
    if len(sig_args) != len(arg_attrs):
        raise ValueError(
            f"expected {len(arg_attrs)} {scope} args for {mode}, got {len(sig_args)}"
        )

    annotated_args: list[str] = []
    for arg, (argument_type, ttir_name) in zip(sig_args, arg_attrs):
        am = re.match(r"(%arg\d+:\s*tensor<[^>]+>)(?:\s+\{.*\})?$", arg, re.S)
        if am is None:
            raise ValueError(
                f"cannot parse function argument for annotation: {arg}"
            )
        annotated_args.append(
            f"{am.group(1)} "
            "{"
            f"ttcore.argument_type = #ttcore.argument_type<{argument_type}>, "
            "ttcore.shard_status = #ttcore.shard_status<unsharded>, "
            f'ttir.name = "{ttir_name}"'
            "}"
        )

    results = (
        _split_mlir_args(m.group("results"))
        if tuple_results
        else [m.group("results")]
    )
    annotated_results = [_annotate_result(result) for result in results]

    result_text = ", ".join(annotated_results)
    if not tuple_results:
        result_text = f"({result_text})"

    return (
        mlir_text[: m.start("args")]
        + ", ".join(annotated_args)
        + mlir_text[m.end("args") : m.start("results")]
        + result_text
        + mlir_text[m.end("results") :]
    )


def _decode_forward_kwargs(
    model, past, prefill_len: int, decode_ids: torch.Tensor
) -> dict:
    """Keyword arguments for a single decode forward (align with golden)."""
    device = decode_ids.device
    kw: dict = {
        "past_key_values": past,
        "use_cache": True,
    }
    try:
        sig = inspect.signature(model.forward)
        if "cache_position" in sig.parameters:
            kw["cache_position"] = torch.tensor(
                [prefill_len], dtype=torch.long, device=device
            )
    except (TypeError, ValueError):
        pass
    return kw


def _fusion_list(skip_flash_attn: bool):
    # GQA must run before flash_attention_prefill so SDPA+KV-cache is still
    # ScaledDotProductFlashAttentionForCpuOp when gqa_attention_fusion runs.
    passes = [simply_fuse, gqa_attention_fusion]
    if not skip_flash_attn:
        passes.append(flash_attention_prefill)
    return passes


def main() -> int:
    args = _parse_args()
    probe_scopes = (
        "block",
        "embed",
        "embed_roundtrip",
        "embed_weightmul",
        "rmsstat",
        "rmsscale",
        "rmsunweighted",
        "rmsnorm",
        "kproj",
        "rmsnorm_kproj",
        "qkvproj_input",
        "rope_input",
        "mlp_input",
        "mlp_split",
        "mlp_down_input",
        "post_attn_norm_input",
        "sdpa_input",
        "sdpa_packed_input",
    )
    if args.scope in probe_scopes and args.static_cache:
        print(
            f"error: --scope {args.scope} does not use --static-cache.",
            file=sys.stderr,
        )
        return 1
    if (
        args.scope == "layer"
        and args.mode == "decode"
        and not args.static_cache
    ):
        print(
            "error: --scope layer --mode decode currently requires --static-cache.",
            file=sys.stderr,
        )
        return 1
    if args.device_argmax and args.scope != "full":
        print(
            "error: --device-argmax is only supported for --scope full.",
            file=sys.stderr,
        )
        return 1
    if args.full_align_wrapper and args.scope != "full":
        print(
            "error: --full-align-wrapper is only supported for --scope full.",
            file=sys.stderr,
        )
        return 1
    if args.use_proxy:
        u = "http://192.168.15.159:7890"
        os.environ.setdefault("http_proxy", u)
        os.environ.setdefault("https_proxy", u)
        os.environ.setdefault("HTTP_PROXY", u)
        os.environ.setdefault("HTTPS_PROXY", u)
    if args.matmul_as_dot_general:
        os.environ["BUDDY_TTIR_MATMUL_AS_DOT_GENERAL"] = "1"
    if args.rmsnorm_mean_as_sum:
        os.environ["BUDDY_TTIR_MEAN_AS_SUM"] = "1"
        os.environ["BUDDY_TTIR_SKIP_RMSNORM_BF16_SCALAR_CAST"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_F32_ADD"] = "1"
    if args.rsqrt_as_sqrt_reciprocal:
        os.environ["BUDDY_TTIR_RSQRT_AS_SQRT_RECIPROCAL"] = "1"
    if args.explicit_mul_broadcast:
        os.environ["BUDDY_TTIR_EXPLICIT_MUL_BROADCAST"] = "1"
    if args.mul_broadcast_as_repeat:
        os.environ["BUDDY_TTIR_MUL_BROADCAST_AS_REPEAT"] = "1"
    if args.mul_broadcast_as_repeat_op:
        os.environ["BUDDY_TTIR_MUL_BROADCAST_AS_REPEAT_OP"] = "1"
    if args.softmax_as_explicit:
        os.environ["BUDDY_TTIR_SOFTMAX_AS_EXPLICIT"] = "1"
    if args.safe_softmax_mask:
        os.environ["BUDDY_TTIR_SAFE_SOFTMAX_MASK"] = "1"
    if args.mask_as_where:
        os.environ["BUDDY_TTIR_MASK_AS_WHERE"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_COMPARE_TYPES"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_BOOL_TENSORS"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_SHAPE_TYPES"] = "1"
    if args.silu_as_sigmoid_mul:
        os.environ["BUDDY_TTIR_SILU_AS_SIGMOID_MUL"] = "1"
    if args.embedding_as_gather:
        os.environ["BUDDY_TTIR_EMBEDDING_AS_GATHER"] = "1"
    if args.scope == "block" and args.mode == "decode":
        os.environ["BUDDY_TTIR_PRESERVE_FULL_TYPES"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_SHAPE_TYPES"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.synthetic_weights:
            if args.scope == "attn_split":
                synthetic_layers = int(args.layer_index) + 1
            else:
                synthetic_layers = (
                    1 if args.scope in ("layer", *probe_scopes) else None
                )
            model, tokenizer = _load_synthetic_model(
                args.model, synthetic_layers
            )
        else:
            model, tokenizer = _load_model_and_tokenizer(args.model)
    except Exception as e:
        print(
            f"error: could not load model {args.model!r} ({e}). "
            "Install transformers and set LLAMA31_MODEL_PATH or --model.",
            file=sys.stderr,
        )
        return 1

    cfg = getattr(model, "config", None)
    if args.attn_implementation != "default" and cfg is not None:
        cfg._attn_implementation = args.attn_implementation
        cfg.attn_implementation = args.attn_implementation
    if args.official_attention_f32:
        _patch_llama_official_attention_f32()
        if cfg is not None:
            cfg._attn_implementation = "eager"
            cfg.attn_implementation = "eager"
    if cfg is not None:
        print(
            f"loaded model: {args.model} "
            f"(hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
            f"heads={cfg.num_attention_heads}, kv_heads="
            f"{getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}, "
            f"vocab={cfg.vocab_size}, "
            f"rope_scaling={getattr(cfg, 'rope_scaling', None)})"
        )

    device = next(model.parameters()).device
    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)

    if args.scope == "layer":
        model = _OneLayerLMWrapper(model)
    elif args.scope == "layer_stages":
        model = _Layer0StageProbeWrapper(model, layer_index=args.layer_index)
    elif args.scope == "attn_split":
        model = _AttentionSplitProbeWrapper(model, layer_index=args.layer_index)
    elif args.scope == "block":
        model = _OneBlockWrapper(model, zero_position_ids=args.mode == "decode")
    elif args.scope == "embed":
        model = _EmbedProbeWrapper(model)
    elif args.scope == "embed_roundtrip":
        model = _EmbedRoundTripProbeWrapper(model)
    elif args.scope == "embed_weightmul":
        model = _EmbedWeightMulProbeWrapper(model)
    elif args.scope == "rmsstat":
        model = _RMSStatProbeWrapper(model)
    elif args.scope == "rmsscale":
        model = _RMSScaleProbeWrapper(model)
    elif args.scope == "rmsunweighted":
        model = _RMSUnweightedProbeWrapper(model)
    elif args.scope == "rmsnorm":
        model = _RMSNormProbeWrapper(model)
    elif args.scope == "kproj":
        model = _KProjProbeWrapper(model)
    elif args.scope == "rmsnorm_kproj":
        model = _RMSNormKProjProbeWrapper(model)
    elif args.scope == "qkvproj_input":
        model = _QKVProjInputProbeWrapper(model)
    elif args.scope == "rope_input":
        model = _RoPEInputProbeWrapper(model)
    elif args.scope == "mlp_input":
        model = _MLPInputProbeWrapper(model, layer_index=args.layer_index)
    elif args.scope == "mlp_split":
        model = _MLPSplitProbeWrapper(model, layer_index=args.layer_index)
    elif args.scope == "mlp_down_input":
        model = _MLPDownInputProbeWrapper(model, layer_index=args.layer_index)
    elif args.scope == "post_attn_norm_input":
        model = _PostAttnNormInputProbeWrapper(model)
    elif args.scope == "sdpa_input":
        model = _SDPAInputProbeWrapper(model)
    elif args.scope == "sdpa_packed_input":
        model = _SDPAPackedInputProbeWrapper(model)
    elif args.full_align_wrapper:
        model = _FullLMAlignedWrapper(model)

    if args.device_argmax:
        model = _ArgmaxHeadWrapper(model)

    if args.static_cache:
        _patch_static_cache_for_buddy()
        from transformers import StaticCache

        L = int(args.max_cache_len)
        if args.mode == "prefill":
            trace_seq = L if args.scope == "full" else int(args.seq)
            input_ids = torch.zeros(
                args.batch, trace_seq, dtype=torch.long, device=device
            )
            attention_kwargs = {}
            if args.runtime_attention_mask:
                attention_kwargs["attention_mask"] = torch.ones(
                    args.batch, L, dtype=torch.long, device=device
                )
            with torch.no_grad():
                if (
                    args.scope in ("layer", "layer_stages", "attn_split")
                    or args.full_align_wrapper
                ):
                    past_kv = _make_static_cache(
                        StaticCache, model.config, L, args.batch, device
                    )
                    if args.full_align_wrapper:
                        _early_initialize_static_cache(
                            past_kv, model.config, args.batch, device
                        )
                    cache_position = torch.arange(
                        trace_seq, dtype=torch.long, device=device
                    )
                    graphs = dynamo_compiler.importer(
                        model,
                        input_ids=input_ids,
                        **attention_kwargs,
                        use_cache=True,
                        cache_position=cache_position,
                        past_key_values=past_kv,
                        cache_implementation="static",
                    )
                else:
                    graphs = dynamo_compiler.importer(
                        model,
                        input_ids=input_ids,
                        **attention_kwargs,
                        use_cache=True,
                        cache_implementation="static",
                    )
        else:
            past_kv = _make_static_cache(
                StaticCache, model.config, L, args.batch, device
            )
            decode_ids = torch.zeros(
                args.batch, 1, dtype=torch.long, device=device
            )
            pos_value = args.cache_position
            if pos_value is None:
                pos_value = min(200, max(0, L - 1))
            cache_position = torch.tensor(
                [pos_value], dtype=torch.long, device=device
            )
            attention_kwargs = {}
            if args.runtime_attention_mask:
                attention_kwargs["attention_mask"] = torch.ones(
                    args.batch, L, dtype=torch.long, device=device
                )
            with torch.no_grad():
                if args.full_align_wrapper:
                    _early_initialize_static_cache(
                        past_kv, model.config, args.batch, device
                    )
                else:
                    model(
                        input_ids=decode_ids,
                        **attention_kwargs,
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_implementation="static",
                    )
                if args.scope != "layer" and not args.full_align_wrapper:
                    # Transformers 5.5: model.forward dropped the
                    # `cache_position` arg; advance per-layer
                    # `cumulative_length` so the patched `update` sees a
                    # non-zero base, which Dynamo traces as a graph-level
                    # placeholder.
                    for layer in getattr(past_kv, "layers", []):
                        base = getattr(layer, "cumulative_length", None)
                        if base is not None:
                            layer.cumulative_length = torch.tensor(
                                [pos_value],
                                dtype=base.dtype,
                                device=base.device,
                            )
                graphs = dynamo_compiler.importer(
                    model,
                    input_ids=decode_ids,
                    **attention_kwargs,
                    use_cache=True,
                    cache_position=cache_position,
                    past_key_values=past_kv,
                    cache_implementation="static",
                )
    elif args.mode == "prefill":
        if args.scope == "rope_input":
            head_dim = (
                model.config.hidden_size // model.config.num_attention_heads
            )
            query_packed = torch.ones(
                args.batch,
                args.seq,
                model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            key_packed = torch.ones(
                args.batch,
                args.seq,
                model.config.num_key_value_heads * head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            cos = torch.ones(
                args.batch,
                args.seq,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            sin = torch.ones_like(cos)
            with torch.no_grad():
                graphs = dynamo_compiler.importer(
                    model, query_packed, key_packed, cos, sin
                )
        elif args.scope == "sdpa_input":
            head_dim = (
                model.config.hidden_size // model.config.num_attention_heads
            )
            query_states = torch.ones(
                args.batch,
                model.config.num_attention_heads,
                args.seq,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            key_states = torch.ones_like(query_states)
            value_states = torch.ones_like(query_states)
            with torch.no_grad():
                graphs = dynamo_compiler.importer(
                    model, query_states, key_states, value_states
                )
        elif args.scope == "sdpa_packed_input":
            query_packed = torch.ones(
                args.batch,
                args.seq,
                model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            key_packed = torch.ones_like(query_packed)
            value_packed = torch.ones_like(query_packed)
            with torch.no_grad():
                graphs = dynamo_compiler.importer(
                    model, query_packed, key_packed, value_packed
                )
        elif args.scope == "mlp_down_input":
            product_states = torch.ones(
                args.batch,
                args.seq,
                model.config.intermediate_size,
                dtype=torch.bfloat16,
                device=device,
            )
            with torch.no_grad():
                graphs = dynamo_compiler.importer(model, product_states)
        elif args.scope in (
            "block",
            "qkvproj_input",
            "mlp_input",
            "mlp_split",
            "post_attn_norm_input",
        ):
            hidden_states = torch.ones(
                args.batch,
                args.seq,
                model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            with torch.no_grad():
                graphs = dynamo_compiler.importer(model, hidden_states)
        else:
            input_ids = torch.ones(
                args.batch, args.seq, dtype=torch.long, device=device
            )
            with torch.no_grad():
                if args.prefill_use_cache:
                    graphs = dynamo_compiler.importer(
                        model, input_ids, use_cache=True
                    )
                else:
                    graphs = dynamo_compiler.importer(model, input_ids)
    elif args.scope == "block":
        hidden_states = torch.ones(
            args.batch,
            1,
            model.config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        with torch.no_grad():
            graphs = dynamo_compiler.importer(model, hidden_states)
    else:
        prefill_ids = torch.ones(
            args.batch, args.seq, dtype=torch.long, device=device
        )
        decode_ids = torch.ones(args.batch, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out0 = model(input_ids=prefill_ids, use_cache=True)
            past = out0.past_key_values
        dec_kw = _decode_forward_kwargs(model, past, args.seq, decode_ids)
        with torch.no_grad():
            graphs = dynamo_compiler.importer(model, decode_ids, **dec_kw)

    if len(graphs) != 1:
        print(f"error: expected one graph, got {len(graphs)}.", file=sys.stderr)
        return 1

    g = graphs[0]
    g.fuse_ops(_fusion_list(args.skip_flash_attn))

    if args.mode == "decode":
        ph_names = [
            str(op.name) for op in g.body if isinstance(op, PlaceholderOp)
        ]
        cur = args.cur_pos_placeholder
        if cur is None:
            for op in g.body:
                if not isinstance(op, PlaceholderOp):
                    continue
                meta = op.tensor_meta or {}
                shape = meta.get("shape")
                dtype = meta.get("dtype")
                dt_str = str(dtype).lower()
                is_int = "int" in dt_str and "64" in dt_str
                if is_int and shape is not None and list(shape) == [1]:
                    cur = str(op.name)
                    break
            if cur is None:
                for n in ph_names:
                    nl = n.lower()
                    if (
                        "pos" in nl
                        or "cache" in nl
                        or "cache_position" in nl
                        or nl.endswith("_pos")
                    ):
                        cur = n
                        break
        if cur:
            _patch_gqa_cur_pos(g, cur)
        else:
            print(
                "Placeholders:",
                ", ".join(ph_names),
                file=sys.stderr,
            )
            print(
                "warning: decode mode without a matching "
                "--cur-pos-placeholder; GQAAttentionFusedOp lowering may "
                "fail until kwargs['cur_pos_tensor'] is set.",
                file=sys.stderr,
            )

    driver = GraphDriver(g)
    sg = driver.subgraphs[0]
    try:
        sg.lower_to_ttir(element_dtype=args.element_dtype)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(
            f"error: missing TTIR lowering for an op: {e}. "
            "Extend buddy.compiler.ops.ttir_llm.",
            file=sys.stderr,
        )
        return 1

    mod = sg.ttir_module
    if args.packed_forward:
        try:
            append_ttir_forward_bf16_f32_packed_i64_runtime(
                mod,
                subgraph_func_name="subgraph0",
                forward_func_name="forward",
            )
        except Exception as e:
            print(
                f"error: --packed-forward failed ({e}). "
                "Decode graphs with multiple integer inputs may need a "
                "different entry.",
                file=sys.stderr,
            )
            return 1

    out_scope = "" if args.scope == "full" else f"_{args.scope}"
    out_stem = f"{args.output_stem_prefix}_{args.mode}{out_scope}"
    out = args.output_dir / (
        f"{out_stem}_module.mlir" if args.packed_forward else f"{out_stem}.mlir"
    )
    mlir_text = str(mod).strip()
    if args.prune_unused_ttir_args:
        mlir_text = _prune_unused_ttir_args(mlir_text).strip()
    if args.annotate_official_arg_attrs:
        if args.scope not in ("full", "layer", "block"):
            print(
                "error: --annotate-official-arg-attrs is only defined for "
                "--scope full, --scope layer, or official-aligned --scope block.",
                file=sys.stderr,
            )
            return 1
        try:
            mlir_text = _annotate_official_arg_attrs(
                mlir_text, args.scope, args.mode
            ).strip()
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    mlir_text, rank_broadcast_count = _legalize_rank_expanding_broadcasts(
        mlir_text
    )
    if rank_broadcast_count:
        print(
            "Patched "
            f"{rank_broadcast_count} rank-expanding broadcast ops with reshapes"
        )
    if args.mode == "decode":
        mlir_text, mask_count = _keep_decode_cache_mask_compares_f32(
            mlir_text, args.max_cache_len
        )
        if mask_count:
            print(
                "Patched "
                f"{mask_count} decode cache-position masks to compare in fp32"
            )
        mlir_text, index_count = _keep_position_index_math_f32(mlir_text)
        if index_count:
            print(
                "Patched "
                f"{index_count} decode position-index ops to stay in fp32"
            )
        mlir_text, update_index_count = _keep_update_cache_indices_i32(
            mlir_text
        )
        if update_index_count:
            print(
                "Patched "
                f"{update_index_count} decode update-cache index ops to stay in i32"
            )
        mlir_text, batched_update_count = _legalize_batched_update_cache_inputs(
            mlir_text
        )
        if batched_update_count:
            print(f"Patched {batched_update_count} batched update-cache inputs")
        mlir_text, batched_sdpa_count = _legalize_batched_sdpa_decode_inputs(
            mlir_text
        )
        if batched_sdpa_count:
            print(f"Patched {batched_sdpa_count} batched decode SDPA inputs")
    out.write_text(mlir_text + "\n", encoding="utf-8")
    print(f"Wrote {out.resolve()}")

    opt = args.ttmlir_opt or shutil.which("ttmlir-opt")
    if opt:
        cmd = [opt, str(out), "-o", os.devnull]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"warning: ttmlir-opt failed: {e}", file=sys.stderr)
            return 1
        print("ttmlir-opt: OK")
    else:
        print("Skip ttmlir-opt (not on PATH).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
