# ===- llama31_chat_prepare.py ------------------------------------------------
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
# Prepare the host-side artifacts required by ``llama31_chat_run.py``.
#
# We re-run the Buddy ``DynamoCompiler`` for prefill and decode under the
# same configuration as ``buddy-llama31-lower-ttir.py --static-cache
# --max-cache-len 1024``, then snapshot for each phase:
#
#   - ``slot_roles.json``: per-``@subgraph0``-input role tag.  Roles include
#     ``"weight"`` (host-side tensor baked into the model), ``"input_ids"``,
#     ``"cache_position"``, ``"past_K"``/``"past_V"``, ``"inv_freq"``.
#   - ``weights.npz``: one array per weight slot, keyed ``"w_%04d" % slot``.
#   - ``inv_freq.npy``: F32 rotary base (if present).
#   - ``shapes.json`` / ``dtypes.json``: per-arg metadata for sanity-check.
#
# The chat runner only needs to load these ``.npz`` / ``.json`` artifacts and
# swap in the runtime tensors (tokens, cache position, past KV) per step.
#
# Usage::
#
#   python llama31_chat_prepare.py -o chat_artifacts --max-cache-len 1024
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from buddy.compiler.ops import tosa

_TTIR_CACHE_OPS_LIB = None
_TTIR_CACHE_OPS_REGISTERED = False


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Prepare prefill+decode artifacts for the interactive Llama 3.1 TT runner."
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "LLAMA31_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"
        ),
    )
    p.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Static cache length (must match lowering).",
    )
    p.add_argument(
        "-o",
        "--artifacts",
        type=Path,
        default=here / "chat_artifacts",
    )
    p.add_argument("--use-proxy", action="store_true")
    p.add_argument(
        "--phases",
        default="prefill,decode",
        help="Comma-separated list of phases to prepare (default prefill,decode).",
    )
    p.add_argument(
        "--device-argmax",
        action="store_true",
        help=(
            "Wrap the model with the same _ArgmaxHeadWrapper used during "
            "lowering, so the dynamo graph dumped here matches the flatbuffer "
            "in input slot order. Required when the chat flatbuffer was "
            "produced with --device-argmax."
        ),
    )
    p.add_argument(
        "--full-align-wrapper",
        action="store_true",
        help=(
            "Prepare artifacts for buddy-llama31-lower-ttir.py "
            "--full-align-wrapper. This uses the explicit StaticCache wrapper "
            "and official-alignment graph flags."
        ),
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write slot roles/shapes/dtypes/summary but skip the large weights.npz.",
    )
    return p.parse_args()


def _proxy_env():
    u = "http://192.168.15.159:7890"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, u)


def _register_ttir_cache_ops() -> None:
    """Register the same cache marker ops used by the lowering script."""
    global _TTIR_CACHE_OPS_LIB, _TTIR_CACHE_OPS_REGISTERED
    if _TTIR_CACHE_OPS_REGISTERED:
        return
    if hasattr(torch.ops, "buddy_ttir") and hasattr(
        torch.ops.buddy_ttir, "fill_cache"
    ):
        _TTIR_CACHE_OPS_REGISTERED = True
        return

    lib = torch.library.Library("buddy_ttir", "DEF")
    lib.define(
        "fill_cache(Tensor cache, Tensor input, int batch_offset) -> Tensor"
    )
    lib.define(
        "update_cache(Tensor cache, Tensor input, Tensor update_index, int batch_offset) -> Tensor"
    )

    @torch.library.impl(lib, "fill_cache", "CompositeExplicitAutograd")
    def _fill_cache(cache, input, batch_offset: int):
        out = cache.clone()
        end = batch_offset + input.shape[0]
        out[batch_offset:end, :, : input.shape[2], :] = input
        return out

    @torch.library.impl(lib, "fill_cache", "Meta")
    def _fill_cache_meta(cache, input, batch_offset: int):
        return torch.empty_strided(
            tuple(cache.shape),
            tuple(cache.stride()),
            dtype=cache.dtype,
            device="meta",
        )

    @torch.library.impl(lib, "update_cache", "CompositeExplicitAutograd")
    def _update_cache(cache, input, update_index, batch_offset: int):
        out = cache.clone()
        try:
            pos = int(update_index.reshape(-1)[0].item())
            payload = input.permute(2, 1, 0, 3)
            end = batch_offset + payload.shape[0]
            out[batch_offset:end, :, pos : pos + payload.shape[2], :] = payload
        except Exception:
            pass
        return out

    @torch.library.impl(lib, "update_cache", "Meta")
    def _update_cache_meta(cache, input, update_index, batch_offset: int):
        return torch.empty_strided(
            tuple(cache.shape),
            tuple(cache.stride()),
            dtype=cache.dtype,
            device="meta",
        )

    _TTIR_CACHE_OPS_LIB = lib
    _TTIR_CACHE_OPS_REGISTERED = True


def _patch_static_cache_for_buddy(exact_cache_ops: bool = False) -> None:
    """Same monkey-patch as the lowering script.

    Replaces ``StaticLayer.update`` with a ``where``-based scatter so we do not
    hit ``aten.index_copy_`` (unregistered in the Buddy TOSA op map).
    Compatible with both Transformers 4.x and 5.5+.
    """
    import transformers.cache_utils as cu

    if exact_cache_ops:
        _register_ttir_cache_ops()

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
                cache_position = torch.arange(S_, device=self.device) + base
            else:
                cache_position = torch.arange(S_, device=self.device)

        L = self.keys.shape[-2]
        S = key_states.shape[-2]

        if exact_cache_ops:
            if S == 1:
                idx = cache_position
                if len(idx.shape) == 0:
                    idx = idx.view(1)
                k_update = key_states.permute(2, 1, 0, 3)
                v_update = value_states.permute(2, 1, 0, 3)
                self.keys = torch.ops.buddy_ttir.update_cache(
                    self.keys, k_update, idx, 0
                )
                self.values = torch.ops.buddy_ttir.update_cache(
                    self.values, v_update, idx, 0
                )
                return self.keys, self.values

            keys = self.keys
            values = self.values
            B = key_states.shape[0]
            for b in range(B):
                keys = torch.ops.buddy_ttir.fill_cache(
                    keys, key_states[b : b + 1, :, :, :], int(b)
                )
                values = torch.ops.buddy_ttir.fill_cache(
                    values, value_states[b : b + 1, :, :, :], int(b)
                )
            self.keys = keys
            self.values = values
            return self.keys, self.values

        if S == L:
            self.keys = key_states
            self.values = value_states
            return self.keys, self.values

        idx = torch.arange(L, device=self.keys.device).view(1, L)
        pos2d = cache_position.view(-1, 1)
        mask = idx == pos2d
        mask = mask.view(1, 1, L, 1)
        B, H, _, D = self.keys.shape
        ks_exp = key_states.expand(B, H, L, D)
        vs_exp = value_states.expand(B, H, L, D)
        self.keys = torch.where(mask, ks_exp, self.keys)
        self.values = torch.where(mask, vs_exp, self.values)
        return self.keys, self.values

    cu.StaticLayer.update = update


def _load_model(model_id: str):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    m.eval()
    return m


def _torch_dtype_str(t: torch.Tensor) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }.get(t.dtype, str(t.dtype))


def _tensor_to_numpy_and_dtype(t: torch.Tensor) -> tuple[np.ndarray, str]:
    """Serialize tensor to numpy in a reversible way.  bf16 is stored as uint16
    bits so numpy can own the buffer; the runner view-casts back."""
    tcpu = t.detach().cpu().contiguous()
    if tcpu.dtype == torch.bfloat16:
        return tcpu.view(torch.uint16).numpy().astype(np.uint16), "bfloat16"
    if tcpu.dtype == torch.float32:
        return tcpu.numpy().astype(np.float32), "float32"
    if tcpu.dtype == torch.float16:
        return tcpu.view(torch.uint16).numpy().astype(np.uint16), "float16"
    if tcpu.dtype == torch.int64:
        return tcpu.numpy().astype(np.int64), "int64"
    if tcpu.dtype == torch.int32:
        return tcpu.numpy().astype(np.int32), "int32"
    if tcpu.dtype == torch.bool:
        return tcpu.numpy().astype(np.uint8), "bool"
    raise ValueError(f"unsupported dtype {tcpu.dtype}")


def _fuse_list():
    return [simply_fuse, gqa_attention_fusion, flash_attention_prefill]


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
) -> torch.Tensor | None:
    if past_key_values is None:
        return None
    bsz, seq_len = inputs_embeds.shape[:2]
    kv_len = int(past_key_values.get_max_cache_shape())
    key_pos = torch.arange(
        kv_len, dtype=torch.long, device=inputs_embeds.device
    ).view(1, 1, kv_len)
    query_pos = position_ids.to(torch.long).view(
        position_ids.shape[0], seq_len, 1
    )
    allowed = (key_pos <= query_pos).unsqueeze(1)
    if allowed.shape[0] != bsz:
        allowed = allowed.expand(bsz, 1, seq_len, kv_len)
    zero = torch.full(
        (bsz, 1, seq_len, kv_len),
        0.0,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    neg_inf = torch.full(
        (bsz, 1, seq_len, kv_len),
        float("-inf"),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    return torch.where(allowed, zero, neg_inf)


def _patch_llama_official_attention_f32() -> None:
    """Patch eager Llama attention to mirror the lowering script."""
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


def _early_initialize_static_cache(past_kv, config, batch: int, device) -> None:
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


class _FullLMAlignedWrapper(torch.nn.Module):
    """Mirror of buddy-llama31-lower-ttir.py::_FullLMAlignedWrapper."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = inner.config

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
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
            hidden_states, past_key_values, position_ids
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
    """Mirror of buddy-llama31-lower-ttir.py::_ArgmaxHeadWrapper.

    Keeps placeholder ordering identical between the lowering script (which
    produced the flatbuffer) and this prepare step (which dumps weights and
    slot roles), so chat_run can keep using its existing slot map.
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


def _kv_shape_from_config(model, max_cache_len: int) -> list[int]:
    """Static past_K / past_V placeholder shape = [1, n_kv, L, head_dim].

    DeepSeek-R1-Distill-Qwen-1.5B has n_kv=2 / head_dim=128; Llama-3.1-8B has
    n_kv=8 / head_dim=128. Reading them off the HF config keeps this script
    model-agnostic.
    """
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    return [1, int(n_kv), int(max_cache_len), int(head_dim)]


def _prepare_phase(
    phase: str,
    model,
    out_dir: Path,
    max_cache_len: int,
    full_align_wrapper: bool = False,
    metadata_only: bool = False,
) -> None:
    from transformers import StaticCache

    L = int(max_cache_len)
    device = next(model.parameters()).device
    dynamo_compiler = DynamoCompiler(primary_registry=tosa.ops_registry)
    kv_shape = _kv_shape_from_config(model, L)

    if phase == "prefill":
        input_ids = torch.zeros(1, L, dtype=torch.long, device=device)
        with torch.no_grad():
            if full_align_wrapper:
                past_kv = StaticCache(config=model.config, max_cache_len=L)
                _early_initialize_static_cache(past_kv, model.config, 1, device)
                cache_position = torch.arange(
                    L, dtype=torch.long, device=device
                )
                graphs = dynamo_compiler.importer(
                    model,
                    input_ids=input_ids,
                    use_cache=True,
                    cache_position=cache_position,
                    past_key_values=past_kv,
                    cache_implementation="static",
                )
            else:
                graphs = dynamo_compiler.importer(
                    model,
                    input_ids=input_ids,
                    use_cache=True,
                    cache_implementation="static",
                )
    elif phase == "decode":
        past_kv = StaticCache(config=model.config, max_cache_len=L)
        decode_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        cache_position = torch.tensor([200], dtype=torch.long, device=device)
        with torch.no_grad():
            if full_align_wrapper:
                _early_initialize_static_cache(past_kv, model.config, 1, device)
            else:
                model(
                    input_ids=decode_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_implementation="static",
                )
                for layer in getattr(past_kv, "layers", []):
                    base = getattr(layer, "cumulative_length", None)
                    if base is not None:
                        layer.cumulative_length = torch.tensor(
                            [200], dtype=base.dtype, device=base.device
                        )
            graphs = dynamo_compiler.importer(
                model,
                input_ids=decode_ids,
                use_cache=True,
                cache_position=cache_position,
                past_key_values=past_kv,
                cache_implementation="static",
            )
    else:
        raise ValueError(f"unknown phase {phase!r}")

    if len(graphs) != 1:
        raise RuntimeError(f"expected 1 graph, got {len(graphs)} for {phase}")
    g = graphs[0]

    params_ref = getattr(g, "_params_ref", None)
    runtime_ref = getattr(g, "_runtime_inputs_ref", None)
    if params_ref is None:
        raise RuntimeError("graph missing _params_ref (Buddy import)")
    if runtime_ref is None:
        raise RuntimeError(
            "graph missing _runtime_inputs_ref (update Buddy frontend)"
        )

    g.fuse_ops(_fuse_list())
    driver = GraphDriver(g)
    if len(driver.subgraphs) != 1:
        raise RuntimeError(
            f"expected 1 subgraph for {phase}, got {len(driver.subgraphs)}"
        )
    sg_name = list(driver._subgraphs.keys())[0]
    sg_input_names = driver._subgraphs_inputs[sg_name]

    fake_ix_list = list(g._fake_params)
    inp_ix_list = list(g._inputs)
    fake_ix_set = {int(x) for x in fake_ix_list}
    inp_ix_set = {int(x) for x in inp_ix_list}

    name_to_tensor: dict[str, torch.Tensor] = {}
    name_to_is_input: dict[str, bool] = {}
    for fi, pidx in enumerate(fake_ix_list):
        ph_op = g.body[pidx]
        if fi < len(params_ref):
            name_to_tensor[ph_op.name] = params_ref[fi]
            name_to_is_input[ph_op.name] = False
    for ii, iidx in enumerate(inp_ix_list):
        ph_op = g.body[iidx]
        if ii < len(runtime_ref):
            name_to_tensor[ph_op.name] = runtime_ref[ii]
            name_to_is_input[ph_op.name] = True

    slot_roles: list[dict] = []
    weights: dict[str, np.ndarray] = {}
    shapes: list[list[int]] = []
    dtypes: list[str] = []

    kv_seen = 0
    inv_freq_arr: np.ndarray | None = None

    for slot, ph_name in enumerate(sg_input_names):
        if ph_name not in name_to_tensor:
            raise RuntimeError(
                f"{phase}: subgraph input slot {slot} ({ph_name}) has no tensor"
            )
        t = name_to_tensor[ph_name]
        is_input = name_to_is_input[ph_name]
        shp = list(t.shape)
        dt = _torch_dtype_str(t)
        shapes.append(shp)
        dtypes.append(dt)

        role: str = "unknown"
        if not is_input:
            role = "weight"
            if not metadata_only:
                arr, _ = _tensor_to_numpy_and_dtype(t)
                weights[f"w_{slot:04d}"] = arr
        else:
            if (
                dt == "int64"
                and shp == [1, L]
                or dt == "int64"
                and shp == [1, 1]
            ):
                role = "input_ids"
            elif dt == "int64" and shp == [L] or dt == "int64" and shp == [1]:
                role = "cache_position"
            elif dt == "bfloat16" and shp == kv_shape:
                if kv_seen % 2 == 0:
                    role = f"past_K_{kv_seen // 2:02d}"
                else:
                    role = f"past_V_{kv_seen // 2:02d}"
                kv_seen += 1
            elif dt == "float32" and shp == [64]:
                role = "inv_freq"
                arr, _ = _tensor_to_numpy_and_dtype(t)
                inv_freq_arr = arr.astype(np.float32)
            else:
                role = f"runtime:{dt}:{shp}"

        slot_roles.append(
            {
                "slot": slot,
                "placeholder": ph_name,
                "role": role,
                "shape": shp,
                "dtype": dt,
            }
        )

    if full_align_wrapper and phase == "decode":
        cfg = model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 32))
        expected = 4 + num_layers * 12 + 2
        if len(slot_roles) != expected:
            raise RuntimeError(
                f"full-align decode prepare expected {expected} GraphDriver "
                f"slots before TTIR-order expansion, got {len(slot_roles)}"
            )

        old_roles = slot_roles
        old_shapes = shapes
        old_dtypes = dtypes
        old_weights = weights
        slot_roles = []
        shapes = []
        dtypes = []
        weights = {}

        def append_old(old_slot: int) -> None:
            new_slot = len(slot_roles)
            entry = dict(old_roles[old_slot])
            entry["slot"] = new_slot
            slot_roles.append(entry)
            shapes.append(old_shapes[old_slot])
            dtypes.append(old_dtypes[old_slot])
            if entry["role"] == "weight" and not metadata_only:
                weights[f"w_{new_slot:04d}"] = old_weights[f"w_{old_slot:04d}"]

        for old_slot in range(4):
            append_old(old_slot)

        old = 4
        for _layer_idx in range(num_layers):
            # GraphDriver has one cache_position before K/V. TTIR update_cache
            # consumes the index once for K and once for V, so the flatbuffer
            # signature needs K, index, V, index in that order.
            order = [
                old + 0,
                old + 1,
                old + 2,
                old + 3,
                old + 5,
                old + 4,
                old + 6,
                old + 4,
                old + 7,
                old + 8,
                old + 9,
                old + 10,
                old + 11,
            ]
            for old_slot in order:
                append_old(old_slot)
            old += 12

        append_old(old)
        append_old(old + 1)

    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "slot_roles.json").write_text(
        json.dumps(slot_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (phase_dir / "shapes.json").write_text(json.dumps(shapes), encoding="utf-8")
    (phase_dir / "dtypes.json").write_text(json.dumps(dtypes), encoding="utf-8")
    if weights:
        np.savez(phase_dir / "weights.npz", **weights)
    if inv_freq_arr is not None:
        np.save(phase_dir / "inv_freq.npy", inv_freq_arr)
    summary = {
        "phase": phase,
        "max_cache_len": L,
        "num_slots": len(slot_roles),
        "num_weight_slots": sum(1 for r in slot_roles if r["role"] == "weight"),
        "num_runtime_slots": sum(
            1 for r in slot_roles if r["role"] != "weight"
        ),
        "kv_seen": kv_seen,
        "has_inv_freq": inv_freq_arr is not None,
    }
    (phase_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[OK] {phase}: {summary}")


def main() -> int:
    args = _parse_args()
    if args.full_align_wrapper and args.device_argmax:
        print(
            "error: --full-align-wrapper is not compatible with --device-argmax yet",
            file=sys.stderr,
        )
        return 1
    if args.use_proxy:
        _proxy_env()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    if args.full_align_wrapper:
        os.environ["BUDDY_TTIR_MATMUL_AS_DOT_GENERAL"] = "1"
        os.environ["BUDDY_TTIR_MEAN_AS_SUM"] = "1"
        os.environ["BUDDY_TTIR_SKIP_RMSNORM_BF16_SCALAR_CAST"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_F32_ADD"] = "1"
        os.environ["BUDDY_TTIR_SOFTMAX_AS_EXPLICIT"] = "1"
        os.environ["BUDDY_TTIR_SAFE_SOFTMAX_MASK"] = "1"
        os.environ["BUDDY_TTIR_MASK_AS_WHERE"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_COMPARE_TYPES"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_BOOL_TENSORS"] = "1"
        os.environ["BUDDY_TTIR_PRESERVE_SHAPE_TYPES"] = "1"
        os.environ["BUDDY_TTIR_SILU_AS_SIGMOID_MUL"] = "1"
        os.environ["BUDDY_TTIR_EMBEDDING_AS_GATHER"] = "1"

    _patch_static_cache_for_buddy(exact_cache_ops=args.full_align_wrapper)
    try:
        model = _load_model(args.model)
    except Exception as e:
        print(f"error: load model: {e}", file=sys.stderr)
        return 1

    if args.full_align_wrapper:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg._attn_implementation = "eager"
            cfg.attn_implementation = "eager"
        _patch_llama_official_attention_f32()
        model = _FullLMAlignedWrapper(model)

    if args.device_argmax:
        model = _ArgmaxHeadWrapper(model)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in phases:
        _prepare_phase(
            phase,
            model,
            args.artifacts,
            args.max_cache_len,
            full_align_wrapper=args.full_align_wrapper,
            metadata_only=args.metadata_only,
        )

    print(f"[OK] chat artifacts -> {args.artifacts.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
