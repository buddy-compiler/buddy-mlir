#!/usr/bin/env python3
# ===- gen_config.py - Generate full model config from variant spec --------===//
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
# Reads a minimal variant spec JSON and a HuggingFace model config to produce
# a complete model configuration that drives all downstream code generators.
#
# Usage:
#   python gen_config.py --spec specs/f32.json -o deepseek_r1_f32.json
#   python gen_config.py --spec specs/f32.json --hf-config /path/to/config.json -o out.json
#
# ===----------------------------------------------------------------------===//

import argparse
import glob
import json
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Static knowledge tables — one entry per supported variant
# ──────────────────────────────────────────────────────────────────────────────

VARIANT_PRECISION = {
    "f32": {"kv_type": "f32", "logits_type": "f32", "activation_type": "f32"},
    "f16": {"kv_type": "f16", "logits_type": "f16", "activation_type": "f16"},
    "bf16": {
        "kv_type": "bf16",
        "logits_type": "bf16",
        "activation_type": "bf16",
    },
    "w8a32": {"kv_type": "f32", "logits_type": "f32", "activation_type": "f32"},
    "w8a16": {"kv_type": "f16", "logits_type": "f16", "activation_type": "f16"},
    "w8a8": {"kv_type": "f32", "logits_type": "f32", "activation_type": "f32"},
    "w4a16": {"kv_type": "f16", "logits_type": "f16", "activation_type": "f16"},
    # int4 in 32-row groups, kernels for RVV VLEN 1024 (graph/transform/k3_w4.py)
    "w4g32": {"kv_type": "f32", "logits_type": "f32", "activation_type": "f32"},
}

VARIANT_WEIGHT_TEMPLATES = {
    "f32": [{"tag": "params", "suffix": "", "element_type": "f32"}],
    "f16": [{"tag": "params", "suffix": "-f16", "element_type": "f16"}],
    "bf16": [{"tag": "params", "suffix": "-bf16", "element_type": "bf16"}],
    "w8a32": [
        {"tag": "f32_params", "suffix": "-w8a32-f32", "element_type": "f32"},
        {"tag": "i8_params", "suffix": "-w8a32-i8", "element_type": "i8"},
    ],
    "w8a16": [
        {"tag": "f16_params", "suffix": "-w8a16-f16", "element_type": "f16"},
        {"tag": "i8_params", "suffix": "-w8a16-i8", "element_type": "i8"},
    ],
    "w8a8": [
        {"tag": "f32_params", "suffix": "-w8a8-f32", "element_type": "f32"},
        {"tag": "i8_params", "suffix": "-w8a8-i8", "element_type": "i8"},
    ],
    "w4a16": [
        {"tag": "f16_params", "suffix": "-w4a16-f16", "element_type": "f16"},
        {"tag": "i4_params", "suffix": "-w4a16-i4packed", "element_type": "i8"},
    ],
    "w4g32": [
        {"tag": "f32_params", "suffix": "-w4g32-f32", "element_type": "f32"},
        {"tag": "i8_params", "suffix": "-w4g32-q4", "element_type": "i8"},
    ],
}

ELEMENT_TYPE_CPP = {
    "f32": "float",
    "f16": "uint16_t",
    "bf16": "uint16_t",
    "i8": "int8_t",
}
ELEMENT_TYPE_MLIR = {"f32": "f32", "f16": "f16", "bf16": "bf16", "i8": "i8"}
ELEMENT_TYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i8": 1}


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace config loading (multiple fallback strategies)
# ──────────────────────────────────────────────────────────────────────────────


def _find_cached_hf_config(model_path: str) -> str | None:
    """Search the HuggingFace cache for a model's config.json."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    slug = f"models--{model_path.replace('/', '--')}"
    pattern = os.path.join(cache_root, slug, "snapshots", "*", "config.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def text_decoder_config(hf: dict) -> dict:
    """The HF config of the model buddy-codegen builds: for a Qwen3-VL-MoE
    model (Qwen3VLMoeForConditionalGeneration), its text decoder as a
    Qwen3MoeForCausalLM, which it is for text-only input (the interleaved
    MRoPE of three equal positions is the RoPE of that position); any other
    config unchanged."""
    if (hf.get("architectures") or [None])[0] != (
        "Qwen3VLMoeForConditionalGeneration"
    ):
        return hf
    text = dict(hf["text_config"])
    text.pop("rope_scaling", None)
    rope = {
        k: v
        for k, v in text.get("rope_parameters", {}).items()
        if not k.startswith("mrope")
    }
    if rope:
        text["rope_parameters"] = rope
        text.setdefault("rope_theta", rope.get("rope_theta"))
    text.update(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        tie_word_embeddings=hf.get("tie_word_embeddings", False),
    )
    return text


def load_hf_config(spec: dict, cli_hf_config: str | None) -> dict:
    """Load HuggingFace config.json with cascading fallbacks; a
    vision-language model gives its text decoder (text_decoder_config)."""
    return text_decoder_config(_load_hf_config(spec, cli_hf_config))


def _load_hf_config(spec: dict, cli_hf_config: str | None) -> dict:
    # 1. Explicit CLI path
    if cli_hf_config:
        with open(cli_hf_config) as f:
            return json.load(f)

    # 2. Path specified in spec
    if "hf_config_path" in spec:
        with open(spec["hf_config_path"]) as f:
            return json.load(f)

    model_path = spec["hf_model_path"]

    # 3. Try transformers library
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path)
        return {k: v for k, v in cfg.to_dict().items() if not k.startswith("_")}
    except ImportError:
        pass

    # 4. Search local HF cache
    cached = _find_cached_hf_config(model_path)
    if cached:
        print(f"[gen_config] Using cached HF config: {cached}", file=sys.stderr)
        with open(cached) as f:
            return json.load(f)

    raise RuntimeError(
        f"Cannot load HF config for '{model_path}'.\n"
        "  Options:\n"
        "    1. Install transformers: pip install transformers\n"
        "    2. Pass --hf-config /path/to/config.json\n"
        "    3. Add hf_config_path to your spec JSON"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ──────────────────────────────────────────────────────────────────────────────


def _count_params_torch(model_path: str) -> dict:
    """Instantiate model from config (no pretrained weights) and count params."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(cfg)

    total = 0
    linear_elements = 0
    linear_output_channels = 0
    other_elements = 0

    for name, p in model.named_parameters():
        total += p.numel()
        is_linear_weight = (
            p.dim() == 2
            and "embed" not in name
            and "norm" not in name
            and "lm_head" not in name
        )
        if is_linear_weight:
            linear_elements += p.numel()
            linear_output_channels += p.shape[0]
        else:
            other_elements += p.numel()

    del model
    return {
        "total": total,
        "linear_elements": linear_elements,
        "linear_output_channels": linear_output_channels,
        "other_elements": other_elements,
    }


def count_params(spec: dict) -> dict:
    """Count parameters, using torch if available, otherwise from overrides."""
    overrides = spec.get("weights_override", {})
    if overrides:
        return {
            "total": overrides.get("total", 0),
            "linear_elements": overrides.get("linear_elements", 0),
            "linear_output_channels": overrides.get(
                "linear_output_channels", 0
            ),
            "other_elements": overrides.get("other_elements", 0),
        }

    try:
        counts = _count_params_torch(spec["hf_model_path"])
        print(
            f"[gen_config] Param count from model: {counts['total']:,}",
            file=sys.stderr,
        )
        return counts
    except ImportError:
        raise RuntimeError(
            "Cannot count model parameters (torch not installed).\n"
            "  Options:\n"
            "    1. Install torch: pip install torch\n"
            "    2. Add weights_override to your spec JSON, e.g.:\n"
            '       "weights_override": {"total": 1777088064}'
        ) from None


# ──────────────────────────────────────────────────────────────────────────────
# Weight layout computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_weights(variant: str, param_counts: dict) -> list[dict]:
    """Build the weights descriptor list for the given variant."""
    templates = VARIANT_WEIGHT_TEMPLATES.get(
        variant, VARIANT_WEIGHT_TEMPLATES["f32"]
    )
    weights = []

    for tmpl in templates:
        tag = tmpl["tag"]
        etype = tmpl["element_type"]

        if variant in ("f32", "f16", "bf16"):
            num_elements = param_counts["total"]
        elif variant == "w4g32":
            num_elements = param_counts[f"{etype}_elements"]
        elif (
            variant in ("w8a32", "w8a16")
            or variant == "w8a8"
            or variant == "w4a16"
        ):
            if etype == "i8":
                num_elements = param_counts["linear_elements"]
            else:
                num_elements = (
                    param_counts["other_elements"]
                    + param_counts["linear_output_channels"]
                )
        else:
            num_elements = param_counts.get("total", 0)

        weights.append(
            {
                "tag": tag,
                "file": f"arg0{tmpl['suffix']}.data",
                "element_type": etype,
                "cpp_type": ELEMENT_TYPE_CPP[etype],
                "mlir_type": ELEMENT_TYPE_MLIR[etype],
                "bytes_per_element": ELEMENT_TYPE_BYTES[etype],
                "num_elements": num_elements,
            }
        )

    return weights


# What the w4g32 kernels (graph/transform/k3_w4.py) require: the prefill
# attention kernel takes the rows of a chunk 32 at a time (ATTN_ROWS) and the
# head dimensions 16 at a time (ATTN_DIMS).
W4G32_CHUNK_MULTIPLE = 32
W4G32_HEAD_DIM_MULTIPLE = 16


# "prefill_ime": the rows of a matrix-engine prefill tile (k3_w4.IME_ROWS).
W4G32_IME_ROWS = 64


def derive_prefill_ime(spec: dict, variant: str, prefill_chunk: int) -> bool:
    """Opt-in "prefill_ime" (docs/K3DeepSeekR1.md): the prefill tiles of
    w4g32 run on the matrix engine of the SpacemiT K3 A100 cores, on a second
    copy of the weights in the IME layout."""
    raw = spec.get("prefill_ime", False)
    if not isinstance(raw, bool):
        raise ValueError(f"prefill_ime must be true or false, got {raw!r}")
    if raw and variant != "w4g32":
        raise ValueError("prefill_ime needs the variant w4g32")
    if raw and prefill_chunk != W4G32_IME_ROWS:
        raise ValueError(
            f"prefill_ime needs prefill_chunk {W4G32_IME_ROWS}, "
            f"got {prefill_chunk}"
        )
    return raw


# "kernels": which collection of w4g32 kernels the model uses: "k3"
# (graph/transform/k3_w4.py, docs/K3DeepSeekR1.md) or "a210"
# (graph/transform/a210_w4.py, docs/A210DeepSeekR1.md).
W4G32_KERNELS = ("k3", "a210")


def derive_kernels(spec: dict, variant: str) -> str | None:
    """The w4g32 kernel collection (default "k3"); None for other variants."""
    raw = spec.get("kernels")
    if variant != "w4g32":
        if raw is not None:
            raise ValueError("kernels needs the variant w4g32")
        return None
    raw = "k3" if raw is None else raw
    if raw not in W4G32_KERNELS:
        raise ValueError(f"kernels must be one of {W4G32_KERNELS}, got {raw!r}")
    if raw == "a210" and spec.get("prefill_ime"):
        raise ValueError("prefill_ime needs the kernels k3")
    return raw


def require_pinned_threads(spec: dict, field: str) -> None:
    """Per-thread fields mean something only when thread i stays on one
    known CPU: the thread pool pins it to the i-th CPU the process may run
    on, libomp threads may migrate."""
    if spec.get("thread_pool") is not True:
        raise ValueError(f"{field} needs thread_pool true")


def derive_thread_weights(spec: dict, kernels: str | None) -> dict | None:
    """Opt-in "thread_weights" of the A210 kernels: {"prefill": [...],
    "decode": [...]}, one positive weight per thread (num_threads), the share
    of each kernel's work thread i gets; a missing list means equal shares."""
    raw = spec.get("thread_weights")
    if raw is None:
        return None
    if kernels != "a210":
        raise ValueError("thread_weights needs the kernels a210")
    require_pinned_threads(spec, "thread_weights")
    threads = spec.get("num_threads", 48)
    if not isinstance(raw, dict) or set(raw) - {"prefill", "decode"}:
        raise ValueError(
            'thread_weights must be {"prefill": [...], "decode": [...]}'
        )
    for key, ws in raw.items():
        if (
            not isinstance(ws, list)
            or len(ws) != threads
            or not all(isinstance(w, (int, float)) and w > 0 for w in ws)
        ):
            raise ValueError(
                f"thread_weights {key} must be {threads} positive numbers "
                f"(num_threads), got {ws!r}"
            )
    return {key: raw.get(key) for key in ("prefill", "decode")}


# "thread_cores": the cores the A210 kernels know (a210_w4.TILE_SHAPES).
A210_CORES = ("c908", "c920")


def derive_thread_cores(spec: dict, kernels: str | None) -> list | None:
    """Opt-in "thread_cores" of the A210 kernels: the core of each thread
    (num_threads entries, "c908" or "c920"), which picks the shape of its
    prefill tiles; none: the C920 shape for all."""
    raw = spec.get("thread_cores")
    if raw is None:
        return None
    if kernels != "a210":
        raise ValueError("thread_cores needs the kernels a210")
    require_pinned_threads(spec, "thread_cores")
    threads = spec.get("num_threads", 48)
    if (
        not isinstance(raw, list)
        or len(raw) != threads
        or not all(c in A210_CORES for c in raw)
    ):
        raise ValueError(
            f"thread_cores must be {threads} of {A210_CORES} (num_threads), "
            f"got {raw!r}"
        )
    return list(raw)


def w4g32_param_counts(hf: dict, prefill_chunk: int, ime: bool) -> dict:
    """w4g32 (graph/transform/k3_w4.py) buffer sizes, from the HF config of a
    Qwen2 model: every Linear weight [K, N] becomes K * N * 9 / 16 bytes of
    int4 tiles with their f16 scales; the embedding, the norms, the q / k / v
    biases and the RoPE inv_freq stay f32. With `ime` ("prefill_ime"), the
    weights of the layers other than the LM head (whose prefill computes one
    row) get a second copy, in the IME layout, of the same size. Rejects the
    models and the prefill_chunk (derive_prefill_chunk) the kernels do not
    support."""
    arch = (hf.get("architectures") or ["?"])[0]
    if arch not in ("Qwen2ForCausalLM", "Qwen3MoeForCausalLM"):
        raise ValueError(
            f"w4g32 supports Qwen2ForCausalLM and Qwen3MoeForCausalLM "
            f"models, not {arch}"
        )
    moe = arch == "Qwen3MoeForCausalLM"
    if hf.get("tie_word_embeddings"):
        raise ValueError("w4g32 does not support tie_word_embeddings")
    if not prefill_chunk:
        raise ValueError("w4g32 needs prefill_chunk")
    if prefill_chunk % W4G32_CHUNK_MULTIPLE:
        raise ValueError(
            f"w4g32 needs prefill_chunk to be a multiple of "
            f"{W4G32_CHUNK_MULTIPLE}, got {prefill_chunk}"
        )
    h = hf["hidden_size"]
    heads = hf["num_attention_heads"]
    d = hf.get("head_dim", h // heads)
    if d % W4G32_HEAD_DIM_MULTIPLE:
        raise ValueError(
            f"w4g32 needs a head_dim that is a multiple of "
            f"{W4G32_HEAD_DIM_MULTIPLE}, got {d}"
        )
    kv = hf.get("num_key_value_heads", heads) * d
    vocab = hf["vocab_size"]
    layers = hf["num_hidden_layers"]
    linears = [
        (h, heads * d),  # q, k, v: one weight, one tile range each
        (h, kv),
        (h, kv),
        (heads * d, h),  # o
    ]
    if moe:
        # Qwen3-MoE: every layer a MoE block (k3_w4.pack_moe); the router
        # [h, experts] stays f32
        if hf.get("mlp_only_layers") or hf.get("decoder_sparse_step", 1) != 1:
            raise ValueError("w4g32 needs a MoE block in every layer")
        if not hf.get("norm_topk_prob", False):
            # the kernel computes the routing: the top-k weights normalized
            raise ValueError("w4g32 needs norm_topk_prob")
        inter = hf["moe_intermediate_size"]
        # transformers 5 writes num_experts as num_local_experts
        experts = hf.get("num_experts", hf.get("num_local_experts"))
        expert_linears = [(h, inter), (h, inter), (inter, h)]
        if h % 128:
            raise ValueError(f"w4g32 needs hidden_size % 128 == 0, got {h}")
    else:
        inter = hf["intermediate_size"]
        experts = 0
        linears += [(h, inter), (h, inter), (inter, h)]  # gate, up, down
        expert_linears = []
    for k, n in linears + expert_linears + [(h, vocab)]:
        # N % 128: the K3 tiles (the A210 ones need N % 16)
        if k % 32 or n % 128:
            raise ValueError(
                f"w4g32 needs Linear weights [K, N] with K % 32 == 0 and "
                f"N % 128 == 0, got [{k}, {n}]"
            )
    # with ime, the dense layers get a second, IME-layout copy (not the
    # experts: the MoE kernel has no matrix-engine tiles)
    per_layer = sum(k * n for k, n in linears) * (2 if ime else 1)
    per_layer += experts * sum(k * n for k, n in expert_linears)
    i8 = (layers * per_layer + h * vocab) * 9 // 16
    f32_layer = 2 * h
    if moe:
        f32_layer += 2 * d + h * experts  # q_norm, k_norm, router
    else:
        f32_layer += heads * d + 2 * kv  # the q / k / v biases
    f32 = vocab * h + layers * f32_layer + h + d // 2
    return {"f32_elements": f32, "i8_elements": i8}


def derive_tiered_kv_cache(spec: dict) -> dict:
    """Return tiered KV cache settings from the variant spec."""
    raw = spec.get("tiered_kv_cache", False)
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", True))
        cache_sizes = raw.get("cache_sizes", spec.get("cache_sizes", []))
    else:
        enabled = bool(raw)
        cache_sizes = spec.get("cache_sizes", [])

    if not enabled:
        return {"enabled": False, "cache_sizes": []}

    if isinstance(cache_sizes, str):
        cache_sizes = [
            int(x.strip()) for x in cache_sizes.split(",") if x.strip()
        ]
    else:
        cache_sizes = [int(x) for x in cache_sizes]

    if not cache_sizes:
        cache_sizes = [32, 64, 128, 256, 512, 1024]

    cache_sizes = sorted(dict.fromkeys(cache_sizes))
    return {"enabled": True, "cache_sizes": cache_sizes}


# ──────────────────────────────────────────────────────────────────────────────
# Shape / token derivation
# ──────────────────────────────────────────────────────────────────────────────


def derive_shapes(hf: dict, spec: dict) -> dict:
    num_attention_heads = hf["num_attention_heads"]
    hidden_size = hf["hidden_size"]
    head_dim = hf.get("head_dim", hidden_size // num_attention_heads)
    return {
        "head_num": hf["num_key_value_heads"],
        "hidden_size": head_dim,
        "kv_layers": hf["num_hidden_layers"] * 2,
        "vocab_size": hf["vocab_size"],
        "max_token_len": spec.get("max_token_len", 1024),
        "num_hidden_layers": hf["num_hidden_layers"],
    }


DEFAULT_PREFILL_CHUNK = 64


def derive_prefill_chunk(spec: dict, shape: dict, tiered: bool) -> int:
    """Opt-in chunked prefill: the number of prompt tokens per forward_prefill
    call, or 0 (off, the default) for one call over max_token_len positions.

    The spec sets `prefill_chunk` to a positive integer, or to `true` for
    DEFAULT_PREFILL_CHUNK. When on, import_model.py traces forward_prefill
    like forward_decode, with `prefill_chunk` tokens and their positions in
    and the KV cache in and out, and the generated ModelSession::prefill()
    walks the prompt chunk by chunk (see docs/ChunkedPrefill.md). Prefill
    then costs about as much as the prompt is long, rounded up to whole
    chunks, instead of always max_token_len positions.
    """
    raw = spec.get("prefill_chunk", 0)
    if raw is True:
        chunk = DEFAULT_PREFILL_CHUNK
    elif raw is False or raw is None:
        chunk = 0
    elif isinstance(raw, int) and raw >= 0:
        chunk = raw
    else:
        raise ValueError(
            f"prefill_chunk must be true/false or a non-negative integer, "
            f"got {raw!r}"
        )
    if chunk == 0:
        return 0
    if chunk > shape["max_token_len"]:
        raise ValueError(
            f"prefill_chunk ({chunk}) must not exceed max_token_len "
            f"({shape['max_token_len']})"
        )
    if tiered:
        raise ValueError(
            "prefill_chunk and tiered_kv_cache are mutually exclusive"
        )
    return chunk


def derive_memory_options(spec: dict, tiered: bool, prefill_chunk: int) -> dict:
    """Opt-in memory options of the model library and session, both off by
    default (see docs/ModelMemoryOptions.md):

    arena      every buffer the model allocates during a forward call comes
               from a bump arena (runtime/arena/BuddyArena.c, linked into the
               model library); frees are no-ops and the session resets the
               arena before each call. compile_pipeline.py lowers the
               allocations to the generic MLIR allocation functions and drops
               the buffer deallocation passes. The arena holds all the buffers
               of a call, so it needs chunked prefill: one prefill call over
               max_token_len positions allocates tens of GiB (42 GiB for f32
               DeepSeek-R1-Distill-Qwen-1.5B).
    hugepages  the session asks for 2 MiB pages (madvise MADV_HUGEPAGE) for
               the weight buffers before reading the weights in.
    """
    options = {}
    for key in ("arena", "hugepages"):
        raw = spec.get(key, False)
        if not isinstance(raw, bool):
            raise ValueError(f"{key} must be true or false, got {raw!r}")
        options[key] = raw
    if options["arena"] and tiered:
        raise ValueError("arena and tiered_kv_cache are mutually exclusive")
    if options["arena"] and not prefill_chunk:
        raise ValueError(
            "arena needs prefill_chunk: a prefill call over max_token_len "
            "positions allocates too much to keep all of it"
        )
    return options


def derive_thread_pool(spec: dict) -> bool:
    """Opt-in "thread_pool": the model library runs its parallel loops on a
    pinned pool of spinning threads (runtime/threadpool/BuddyThreadPool.c,
    linked in place of libomp by buddy_model.cmake; docs/ModelThreadPool.md).
    """
    raw = spec.get("thread_pool", False)
    if not isinstance(raw, bool):
        raise ValueError(f"thread_pool must be true or false, got {raw!r}")
    return raw


def derive_decode_pack(hf: dict, spec: dict) -> dict:
    """Opt-in panel-packing of the decode matmul weights (see the
    pack_decode_matmul_weights graph transform). Off unless the spec sets
    `decode_pack_vector_size`, so existing specs are unaffected.

    When on, import_model.py runs the pack_decode_matmul_weights graph
    transform over the decode graph -- rewriting every matmul weight into
    N-tile panel layout -- and writes decode's parameters to their own file.
    compile_pipeline.py then compiles decode with
    -matmul-vectorization-decode-packed instead of the plain row-major kernel.

    Decode needs its own file because the two phases want opposite layouts:
    decode is a GEMV that reads each weight byte once (so the layout is what
    limits it), prefill is compute-bound and its matmul kernel expects plain
    row-major B. Handing prefill the packed bytes does not fail loudly; it just
    reads them as row-major and produces fluent, wrong output.

    Packing is a permutation of each weight's bytes: same shape, same offsets,
    same element count. So this is not an extra weight blob -- it is the same
    blob with a second file, and the only thing that differs between the phases
    is which of the two they are handed.
    """
    vecsize = spec.get("decode_pack_vector_size")
    if not vecsize:
        return {"enabled": False}

    return {
        "enabled": True,
        "vector_size": vecsize,
        # Divisibility of every weight's N by vecsize is checked by the graph
        # transform, which is the only thing that knows the full weight set --
        # it covers lm_head and the attention projections, not just the FFN.
        "decode_file": "arg0-decode.data",
    }


def derive_tokens(hf: dict, spec: dict) -> dict:
    return {
        "eos_id": hf.get("eos_token_id", 0),
        "rope_theta": hf.get("rope_theta", 10000.0),
        "tokenizer": spec.get("model_family", "unknown"),
        "vocab_file": spec.get("vocab_file", "vocab.txt"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────


def gen_config(spec: dict, hf_config_path: str | None = None) -> dict:
    """Generate a complete model configuration from a variant spec."""
    hf = load_hf_config(spec, hf_config_path)

    variant = spec["variant"]
    model_family = spec.get("model_family", "unknown")

    shape = derive_shapes(hf, spec)
    tokens = derive_tokens(hf, spec)
    precision = VARIANT_PRECISION.get(variant, VARIANT_PRECISION["f32"])
    tiered_kv_cache = derive_tiered_kv_cache(spec)
    prefill_chunk = derive_prefill_chunk(
        spec, shape, tiered_kv_cache["enabled"]
    )
    param_counts = (
        w4g32_param_counts(
            hf,
            prefill_chunk,
            derive_prefill_ime(spec, variant, prefill_chunk),
        )
        if variant == "w4g32"
        else count_params(spec)
    )
    weights = compute_weights(variant, param_counts)
    decode_pack = derive_decode_pack(hf, spec)
    memory_options = derive_memory_options(
        spec, tiered_kv_cache["enabled"], prefill_chunk
    )
    if decode_pack["enabled"] and variant not in ("f32", "f16", "bf16"):
        raise RuntimeError(
            f"decode_pack_vector_size is only supported for f32/f16/bf16 "
            f"variants (got variant={variant!r})"
        )
    if decode_pack["enabled"] and tiered_kv_cache["enabled"]:
        # Refuse here rather than at import time: only gen_impl was taught to
        # hand decode a different weight buffer, so gen_impl_tiered would
        # silently give decode the plain weights and compile it with the packed
        # kernel -- fluent, wrong output, nothing to catch it.
        raise RuntimeError(
            "decode_pack_vector_size is not supported together with "
            "tiered_kv_cache"
        )
    if decode_pack["enabled"]:
        # Not a second blob: the same blob with a second file. weights[0]
        # ("file") stays plain for prefill; decode is handed "decode_file".
        weights[0]["decode_file"] = decode_pack["decode_file"]

    if tiered_kv_cache["enabled"]:
        if variant != "f32":
            raise RuntimeError(
                "tiered_kv_cache is currently implemented for the f32 "
                "DeepSeek R1 variant only."
            )
        if tiered_kv_cache["cache_sizes"][-1] != shape["max_token_len"]:
            raise RuntimeError(
                "The largest tiered KV cache size must equal max_token_len "
                f"({shape['max_token_len']}); got "
                f"{tiered_kv_cache['cache_sizes'][-1]}."
            )
        # Keep compatibility with the legacy tiered example artifact name.
        if len(weights) == 1:
            weights[0]["file"] = spec.get("weights_file", "arg0_mc.data")

    kv_type = precision["kv_type"]
    model_id = f"{model_family}_{variant}"
    if tiered_kv_cache["enabled"]:
        model_id = f"{model_id}_tiered_kv_cache"
    if decode_pack["enabled"]:
        model_id = f"{model_id}_packed_ffn"
    if prefill_chunk:
        model_id = f"{model_id}_prefill_chunk{prefill_chunk}"

    return {
        "model_family": model_family,
        "variant": variant,
        "model_id": model_id,
        "hf_model_path": spec["hf_model_path"],
        "architecture": (
            hf["architectures"][0]
            if "architectures" in hf
            else hf.get("model_type", "unknown")
        ),
        "shape": shape,
        "precision": precision,
        "weights": weights,
        "tokens": tokens,
        "tiered_kv_cache": tiered_kv_cache,
        "decode_pack": decode_pack,
        "arena": memory_options["arena"],
        "hugepages": memory_options["hugepages"],
        "thread_pool": derive_thread_pool(spec),
        "prefill_ime": derive_prefill_ime(spec, variant, prefill_chunk),
        "kernels": derive_kernels(spec, variant),
        "thread_weights": derive_thread_weights(
            spec, derive_kernels(spec, variant)
        ),
        "thread_cores": derive_thread_cores(
            spec, derive_kernels(spec, variant)
        ),
        # >0: forward_prefill takes this many prompt tokens per call
        "prefill_chunk": prefill_chunk,
        "cpp_types": {
            "kv": ELEMENT_TYPE_CPP[kv_type],
            "logits": ELEMENT_TYPE_CPP[precision["logits_type"]],
            "kv_memref": f"MemRef<{ELEMENT_TYPE_CPP[kv_type]}, 4>",
            "logits_memref": f"MemRef<{ELEMENT_TYPE_CPP[precision['logits_type']]}, 3>",
        },
        "mlir_types": {
            "kv": ELEMENT_TYPE_MLIR[kv_type],
            "logits": ELEMENT_TYPE_MLIR[precision["logits_type"]],
        },
        "compilation": {
            "num_threads": spec.get("num_threads", 48),
            "so_name": f"{model_family}_model.so",
            "pipelines": {
                "forward_prefill": "standard",
                "subgraph_prefill": "subgraph",
                "forward_decode": "standard",
                "subgraph_decode": "subgraph_decode",
                # with prefill_ime (the K3 A100 cores), all the kernels are
                # compiled for the A100
                **(
                    {
                        "k3_kernels": (
                            "kernels_a100"
                            if derive_prefill_ime(spec, variant, prefill_chunk)
                            else "kernels"
                        )
                    }
                    if derive_kernels(spec, variant) == "k3"
                    else {}
                ),
                **(
                    {"a210_kernels": "kernels"}
                    if derive_kernels(spec, variant) == "a210"
                    else {}
                ),
                **(
                    {"k3_kernels_ime": "kernels_ime"}
                    if derive_prefill_ime(spec, variant, prefill_chunk)
                    else {}
                ),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate full model config from a variant spec + HuggingFace config."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to variant spec JSON"
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Path to HF config.json (auto-detected if omitted)",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    config = gen_config(spec, args.hf_config)
    output_text = json.dumps(config, indent=2) + "\n"

    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"[gen_config] Written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
