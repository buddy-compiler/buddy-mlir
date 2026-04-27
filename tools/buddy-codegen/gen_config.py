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


def load_hf_config(spec: dict, cli_hf_config: str | None) -> dict:
    """Load HuggingFace config.json with cascading fallbacks."""
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
        )


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
        elif variant in ("w8a32", "w8a16"):
            if etype == "i8":
                num_elements = param_counts["linear_elements"]
            else:
                num_elements = (
                    param_counts["other_elements"]
                    + param_counts["linear_output_channels"]
                )
        elif variant == "w8a8":
            if etype == "i8":
                num_elements = param_counts["linear_elements"]
            else:
                num_elements = (
                    param_counts["other_elements"]
                    + param_counts["linear_output_channels"]
                )
        elif variant == "w4a16":
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
    param_counts = count_params(spec)
    weights = compute_weights(variant, param_counts)

    kv_type = precision["kv_type"]

    return {
        "model_family": model_family,
        "variant": variant,
        "model_id": f"{model_family}_{variant}",
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
        "cpp_types": {
            "kv": ELEMENT_TYPE_CPP[kv_type],
            "logits": ELEMENT_TYPE_CPP[precision["logits_type"]],
            "kv_memref": f"MemRef<{ELEMENT_TYPE_CPP[kv_type]}, 4>",
            "logits_memref": f'MemRef<{ELEMENT_TYPE_CPP[precision["logits_type"]]}, 3>',
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
