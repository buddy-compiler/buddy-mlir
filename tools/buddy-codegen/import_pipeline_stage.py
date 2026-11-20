#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
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
"""Export one DeepSeek R1 Distill Qwen FP16 stage with a stage-local KV ABI."""

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from import_model import (
    apply_fusion,
    apply_pre_transforms,
    compile_graphs,
    create_drivers,
    export_mlir,
    export_plain_weights_direct,
    load_model,
    timed_import_step,
)
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

_LAYER_PARAM_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def resolve_model_path(config):
    """Resolve the HuggingFace checkpoint directory or repo ID."""
    return os.environ.get("DEEPSEEKR1_MODEL_PATH") or config["hf_model_path"]


def stage_dtype(config):
    """Return the torch dtype matching config['precision']['activation_type']."""
    activation = config["precision"]["activation_type"]
    if activation == "f16":
        return torch.float16
    if activation == "bf16":
        return torch.bfloat16
    return torch.float32


def validate_stage_interval(hf_config, start, end):
    """Validate [start, end) against model config and return (total, width)."""
    total = hf_config.num_hidden_layers
    if not 0 <= start < end <= total:
        raise ValueError(
            f"invalid layer interval [{start}, {end}) for {total} layers"
        )
    if any(kind != "full_attention" for kind in hf_config.layer_types):
        raise ValueError("stage export requires full attention layers")
    return total, hf_config.hidden_size


def resolve_checkpoint_dir(model_path):
    """Resolve a local checkpoint directory from a path or cached HF repo ID."""
    local = Path(model_path)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise FileNotFoundError(
            f"lazy loading requires a local checkpoint directory: {model_path}"
        ) from err
    return Path(snapshot_download(model_path, local_files_only=True))


def resolve_safetensors_map(model_dir):
    """Map checkpoint tensor names to their safetensors file paths."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_file}")
        return {key: model_dir / shard for key, shard in weight_map.items()}
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        with safe_open(single_file, framework="pt", device="cpu") as reader:
            return dict.fromkeys(reader.keys(), single_file)
    raise FileNotFoundError(
        f"lazy loading requires model.safetensors or "
        f"model.safetensors.index.json in {model_dir}; "
        f"use --load-mode full as fallback"
    )


def checkpoint_key_for_param(
    local_name, start, weight_map, tie_word_embeddings
):
    """Map a stage-local parameter name to its global checkpoint tensor key."""
    match = _LAYER_PARAM_RE.match(local_name)
    if match:
        global_idx = start + int(match.group(1))
        ckpt_key = f"model.layers.{global_idx}.{match.group(2)}"
    else:
        ckpt_key = local_name
    if ckpt_key in weight_map:
        return ckpt_key
    if (
        local_name == "lm_head.weight"
        and tie_word_embeddings
        and "model.embed_tokens.weight" in weight_map
    ):
        return "model.embed_tokens.weight"
    raise KeyError(
        f"checkpoint is missing tensor '{ckpt_key}' for stage parameter "
        f"'{local_name}'"
    )


def select_layers(model, start, end, total):
    """Select a half-open layer interval before Dynamo traces Qwen2.forward."""
    decoder = model.model
    decoder.layers = torch.nn.ModuleList(decoder.layers[start:end])
    model.config.layer_types = model.config.layer_types[start:end]
    model.config.num_hidden_layers = end - start
    for local_idx, layer in enumerate(decoder.layers):
        layer.self_attn.layer_idx = local_idx
    if start:
        decoder.embed_tokens = torch.nn.Identity()
    if end != total:
        decoder.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()


def load_stage_full(config, start, end):
    """Legacy loader: load the full checkpoint, then crop to [start, end)."""
    model = load_model(config)
    total, width = validate_stage_interval(model.config, start, end)
    select_layers(model, start, end, total)
    return model, total, width


def _checkpoint_bucket(ckpt_key):
    """Group keys so non-layer tables load first and each layer unmaps promptly."""
    match = _LAYER_PARAM_RE.match(ckpt_key)
    if match:
        return (1, int(match.group(1)))
    return (0, ckpt_key)


def load_stage_lazy(config, start, end):
    """Build an empty meta model, crop layers, and load only stage tensors."""
    model_path = resolve_model_path(config)
    model_dir = resolve_checkpoint_dir(model_path)
    hf_config = AutoConfig.from_pretrained(model_dir)
    total, width = validate_stage_interval(hf_config, start, end)
    target_dtype = stage_dtype(config)
    tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config)
    select_layers(model, start, end, total)

    weight_map = resolve_safetensors_map(model_dir)
    grouped = defaultdict(lambda: defaultdict(list))
    shards_used = set()
    for local_name, _ in model.named_parameters():
        ckpt_key = checkpoint_key_for_param(
            local_name, start, weight_map, tie_word_embeddings
        )
        shard_path = weight_map[ckpt_key]
        shards_used.add(shard_path)
        grouped[(_checkpoint_bucket(ckpt_key), shard_path)][ckpt_key].append(
            local_name
        )

    state_dict = {}
    loaded_tensors = 0
    for (_, shard_path), key_to_locals in sorted(grouped.items()):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for ckpt_key, local_names in key_to_locals.items():
                tensor = reader.get_tensor(ckpt_key)
                if tensor.dtype != target_dtype:
                    tensor = tensor.to(dtype=target_dtype)
                for local_name in local_names:
                    state_dict[local_name] = tensor
                loaded_tensors += 1

    # Re-instantiate non-persistent RoPE buffers on CPU and cast to match
    # AutoModelForCausalLM.from_pretrained(...).eval().half().
    rotary_cls = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_cls(config=model.config).to(
        dtype=target_dtype
    )

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict

    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise RuntimeError(
            f"lazy stage load left meta tensors unmaterialized: {meta_tensors}"
        )

    model.config.use_cache = False
    model.eval()
    print(
        f"[import] Stage [{start}, {end}) lazy-loaded from {model_dir}: "
        f"{loaded_tensors} tensors from {len(shards_used)} shard(s) "
        f"({config['precision']['activation_type']})",
        file=sys.stderr,
    )
    return model, total, width


def check_parameter_layout(prefill, decode):
    """The two entry points share the exact bytes written in prefill order."""
    if len(prefill._params_ref) != len(decode._params_ref):
        raise ValueError("prefill and decode stage parameter counts differ")
    for index, (left, right) in enumerate(
        zip(prefill._params_ref, decode._params_ref)
    ):
        same_view = (
            left.data_ptr() == right.data_ptr()
            and left.stride() == right.stride()
        )
        if left.shape != right.shape or (
            not same_view and not torch.equal(left, right)
        ):
            raise ValueError(
                f"entry point parameter layouts differ at index {index}"
            )


def export_stage(config, start, end, out, load_mode="lazy"):
    if (
        config["architecture"] != "Qwen2ForCausalLM"
        or config["variant"] != "f16"
        or config.get("prefill_chunk")
        or config.get("arena")
        or config.get("tiered_kv_cache", {}).get("enabled")
        or config.get("decode_pack", {}).get("enabled")
    ):
        raise ValueError(
            "stage export needs the plain FP16 Qwen2 configuration"
        )
    if load_mode not in ("lazy", "full"):
        raise ValueError(f"unsupported load mode: {load_mode}")

    with timed_import_step("load_stage"):
        if load_mode == "lazy":
            model, total, width = load_stage_lazy(config, start, end)
        else:
            model, total, width = load_stage_full(config, start, end)

    with timed_import_step("compile_stage_graphs"):
        (prefill,), (decode,), params = compile_graphs(
            model, config, hidden_input=start > 0
        )

    del model
    torch._dynamo.reset()
    gc.collect()

    apply_pre_transforms(prefill, decode)
    check_parameter_layout(prefill, decode)
    decode._params_ref = prefill._params_ref
    gc.collect()
    apply_fusion(prefill, decode)
    drivers = create_drivers(prefill, decode)
    out.mkdir(parents=True, exist_ok=True)
    export_mlir(*drivers, config, out)
    del drivers, prefill, decode
    gc.collect()

    # "hidden_size" in the legacy config is the size per KV head.
    config["shape"]["num_hidden_layers"] = end - start
    config["shape"]["kv_layers"] = 2 * (end - start)
    config["pipeline_stage"] = {
        "start_layer": start,
        "end_layer": end,
        "total_layers": total,
        "activation_type": config["precision"]["activation_type"],
        "activation_width": width,
    }
    with timed_import_step("export_stage_weights"):
        export_plain_weights_direct(params, config, out)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="zero-based interval [START, END)",
    )
    parser.add_argument(
        "--load-mode",
        choices=("lazy", "full"),
        default="lazy",
        help="checkpoint loading strategy: lazy (stage weights only) or full",
    )
    args = parser.parse_args()
    export_stage(
        json.loads(args.config.read_text()),
        *args.layers,
        args.output_dir,
        load_mode=args.load_mode,
    )


if __name__ == "__main__":
    main()
