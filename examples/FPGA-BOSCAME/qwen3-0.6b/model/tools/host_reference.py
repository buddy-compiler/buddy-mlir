#!/usr/bin/env python3
"""Authoritative FP32 host reference for the fixed 16+8 acceptance case.

This is the transformers Qwen3 implementation running on the official checkpoint
in float32. It is the ground truth for "is the model right", independent of
Buddy, Triton or the board.

It deliberately records more than the final text, because a readable answer is
not evidence of a correct kernel chain:

  * the 16 prefill token ids and the logits for all 16 positions;
  * for each of the 8 decode steps: the input token id, the cache position, the
    selected (argmax) token, the top-k logits, and the logits' hash;
  * optional per-layer hidden states, so a single layer can be compared in
    isolation (stage B) without re-deriving the reference.

The decode loop feeds back *its own* argmax, so the reference token trajectory is
the model's own. A later stage that compares a different trajectory is comparing
different inputs and must say so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def build_model(assets, checkpoint, dtype_name="float32", layers=None):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(assets)
    if layers is not None:
        config.num_hidden_layers = layers
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[:layers]
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
    model.requires_grad_(False)
    state = load_file(str(Path(checkpoint) / "model.safetensors"))
    model.load_state_dict(state, strict=False)
    return model, config


def tensor_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def top_k(logits, k):
    order = np.argsort(-logits)
    return [[int(i), float(logits[i])] for i in order[:k]]


def error_stats(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    diff = np.abs(reference - candidate)
    denominator = np.maximum(np.abs(reference), 1e-30)
    cosine = float(np.dot(reference, candidate) /
                   (np.linalg.norm(reference) * np.linalg.norm(candidate) + 1e-30))
    return {
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "max_relative_error": float((diff / denominator).max()) if diff.size else 0.0,
        "cosine_similarity": cosine,
        "elements": int(diff.size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", required=True,
                        help="comma-separated token ids for the 16-token prefill")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--capture-layer", type=int, default=None,
                        help="also dump the output hidden state of this layer")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    import torch
    from transformers import StaticCache

    prompt_ids = [int(v) for v in args.prompt_ids.split(",") if v.strip()]
    model, config = build_model(args.assets, args.checkpoint, args.dtype, args.layers)

    captured = {}
    handles = []
    if args.capture_layer is not None:
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["layer_output"] = hidden.detach().to(torch.float32).numpy()
        handles.append(model.model.layers[args.capture_layer]
                       .register_forward_hook(hook))

    report = {
        "reference": "transformers Qwen3, official checkpoint",
        "dtype": args.dtype,
        "prompt_ids": prompt_ids,
        "prefill_len": len(prompt_ids),
        "decode_steps": args.decode_steps,
        "max_cache_len": args.max_cache_len,
        "top_k": args.top_k,
        "config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "head_dim": getattr(config, "head_dim", None),
            "rms_norm_eps": config.rms_norm_eps,
            "vocab_size": config.vocab_size,
        },
    }

    cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len,
                        batch_size=1)
    arrays = {}
    with torch.no_grad():
        ids = torch.tensor([prompt_ids], dtype=torch.int64)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    cache_implementation="static",
                    cache_position=torch.arange(len(prompt_ids)))
        prefill_logits = out.logits.to(torch.float32).numpy()
        arrays["prefill_logits"] = prefill_logits
        report["prefill_logits_last"] = top_k(prefill_logits[0, -1], args.top_k)
        report["prefill_argmax_all"] = [int(v) for v in
                                        np.argmax(prefill_logits[0], axis=-1)]
        report["prefill_logits_sha256"] = tensor_digest(prefill_logits)

        steps = []
        next_token = int(np.argmax(prefill_logits[0, -1]))
        for step in range(args.decode_steps):
            position = len(prompt_ids) + step
            step_ids = torch.tensor([[next_token]], dtype=torch.int64)
            out = model(input_ids=step_ids, past_key_values=cache, use_cache=True,
                        cache_implementation="static",
                        cache_position=torch.tensor([position], dtype=torch.int64))
            logits = out.logits.to(torch.float32).numpy()[0, -1]
            arrays[f"decode_logits_{step}"] = logits
            chosen = int(np.argmax(logits))
            steps.append({
                "step": step,
                "input_token": next_token,
                "cache_position": position,
                "generated_token": chosen,
                "top_k": top_k(logits, args.top_k),
                "logits_sha256": tensor_digest(logits),
                "logits_max_abs": float(np.abs(logits).max()),
            })
            next_token = chosen
        report["decode_steps_recorded"] = steps
        report["generated_ids"] = [s["generated_token"] for s in steps]

        if "layer_output" in captured:
            arrays["layer_output"] = captured["layer_output"]
            report["captured_layer"] = args.capture_layer
            report["layer_output_shape"] = list(captured["layer_output"].shape)

        # Final KV cache state, so a deployment can be checked for cache
        # divergence and not only for a matching next token.
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache:
            report["kv_cache_layers"] = len(key_cache)
            report["kv_cache_shape"] = list(key_cache[0].shape)
            used = len(prompt_ids) + args.decode_steps
            stacked_k = np.stack([t[0].to(torch.float32).numpy()
                                  for t in key_cache])
            stacked_v = np.stack([t[0].to(torch.float32).numpy()
                                  for t in value_cache])
            arrays["kv_key_used"] = stacked_k[:, :, :used, :]
            arrays["kv_value_used"] = stacked_v[:, :, :used, :]
            report["kv_used_positions"] = used
            report["kv_key_used_sha256"] = tensor_digest(arrays["kv_key_used"])

    for handle in handles:
        handle.remove()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("decode_steps_recorded",)}, indent=2)[:4000])
    print("\ndecode trajectory:")
    for step in report["decode_steps_recorded"]:
        print(f"  step {step['step']} pos={step['cache_position']} "
              f"in={step['input_token']} out={step['generated_token']} "
              f"top1={step['top_k'][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
