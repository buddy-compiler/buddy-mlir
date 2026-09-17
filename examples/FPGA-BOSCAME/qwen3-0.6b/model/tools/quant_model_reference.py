#!/usr/bin/env python3
"""W8A8 host reference for the fixed 16+8 acceptance case.

This follows the *imported graph's* operation order, not a generic transformer
formula, so that an FPGA result can be compared against it op for op. It exists
to separate two very different error sources:

  1. FPGA vs this reference      -> compilation / kernel / chain correctness;
  2. this reference vs the FP32  -> error introduced by quantization itself.

Both are reported separately; mixing them would hide which side is wrong.

Quantization contract (identical to the verified Triton quantize/dequantize and
integer linear kernels, and to tools/quant_reference.py):

  weights      per-output-channel symmetric signed int8, range [-127, 127],
               scale = max|row| / 127, zero rows use scale 1;
  activations  per-token symmetric signed int8, dynamic, same rounding;
  rounding     FP32 divide, add +/-0.5 by sign, truncate, saturate;
  accumulator  int32;
  dequantize   (acc_f32 * activation_scale) * weight_scale, in that order;
  everything else (RMSNorm, RoPE, attention, softmax, SiLU, residuals) stays
               FP32 and is not quantized.

The shared embedding / lm_head matrix is quantized once and used for both the
gather and the output projection, because the compiled graph reads a single
buffer for both.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_reference import linear_w8a8, quantize_rows  # noqa: E402


def rms_norm(value, weight, eps, profile="numpy-graph"):
    """Pow/Mean/Add(eps)/Rsqrt/Mul/Mul(weight), the graph's own chain."""
    squares = np.square(value.astype(np.float32))
    if profile in ("triton-host", "nr-fpga"):
        from reference_fp32 import ordered_sum
        mean = ordered_sum(squares, keepdims=True) / np.float32(value.shape[-1])
    else:
        mean = np.mean(squares, axis=-1, keepdims=True, dtype=np.float32)
    denominator = np.sqrt(mean + np.float32(eps), dtype=np.float32)
    return (value.astype(np.float32) / denominator) * weight.astype(np.float32)


def inv_freq(theta, head_dim):
    index = np.arange(0, head_dim, 2, dtype=np.float64)
    return (1.0 / (theta ** (index / head_dim))).astype(np.float32)


def rope_tables(positions, frequencies, profile="numpy-graph"):
    """freqs[s,k] = position * inv_freq[k], duplicated to head_dim."""
    angles = np.outer(positions.astype(np.float32), frequencies.astype(np.float32))
    doubled = np.concatenate([angles, angles], axis=-1)
    if profile in ("triton-host", "nr-fpga"):
        from reference_fp32 import unary
        return unary("cosf", doubled, profile), unary("sinf", doubled, profile)
    return np.cos(doubled, dtype=np.float32), np.sin(doubled, dtype=np.float32)


def apply_rope(value, cosine, sine):
    """Split-half rotation, matching the graph's slice/neg/cat/mul/add chain."""
    half = value.shape[-1] // 2
    first = value[..., :half]
    second = value[..., half:]
    cosine_half = cosine[..., :half]
    sine_half = sine[..., :half]
    out = np.empty_like(value, dtype=np.float32)
    out[..., :half] = first * cosine_half - second * sine_half
    out[..., half:] = second * cosine_half + first * sine_half
    return out


def graph_softmax(score, mask):
    """Softmax exactly as the compiled graph computes it.

    The IR is  max -> sub -> exp -> reduce_sum -> log -> add(max) -> sub -> exp,
    i.e. exp(x - logsumexp(x)). That is a different rounding sequence from the
    textbook exp(x-max)/sum, so the reference must use the graph's form or the
    comparison would report a difference the hardware does not have.
    """
    masked = score + mask
    maximum = np.max(masked, axis=-1, keepdims=True)
    shifted = masked - maximum
    exponentials = np.exp(shifted, dtype=np.float32)
    total = np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)
    log_sum_exp = maximum + np.log(total, dtype=np.float32)
    return np.exp(masked - log_sum_exp, dtype=np.float32)


class QuantizedWeights:
    """Per-output-channel int8 view of every matrix the graph reads.

    ``quantize=False`` keeps the same forward code but skips the int8 step, so a
    mismatch against the FP32 reference can be attributed: if the exact-weight
    run does not match, the model implementation is wrong and no conclusion about
    quantization error is valid yet.
    """

    def __init__(self, checkpoint, layout, config, quantize=True):
        from safetensors.torch import load_file
        import torch
        state = load_file(str(Path(checkpoint) / "model.safetensors"))
        self.config = config
        self.quantize = quantize
        self.fp32 = {}
        for name, tensor in state.items():
            self.fp32[name] = tensor.to(dtype=torch.float32).numpy()
        self.quantized = {}
        self.scheme = {
            "enabled": quantize,
            "weight": "per-output-channel symmetric int8, [-127,127], "
                      "scale = max|row|/127, zero rows -> scale 1",
            "activation": "per-token symmetric int8, dynamic, same rounding",
            "accumulator": "int32",
            "dequantize": "(acc_f32 * activation_scale) * weight_scale",
        }

    def matrix(self, name):
        if name not in self.quantized:
            self.quantized[name] = quantize_rows(self.fp32[name])
        return self.quantized[name]

    def linear(self, activation, name):
        if not self.quantize:
            return (activation.astype(np.float32)
                    @ self.fp32[name].astype(np.float32).T).astype(np.float32)
        quantized, scale = self.matrix(name)
        output = linear_w8a8(activation, quantized, scale)
        if getattr(self, "capture_intermediates", False):
            q, a_scale = quantize_rows(activation)
            prefix = self.trace_label + "_" + name
            self.trace[prefix + "_activation"] = activation.copy()
            self.trace[prefix + "_q"] = q
            self.trace[prefix + "_a_scale"] = a_scale
            self.trace[prefix + "_out"] = output.copy()
        return output

    def embed(self, token_ids):
        """Row lookup; from the quantized matrix when quantizing is enabled.

        ``--f32-embedding`` keeps the gather in FP32 while the linears stay W8A8.
        That is what the compiled graph currently does: the embedding op is not
        part of the W8A8 rewrite yet, so comparing against a reference that
        quantises the gather would mix a known difference into the result.
        """
        if not self.quantize or getattr(self, "f32_embedding", False):
            return self.fp32["model.embed_tokens.weight"][token_ids].astype(np.float32)
        quantized, scale = self.matrix("model.embed_tokens.weight")
        rows = quantized[token_ids].astype(np.float32)
        return rows * scale[token_ids][:, None]


def forward_layer(hidden, layer, weights, start_position, cache_key, cache_value,
                  config, eps, theta):
    """One decoder layer, in the order the imported graph performs it.

    ``start_position`` is the *scalar* first position of this call, matching the
    graph's ``tensor<1xi64>`` argument. The compiled prefill IR derives three
    things from it with the same expression ``start + arange(S)``:

      %4  = tosa.add arange(16), %arg12   -> rotary positions and KV slots
      %14 = reshape(%4) as [1,1,S,1]      -> per-query mask boundary

    so the absolute position vector must be built exactly once. Adding a
    per-token vector on top of that would double-count the offset and scatter
    the KV writes across the wrong slots.
    """
    prefix = f"model.layers.{layer}"
    profile = getattr(weights, "arithmetic_profile", "numpy-graph")
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    sequence = hidden.shape[0]
    # cache_key is one layer's slice: [capacity, kv_heads, head_dim].
    capacity = cache_key.shape[0]
    total = capacity
    absolute = start_position + np.arange(sequence, dtype=np.int64)
    def capture(name, value):
        if getattr(weights, "capture_intermediates", False):
            weights.trace[f"{weights.trace_label}_{prefix}_{name}"] = value.copy()
    capture("input_hidden", hidden)

    # 1. input RMSNorm, then Q/K/V projections.
    normed = rms_norm(hidden, weights.fp32[f"{prefix}.input_layernorm.weight"], eps, profile)
    capture("input_norm", normed)
    query = weights.linear(normed, f"{prefix}.self_attn.q_proj.weight")
    key = weights.linear(normed, f"{prefix}.self_attn.k_proj.weight")
    value = weights.linear(normed, f"{prefix}.self_attn.v_proj.weight")

    query = query.reshape(sequence, heads, head_dim)
    key = key.reshape(sequence, kv_heads, head_dim)
    value = value.reshape(sequence, kv_heads, head_dim)

    # 2. Q/K RMSNorm over head_dim, then RoPE at the absolute positions.
    query = rms_norm(query, weights.fp32[f"{prefix}.self_attn.q_norm.weight"], eps, profile)
    key = rms_norm(key, weights.fp32[f"{prefix}.self_attn.k_norm.weight"], eps, profile)
    capture("q_norm", query)
    capture("k_norm", key)
    cosine, sine = rope_tables(absolute, inv_freq(theta, head_dim), profile)
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    query = apply_rope(query, cosine, sine)
    key = apply_rope(key, cosine, sine)
    capture("q_rope", query)
    capture("k_rope", key)

    # 3. KV cache write into the absolute slots.
    cache_key[absolute] = key
    cache_value[absolute] = value

    # 4. GQA head expansion (kv_head = q_head // 2) and per-head attention.
    expanded_key = np.repeat(cache_key, heads // kv_heads, axis=1)
    expanded_value = np.repeat(cache_value, heads // kv_heads, axis=1)

    # score[h, s, t] = sum_d q[s,h,d] * k[t,h,d]
    if profile in ("triton-host", "nr-fpga"):
        from reference_fp32 import blocked_dot
        score = blocked_dot(query.transpose(1, 0, 2), expanded_key.transpose(1, 2, 0), profile=profile)
    else:
        score = np.einsum("shd,thd->hst", query.astype(np.float32),
                          expanded_key.astype(np.float32), optimize=True)
    score = score * np.float32(0.0883883461)  # the graph's rounded 1/sqrt(128)
    # mask[s, t] = 0 while slot t is at or before the query's own position.
    # The graph builds it as greater_equal(absolute[1,1,S,1], slot[1,1,1,T])
    # and broadcasts it across heads.
    boundary = absolute[:, None]
    slots_axis = np.arange(total)[None, :]
    mask = np.where(slots_axis <= boundary, np.float32(0), -np.inf).astype(np.float32)
    if profile in ("triton-host", "nr-fpga"):
        from reference_fp32 import blocked_dot, softmax
        probabilities = softmax(score, mask[None, :, :], profile)
        context = blocked_dot(probabilities, expanded_value.transpose(1, 0, 2), profile=profile).transpose(1, 0, 2)
    else:
        probabilities = graph_softmax(score, mask[None, :, :])
        context = np.einsum("hst,thd->shd", probabilities.astype(np.float32),
                            expanded_value.astype(np.float32), optimize=True)
    context = context.reshape(sequence, heads * head_dim)
    capture("attention_context", context)
    capture("attention_probabilities", probabilities)

    # 5. Output projection and residual.
    projected = weights.linear(context, f"{prefix}.self_attn.o_proj.weight")
    hidden = hidden + projected
    capture("attention_residual", hidden)

    # 6. MLP: second RMSNorm, gate/up, SiLU(gate)*up, down, residual.
    normed = rms_norm(hidden, weights.fp32[f"{prefix}.post_attention_layernorm.weight"], eps, profile)
    capture("post_attention_norm", normed)
    gate = weights.linear(normed, f"{prefix}.mlp.gate_proj.weight")
    up = weights.linear(normed, f"{prefix}.mlp.up_proj.weight")
    # x * sigmoid(x), written branch-free and without exp overflow for large
    # negative gate values: exp(-|x|) is always in [0, 1].
    if profile in ("triton-host", "nr-fpga"):
        from reference_fp32 import silu
        activated = silu(gate, profile)
    else:
        magnitude = np.exp(-np.abs(gate), dtype=np.float32)
        sigmoid = np.where(gate >= 0, np.float32(1) / (np.float32(1) + magnitude),
                           magnitude / (np.float32(1) + magnitude)).astype(np.float32)
        activated = (gate * sigmoid).astype(np.float32)
    capture("silu", activated)
    capture("swiglu", activated * up)
    hidden = hidden + weights.linear(activated * up, f"{prefix}.mlp.down_proj.weight")
    capture("output_hidden", hidden)
    return hidden


def forward(prompt_ids, weights, config, decode_steps, capacity, top_k,
            capture_layers=()):
    if not prompt_ids or decode_steps < 0 or len(prompt_ids) + decode_steps > capacity:
        raise ValueError("prompt plus decode must fit cache capacity")
    eps = config["rms_norm_eps"]
    theta = config["rope_theta"]
    layers = config["num_hidden_layers"]
    head_dim = config["head_dim"]
    kv_heads = config["num_key_value_heads"]
    sequence = len(prompt_ids)

    cache_key = np.zeros((layers, capacity, kv_heads, head_dim), dtype=np.float32)
    cache_value = np.zeros_like(cache_key)
    report = {"capacity": capacity, "layers": layers,
              "arithmetic_profile": getattr(weights, "arithmetic_profile", "numpy-graph"),
              "kv_layout": "layer,position,kv_head,head_dim",
              "quantization": weights.scheme}
    profile = report["arithmetic_profile"]
    report["floating_arithmetic"] = {
        "numpy-graph": "NumPy reductions/BLAS, logsumexp softmax and stable sigmoid; not the exact deployed kernel rounding sequence",
        "triton-host": "ordered FP32 reductions; BK64 dot partials with separate multiply/add; platform libm; exp/sum softmax; direct SiLU divide",
        "nr-fpga": "ordered FP32 reductions; BK64 dot partials with fmaf (observed NR vfmacc.vf); shared NR scalar math; exp/sum softmax; direct SiLU divide",
    }[profile]
    report["reference_independence"] = (
        "graph ordering, int8 quantization/accumulation and attention dot implemented independently; public NR scalar math reused, not independently validated here"
        if profile == "nr-fpga" else "No compiled Triton kernels or Buddy graph loaded")
    arrays = {}
    weights.trace = {}
    captured = {}

    def run(ids, start_position):
        weights.trace_label = "prefill" if start_position == 0 else f"decode_{start_position - sequence}"
        hidden = weights.embed(np.asarray(ids, dtype=np.int64))
        for layer in range(layers):
            hidden = forward_layer(hidden, layer, weights, start_position,
                                   cache_key[layer], cache_value[layer],
                                   config, eps, theta)
            if layer in capture_layers:
                captured.setdefault(layer, []).append(hidden.copy())
        hidden = rms_norm(hidden, weights.fp32["model.norm.weight"], eps,
                          getattr(weights, "arithmetic_profile", "numpy-graph"))
        logits = weights.linear(hidden, "model.embed_tokens.weight")
        return logits

    # Prefill: start position 0, one prediction per position.
    prefill_logits = run(prompt_ids, 0)
    arrays["prefill_logits"] = prefill_logits.astype(np.float32)
    report["prefill_argmax_all"] = [int(v) for v in np.argmax(prefill_logits, axis=-1)]
    report["prefill_argmax_last"] = int(np.argmax(prefill_logits[-1]))

    # Decode: one token at a time, the cache carried forward.
    steps = []
    next_token = report["prefill_argmax_last"]
    for step in range(decode_steps):
        position = sequence + step
        logits = run(np.array([next_token]), position)[-1]
        arrays[f"decode_logits_{step}"] = logits.astype(np.float32)
        arrays[f"decode_kv_key_{step}"] = cache_key[:, :position + 1].copy()
        arrays[f"decode_kv_value_{step}"] = cache_value[:, :position + 1].copy()
        chosen = int(np.argmax(logits))
        order = np.argsort(-logits)
        steps.append({"step": step, "cache_position": position,
                      "input_token": next_token, "generated_token": chosen,
                      "top_k": [[int(i), float(logits[i])] for i in order[:top_k]]})
        next_token = chosen
    report["decode_steps_recorded"] = steps
    report["generated_ids"] = [s["generated_token"] for s in steps]
    arrays["kv_key_used"] = cache_key[:, :sequence + decode_steps].copy()
    arrays["kv_value_used"] = cache_value[:, :sequence + decode_steps].copy()
    for layer, values in captured.items():
        # Prefill has S rows; decode has one. They cannot be stacked together.
        arrays[f"prefill_layer_{layer}_hidden"] = values[0]
        if len(values) > 1:
            arrays[f"decode_layer_{layer}_hidden"] = np.stack(values[1:])
    arrays.update(weights.trace)
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--capture-layers", default="")
    parser.add_argument("--capture-intermediates", action="store_true")
    parser.add_argument("--arithmetic-profile", choices=["numpy-graph", "triton-host", "nr-fpga"],
                        default="numpy-graph", help="floating arithmetic contract: scalar host kernels, or fused RVV attention plus NR runtime math")
    parser.add_argument("--f32-embedding", action="store_true",
                        help="keep the embedding gather in FP32 while the linears "
                             "stay W8A8 (matches the current compiled graph)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="exact-weight control run: same code path without int8")
    parser.add_argument("--fp32-reference", type=Path, default=None,
                        help="directory with the FP32 reference arrays.npz")
    args = parser.parse_args()
    source_paths = [Path(__file__), Path(__file__).with_name("quant_reference.py"),
                    Path(__file__).with_name("reference_fp32.py"),
                    Path(__file__).with_name("reference_fp32.c"),
                    args.assets / "config.json"]
    if args.arithmetic_profile == "nr-fpga":
        source_paths.append(Path(__file__).resolve().parents[3] / "common/nr/nr_math.c")
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths}

    from transformers import AutoConfig
    from host_reference import error_stats

    config = AutoConfig.from_pretrained(args.assets)
    if args.layers is not None:
        config.num_hidden_layers = args.layers
    parameters = getattr(config, "rope_parameters", None) or {}
    config_dict = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0)),
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    prompt_ids = [int(v) for v in args.prompt_ids.split(",") if v.strip()]
    capture = {int(v) for v in args.capture_layers.split(",") if v.strip()}

    weights = QuantizedWeights(args.checkpoint, args.layout, config_dict,
                               quantize=not args.no_quantize)
    weights.f32_embedding = args.f32_embedding
    weights.capture_intermediates = args.capture_intermediates
    weights.arithmetic_profile = args.arithmetic_profile
    report, arrays = forward(prompt_ids, weights, config_dict, args.decode_steps,
                             args.max_cache_len, args.top_k, capture)
    report["config"] = config_dict
    report["prompt_ids"] = prompt_ids
    report["invocation"] = [sys.executable, *sys.argv]
    report["source_sha256"] = provenance
    report["checkpoint"] = str(args.checkpoint / "model.safetensors")
    report["layout_usage"] = "independent named-checkpoint reference; --layout retained for CLI compatibility, not used for graph ordering"

    # Weight quantization error, measured on the source matrices themselves.
    matrix_error = {}
    for name, tensor in weights.fp32.items():
        if tensor.ndim != 2:
            continue
        quantized, scale = weights.matrix(name)
        restored = quantized.astype(np.float32) * scale[:, None]
        diff = np.abs(tensor - restored)
        matrix_error[name] = {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "max_abs_weight": float(np.abs(tensor).max()),
        }
    worst = sorted(matrix_error.items(),
                   key=lambda kv: -kv[1]["max_abs_error"])[:5]
    report["weight_quantization_error_worst5"] = dict(worst)

    if args.fp32_reference:
        reference = np.load(args.fp32_reference / "arrays.npz")
        reference_json = json.loads(
            (args.fp32_reference / "reference.json").read_text())
        comparisons = {"prefill_logits": error_stats(
            reference["prefill_logits"], arrays["prefill_logits"])}
        for step in range(args.decode_steps):
            comparisons[f"decode_logits_{step}"] = error_stats(
                reference[f"decode_logits_{step}"], arrays[f"decode_logits_{step}"])
        report["quantized_vs_fp32"] = comparisons
        # The prefill argmax is the *input* to decode step 0. If it differs, the
        # two runs are decoding from different tokens and a later "trajectory
        # match" is a coincidence, not agreement -- so it is compared explicitly.
        report["prefill_argmax_vs_fp32"] = {
            "quantized": report["prefill_argmax_last"],
            "fp32": int(np.argmax(reference["prefill_logits"][0, -1])),
            "match": report["prefill_argmax_last"]
            == int(np.argmax(reference["prefill_logits"][0, -1])),
        }
        report["token_trajectory_vs_fp32"] = {
            "quantized": report["generated_ids"],
            "fp32": reference_json["generated_ids"],
            "match": (report["prefill_argmax_vs_fp32"]["match"]
                      and report["generated_ids"] == reference_json["generated_ids"]),
        }
        context_equal = report["prefill_argmax_vs_fp32"]["match"]
        contexts = []
        for actual, expected in zip(report["decode_steps_recorded"], reference_json["decode_steps_recorded"]):
            context_equal = context_equal and actual["input_token"] == expected["input_token"]
            contexts.append(context_equal)
        report["decode_same_input_context_vs_fp32"] = contexts
        report["quantized_vs_fp32_interpretation"] = "Deployment precision error; after differing tokens, includes free-running trajectory divergence"

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "quant-reference.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("decode_steps_recorded", "quantization")},
                     indent=2)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
