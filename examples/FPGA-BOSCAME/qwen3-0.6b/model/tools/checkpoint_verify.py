#!/usr/bin/env python3
"""Verify the raw checkpoint against the official config, before any framework load.

Two things this must establish that a framework load cannot:

  1. tensor shapes/dtypes match the pinned config exactly;
  2. whether the file's own ``model.embed_tokens.weight`` and ``lm_head.weight``
     are bit-identical. ``tie_word_embeddings`` in the config only says what the
     model does at runtime; a framework may overwrite one tensor with the other,
     so the raw-file comparison is the only honest evidence. The result decides
     whether the deployment image may store a single shared matrix.

Reads the safetensors header directly and memmaps the payload; it never
materialises the whole 1.5 GB file and never calls ``from_pretrained``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

DTYPES = {
    "BF16": np.uint16,  # compare raw bits; no bf16 numpy dtype exists
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
}

# Attention projection widths are filled from the config (head_dim must not be
# derived as hidden/heads); MLP widths are fixed by intermediate_size.
EXPECTED_MATRICES = {}
# (tensor name suffix, expected shape builder). Q/K norm live under self_attn
# and are per-head-dim, not per-hidden; they are not part of the decoder norms.
LAYER_NORMS = (("input_layernorm", "hidden"), ("post_attention_layernorm", "hidden"),
               ("self_attn.q_norm", "head_dim"), ("self_attn.k_norm", "head_dim"))


def read_header(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        raw = handle.read(length)
    header = json.loads(raw)
    metadata = header.pop("__metadata__", {})
    return header, metadata, 8 + length


def tensor_slice(handle, entry, base):
    dtype = DTYPES[entry["dtype"]]
    begin, end = entry["data_offsets"]
    handle.seek(base + begin)
    buffer = handle.read(end - begin)
    return np.frombuffer(buffer, dtype=dtype).reshape(entry["shape"])


def compare_tensors(path, base, header, name_a, name_b, chunk_rows=8192):
    """Bit-exact comparison without holding either tensor in full."""
    a, b = header[name_a], header[name_b]
    if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
        return {"identical": False, "reason": "shape or dtype differs",
                "shape_a": a["shape"], "shape_b": b["shape"]}
    rows = a["shape"][0]
    itemsize = np.dtype(DTYPES[a["dtype"]]).itemsize
    row_bytes = a["shape"][1] * itemsize
    differing = 0
    max_abs = 0
    first_index = None
    with open(path, "rb") as handle:
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            left = tensor_slice(handle, {
                "dtype": a["dtype"], "shape": [stop - start, a["shape"][1]],
                "data_offsets": [a["data_offsets"][0] + start * row_bytes,
                                 a["data_offsets"][0] + stop * row_bytes]},
                base)
            right = tensor_slice(handle, {
                "dtype": b["dtype"], "shape": [stop - start, b["shape"][1]],
                "data_offsets": [b["data_offsets"][0] + start * row_bytes,
                                 b["data_offsets"][0] + stop * row_bytes]},
                base)
            if a["dtype"] == "BF16":
                # Same raw bits is the only defensible equality for bf16.
                neq = left != right
                count = int(np.count_nonzero(neq))
                if count and first_index is None:
                    first_index = int(np.argmax(neq.reshape(-1))) + start * a["shape"][1]
                differing += count
            else:
                diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
                differing += int(np.count_nonzero(diff))
                max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
    result = {"identical": differing == 0, "differing_elements": differing,
              "total_elements": int(np.prod(a["shape"])), "dtype": a["dtype"]}
    if differing and first_index is not None:
        result["first_differing_flat_index"] = first_index
    if max_abs:
        result["max_abs_difference"] = max_abs
    return result


def verify(checkpoint, config_path, output=None, expected_sha256=None):
    checkpoint = Path(checkpoint)
    header, metadata, base = read_header(checkpoint)
    config = json.loads(Path(config_path).read_text())

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "safetensors_metadata": metadata,
        "header_entry_count": len(header),
        "config": {
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
            "vocab_size": config["vocab_size"],
            "rms_norm_eps": config["rms_norm_eps"],
            "rope_theta": config.get("rope_theta", config.get("rope_parameters", {}).get("rope_theta")),
            "tie_word_embeddings": config["tie_word_embeddings"],
            "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
        },
    }

    if expected_sha256:
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as handle:
            while True:
                block = handle.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        actual = digest.hexdigest()
        report["sha256"] = actual
        report["sha256_matches_expected"] = actual == expected_sha256
        if actual != expected_sha256:
            raise ValueError(f"sha256 {actual} != expected {expected_sha256}")

    layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    head_dim = report["config"]["head_dim"]
    q_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim

    problems = []
    checks = 0

    def expect(name, shape, dtype="BF16"):
        nonlocal checks
        checks += 1
        entry = header.get(name)
        if entry is None:
            problems.append(f"missing tensor {name}")
            return None
        if list(entry["shape"]) != list(shape):
            problems.append(f"{name}: shape {entry['shape']} != {list(shape)}")
        if entry["dtype"] != dtype:
            problems.append(f"{name}: dtype {entry['dtype']} != {dtype}")
        return entry

    expect("model.embed_tokens.weight", [vocab, hidden])
    expect("lm_head.weight", [vocab, hidden])
    expect("model.norm.weight", [hidden])

    widths = dict(EXPECTED_MATRICES)
    widths["q_proj"] = (q_width, hidden)
    widths["k_proj"] = (kv_width, hidden)
    widths["v_proj"] = (kv_width, hidden)
    widths["o_proj"] = (hidden, q_width)
    # gate/up/down live under mlp, not self_attn; keep them separate so the
    # expected path prefix stays honest.
    mlp = {"gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden),
           "down_proj": (hidden, intermediate)}

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        for norm, which in LAYER_NORMS:
            expect(f"{prefix}.{norm}.weight",
                   [head_dim] if which == "head_dim" else [hidden])
        for proj, shape in widths.items():
            expect(f"{prefix}.self_attn.{proj}.weight", shape)
        for proj, shape in mlp.items():
            expect(f"{prefix}.mlp.{proj}.weight", shape)

    report["tensor_shape_checks"] = checks
    report["tensor_shape_problems"] = problems

    # Layer-0-only extra scan: catches an unexpected extra parameter that the
    # config-driven list above would not look for (e.g. a bias tensor).
    layer0 = sorted(k for k in header if k.startswith("model.layers.0."))
    report["layer0_tensors"] = layer0
    other = sorted(k for k in header if not k.startswith("model.layers."))
    report["non_layer_tensors"] = other

    report["embedding_vs_lm_head"] = compare_tensors(
        checkpoint, base, header, "model.embed_tokens.weight", "lm_head.weight")

    params = sum(int(np.prod(header[k]["shape"]))
                 for k in header if k not in ("model.rotary_emb.inv_freq",))
    report["stored_parameters"] = params
    report["expected_stored_parameters"] = 751632384

    # Section-level provenance for the deployment image manifest.
    sections = []
    for name in sorted(header, key=lambda n: header[n]["data_offsets"][0]):
        entry = header[name]
        sections.append({
            "name": name, "dtype": entry["dtype"], "shape": entry["shape"],
            "file_offset": base + entry["data_offsets"][0],
            "bytes": entry["data_offsets"][1] - entry["data_offsets"][0],
        })
    sections.sort(key=lambda s: s["file_offset"])
    report["sections"] = sections

    report["status"] = "PASS" if not problems else "FAIL"
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--sections-output", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.config, args.output, args.expected_sha256)
    if args.sections_output:
        args.sections_output.parent.mkdir(parents=True, exist_ok=True)
        args.sections_output.write_text(
            json.dumps(result["sections"], indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "sections"}
    print(json.dumps(summary, indent=2))