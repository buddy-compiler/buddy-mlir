#!/usr/bin/env python3
"""Compare a RAX last-position logits dump with the HF reference model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def load(path, dtype, shape):
    array = np.fromfile(path, dtype=dtype)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"Wrong tensor size: {path}")
    return torch.from_numpy(array.reshape(shape).copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rax-logits", type=Path)
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline-logits", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    seq = spec["max_seq_len"]
    patches = spec["num_image_patches"]
    patch = spec["patch_size"]
    vocab = spec["vocab_size"]
    f16 = spec.get("precision") == "f16"

    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16 if f16 else torch.float32,
        attn_implementation="eager",
    ).eval()
    inputs = {
        "input_ids": load(args.input_dir / "input_ids.i64", np.int64, (1, seq)),
        "pixel_values": load(
            args.input_dir
            / ("pixel_values.f16" if f16 else "pixel_values.f32"),
            np.float16 if f16 else np.float32,
            (patches, 3, patch, patch),
        ),
        "attention_mask": load(
            args.input_dir / "attention_mask.i64", np.int64, (1, seq)
        ),
        "position_ids": load(
            args.input_dir / "position_ids.i64", np.int64, (3, 1, seq)
        ),
        "image_grid_thw": torch.tensor([spec["image_grid_thw"]]),
    }
    with torch.inference_mode():
        expected = (
            model(**inputs, use_cache=False).logits[0, -1].float().numpy()
        )
    if args.reference_out:
        expected.tofile(args.reference_out)
    if not args.rax_logits:
        print(json.dumps({"reference_top1": int(expected.argmax())}))
        return
    actual = np.fromfile(args.rax_logits, dtype=np.float32)
    if actual.size != vocab:
        raise ValueError(
            f"RAX logits have {actual.size} elements; expected {vocab}"
        )
    error = actual - expected
    max_abs = float(np.max(np.abs(error)))
    relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(expected))
    centered_error = (actual - actual.mean()) - (expected - expected.mean())
    centered_rel_l2 = float(
        np.linalg.norm(centered_error)
        / np.linalg.norm(expected - expected.mean())
    )
    cosine_similarity = float(
        np.dot(actual, expected)
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    ref_topk = np.argsort(expected)[::-1][: args.top_k].tolist()
    rax_topk = np.argsort(actual)[::-1][: args.top_k].tolist()
    top1_equal = rax_topk[0] == ref_topk[0]
    topk_equal = rax_topk == ref_topk
    report = {
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "centered_relative_l2_error": centered_rel_l2,
        "cosine_similarity": cosine_similarity,
        "reference_top1": ref_topk[0],
        "rax_top1": rax_topk[0],
        "top1_equal": top1_equal,
        f"reference_top{args.top_k}": ref_topk,
        f"rax_top{args.top_k}": rax_topk,
        f"top{args.top_k}_equal": topk_equal,
    }
    if args.baseline_logits:
        baseline = np.fromfile(args.baseline_logits, dtype=np.float32)
        if baseline.size != vocab:
            raise ValueError(
                f"Baseline logits have {baseline.size} elements; expected {vocab}"
            )
        b_err = actual - baseline
        b_centered = (actual - actual.mean()) - (baseline - baseline.mean())
        report["baseline_max_abs_error"] = float(np.max(np.abs(b_err)))
        report["baseline_relative_l2_error"] = float(
            np.linalg.norm(b_err) / np.linalg.norm(baseline)
        )
        report["baseline_centered_relative_l2_error"] = float(
            np.linalg.norm(b_centered)
            / np.linalg.norm(baseline - baseline.mean())
        )
        report["baseline_cosine_similarity"] = float(
            np.dot(actual, baseline)
            / (np.linalg.norm(actual) * np.linalg.norm(baseline))
        )
    print(json.dumps(report, indent=2))
    # PyTorch CPU FP16 accumulates conv2d and attention batch_matmul in FP32,
    # whereas the compiled TOSA->Linalg FP16 graph accumulates them in FP16,
    # giving a ~3.3%-3.9% inherent baseline delta across x86 FP16, K3 RVV FP16,
    # and full 291-GEMM K3 FP16 IME (while rejecting partial 137-GEMM IME).
    max_abs_limit = (
        args.max_abs if args.max_abs is not None else (0.35 if f16 else 0.001)
    )
    relative_limit = (
        args.max_relative_l2
        if args.max_relative_l2 is not None
        else (0.045 if f16 else 0.0001)
    )
    if (
        not np.isfinite(actual).all()
        or not top1_equal
        or not topk_equal
        or max_abs > max_abs_limit
        or relative_l2 > relative_limit
    ):
        raise SystemExit("RAX logits failed validation")


if __name__ == "__main__":
    main()
