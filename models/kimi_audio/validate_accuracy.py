#!/usr/bin/env python3
# ===- validate_accuracy.py -------------------------------------------------
#
# Kimi-Audio-7B-Instruct Accuracy Validation Script
#
# Compares HuggingFace model outputs against Buddy-MLIR compiled outputs.
# Kimi-Audio is an audio foundation model with dual text+audio output heads.
# For text-only validation, we compare the text logits from the lm_head.
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def generate_reference(output_dir: str, max_seq_len: int = 128):
    """Generate HuggingFace reference outputs for accuracy validation."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = "moonshotai/Kimi-Audio-7B-Instruct"
    print(f"[KimiAudio-Validate] Loading HF model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    ).eval()
    model.config.use_cache = False

    # Text-only forward pass (no audio features)
    test_prompts = [
        "Hello, how are you?",
        "What is the weather like today?",
        "Tell me a short story.",
    ]

    references = {}

    for i, prompt in enumerate(test_prompts):
        print(f"\n[KimiAudio-Validate] Processing prompt {i}: {prompt!r}")

        # Simple tokenization with dummy IDs
        input_ids = torch.zeros((1, max_seq_len), dtype=torch.int64)
        for j, c in enumerate(prompt[:max_seq_len]):
            input_ids[0, j] = ord(c) % 1000 + 100  # Dummy token IDs

        text_input_ids = torch.zeros((1, max_seq_len), dtype=torch.int64)
        whisper_feat = torch.zeros((1, 1, 5120), dtype=torch.float32)
        is_cont_mask = torch.zeros((1, max_seq_len), dtype=torch.int64)
        attn_mask = torch.ones((1, max_seq_len), dtype=torch.int64)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=whisper_feat,
                is_continuous_mask=is_cont_mask,
                attention_mask=attn_mask,
                use_cache=False,
            )

        # outputs is a tuple (audio_logits, text_logits)
        if isinstance(outputs, tuple):
            text_logits = outputs[1].detach().numpy()
        else:
            text_logits = outputs.logits
            if isinstance(text_logits, tuple):
                text_logits = text_logits[1].detach().numpy()

        last_logits = text_logits[0, -1, :].astype(np.float32)

        ref = {
            "prompt": prompt,
            "last_logits_sum": float(np.sum(last_logits)),
            "last_logits_mean": float(np.mean(last_logits)),
            "last_logits_std": float(np.std(last_logits)),
            "top5_indices": np.argsort(last_logits)[-5:][::-1].tolist(),
            "top5_values": last_logits[np.argsort(last_logits)[-5:][::-1]].tolist(),
        }
        references[f"prompt_{i}"] = ref

        np.save(os.path.join(output_dir, f"ref_logits_{i}.npy"), last_logits)
        print(f"   logits shape: {text_logits.shape}, top-5: {ref['top5_indices']}")

    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)

    print(f"\n[KimiAudio-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-3):
    """Compare Buddy-MLIR compiled model outputs against HF reference."""
    manifest_path = os.path.join(reference_dir, "reference_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[KimiAudio-Validate] ERROR: reference manifest not found")
        return False

    with open(manifest_path) as f:
        references = json.load(f)

    all_passed = True

    for key, ref in references.items():
        buddy_path = os.path.join(buddy_output_dir, f"buddy_logits_{key.split('_')[1]}.npy")
        if not os.path.exists(buddy_path):
            print(f"[KimiAudio-Validate] SKIP: buddy logits not found: {buddy_path}")
            continue

        buddy_logits = np.load(buddy_path)
        ref_logits = np.array(ref["last_logits"], dtype=np.float32)

        max_abs_error = float(np.max(np.abs(ref_logits - buddy_logits)))
        cos_sim = float(np.dot(ref_logits, buddy_logits) / (
            np.linalg.norm(ref_logits) * np.linalg.norm(buddy_logits) + 1e-10
        ))
        ref_top5 = set(np.argsort(ref_logits)[-5:])
        buddy_top5 = set(np.argsort(buddy_logits)[-5:])
        top5_match = len(ref_top5 & buddy_top5)

        passed = max_abs_error < tolerance and cos_sim > 0.99
        status = "PASS" if passed else "FAIL"
        print(f"  {status} | {ref['prompt'][:40]:40s} | max_err={max_abs_error:.2e} | cos={cos_sim:.6f} | top5={top5_match}/5")

        if not passed:
            all_passed = False

    print(f"\n[KimiAudio-Validate] {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Kimi-Audio-7B-Instruct Accuracy Validation")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["reference", "compare"],
                       help="'reference' to generate HF outputs, 'compare' to validate buddy outputs")
    parser.add_argument("--output-dir", type=str, default="./validation_data")
    parser.add_argument("--reference-dir", type=str, default="./validation_data")
    parser.add_argument("--buddy-output-dir", type=str, default="./build")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "reference":
        generate_reference(args.output_dir, args.max_seq_len)
    elif args.mode == "compare":
        success = compare_outputs(args.reference_dir, args.buddy_output_dir, args.tolerance)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
