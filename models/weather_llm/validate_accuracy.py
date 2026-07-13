#!/usr/bin/env python3
# ===- validate_accuracy.py -------------------------------------------------
#
# Weather-LLM-SFT Accuracy Validation Script
#
# Compares HuggingFace LlamaForCausalLM outputs against Buddy-MLIR compiled
# outputs using cosine similarity of logits for weather-domain prompts.
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_reference(output_dir: str, max_seq_len: int = 128):
    """Generate HuggingFace reference outputs for accuracy validation."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.environ.get("WEATHER_LLM_MODEL_PATH", "AuraWorxAI/weather-llm-sft")
    print(f"[WeatherLLM-Validate] Loading HF model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Weather-domain test prompts
    test_prompts = [
        "The weather forecast for tomorrow indicates that",
        "Based on the atmospheric pressure readings,",
        "Precipitation levels are expected to",
        "The temperature trend over the next 48 hours shows",
    ]

    references = {}
    for i, prompt in enumerate(test_prompts):
        print(f"\n[WeatherLLM-Validate] Processing prompt {i}: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=max_seq_len, padding=False)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)

        logits = outputs.logits.detach().cpu().numpy()
        last_logits = logits[0, -1, :].astype(np.float32)

        ref = {
            "prompt": prompt,
            "input_ids": input_ids.squeeze(0).tolist(),
            "last_logits_sum": float(np.sum(last_logits)),
            "last_logits_mean": float(np.mean(last_logits)),
            "last_logits_std": float(np.std(last_logits)),
            "top5_indices": np.argsort(last_logits)[-5:][::-1].tolist(),
            "top5_values": last_logits[np.argsort(last_logits)[-5:][::-1]].tolist(),
        }
        references[f"prompt_{i}"] = ref
        np.save(os.path.join(output_dir, f"ref_logits_{i}.npy"), last_logits)
        np.save(os.path.join(output_dir, f"ref_input_ids_{i}.npy"), input_ids.squeeze(0).numpy())
        print(f"   input shape: {input_ids.shape}, top-5: {ref['top5_indices']}")

    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)
    print(f"\n[WeatherLLM-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-3):
    """Compare Buddy-MLIR outputs against HF reference."""
    with open(os.path.join(reference_dir, "reference_manifest.json")) as f:
        references = json.load(f)

    results = []
    all_passed = True
    for key, ref in references.items():
        idx = key.split("_")[1]
        buddy_path = os.path.join(buddy_output_dir, f"buddy_logits_{idx}.npy")
        if not os.path.exists(buddy_path):
            print(f"[WeatherLLM-Validate] SKIP: {buddy_path} not found")
            continue

        buddy_logits = np.load(buddy_path)
        ref_logits = np.array(ref["last_logits"], dtype=np.float32)

        max_abs_error = float(np.max(np.abs(ref_logits - buddy_logits)))
        cos_sim = float(np.dot(ref_logits, buddy_logits) /
                        (np.linalg.norm(ref_logits) * np.linalg.norm(buddy_logits) + 1e-10))
        ref_top5 = set(np.argsort(ref_logits)[-5:])
        buddy_top5 = set(np.argsort(buddy_logits)[-5:])
        top5_match = len(ref_top5 & buddy_top5)

        passed = max_abs_error < tolerance and cos_sim > 0.99
        results.append({
            "prompt": ref["prompt"],
            "max_abs_error": max_abs_error,
            "cosine_similarity": cos_sim,
            "top5_match_count": top5_match,
            "passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  {status} | {ref['prompt'][:50]:50s} | max_err={max_abs_error:.2e} | cos={cos_sim:.6f} | top5={top5_match}/5")
        if not passed:
            all_passed = False

    print(f"\n[WeatherLLM-Validate] {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Weather-LLM-SFT Accuracy Validation")
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
