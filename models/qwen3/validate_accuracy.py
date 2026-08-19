#!/usr/bin/env python3
# ===- validate_accuracy.py -------------------------------------------------
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
# Qwen3-0.6B Accuracy Validation Script
#
# Compares HuggingFace model outputs against Buddy-MLIR compiled outputs.
# When run without buddy-mlir artifacts, generates reference outputs for
# later comparison.
#
# Usage:
#   # Generate HF reference outputs only:
#   python validate_accuracy.py --mode reference --output-dir ./validation_data
#
#   # Compare buddy outputs against reference:
#   python validate_accuracy.py --mode compare --reference-dir ./validation_data \
#       --buddy-output-dir ./build
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

    model_path = os.environ.get("QWEN3_MODEL_PATH", "Qwen/Qwen3-0.6B")
    print(f"[Qwen3-Validate] Loading HF model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test prompts covering different scenarios
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2+2? The answer is",
        "Once upon a time, in a land far away,",
    ]

    references = {}

    for i, prompt in enumerate(test_prompts):
        print(f"\n[Qwen3-Validate] Processing prompt {i}: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=max_seq_len, padding=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        logits = outputs.logits.detach().numpy()
        # Save last-token logits (used for next-token prediction)
        last_logits = logits[0, -1, :].astype(np.float32)

        ref = {
            "prompt": prompt,
            "input_ids": input_ids.squeeze(0).tolist(),
            "last_logits": last_logits.tolist(),
            "logits_sum": float(np.sum(last_logits)),
            "logits_mean": float(np.mean(last_logits)),
            "logits_std": float(np.std(last_logits)),
            "top5_indices": np.argsort(last_logits)[-5:][::-1].tolist(),
            "top5_values": last_logits[np.argsort(last_logits)[-5:][::-1]].tolist(),
        }
        references[f"prompt_{i}"] = ref

        # Also save full logits for detailed comparison
        np.save(os.path.join(output_dir, f"ref_logits_{i}.npy"), last_logits)
        np.save(os.path.join(output_dir, f"ref_input_ids_{i}.npy"),
                input_ids.squeeze(0).numpy())

        print(f"   input shape: {input_ids.shape}")
        print(f"   logits shape: {logits.shape}")
        print(f"   top-5 tokens: {ref['top5_indices']}")

    # Save reference metadata
    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)

    print(f"\n[Qwen3-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-3):
    """Compare Buddy-MLIR compiled model outputs against HF reference."""
    # Load reference manifest
    with open(os.path.join(reference_dir, "reference_manifest.json")) as f:
        references = json.load(f)

    results = []
    all_passed = True

    for key, ref in references.items():
        print(f"\n[Qwen3-Validate] Checking {key}: {ref['prompt']!r}")

        # Load buddy output logits
        buddy_logits_path = os.path.join(buddy_output_dir, f"buddy_logits_{key.split('_')[1]}.npy")
        if not os.path.exists(buddy_logits_path):
            print(f"   SKIP: buddy logits not found at {buddy_logits_path}")
            continue

        buddy_logits = np.load(buddy_logits_path)
        ref_logits = np.array(ref["last_logits"], dtype=np.float32)

        # Compute metrics
        abs_diff = np.abs(ref_logits - buddy_logits)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        rel_error = np.max(abs_diff / (np.abs(ref_logits) + 1e-10))

        # Cosine similarity
        cos_sim = np.dot(ref_logits, buddy_logits) / (
            np.linalg.norm(ref_logits) * np.linalg.norm(buddy_logits) + 1e-10
        )

        # Top-5 match
        ref_top5 = set(np.argsort(ref_logits)[-5:])
        buddy_top5 = set(np.argsort(buddy_logits)[-5:])
        top5_match = len(ref_top5 & buddy_top5)

        passed = max_abs_error < tolerance and cos_sim > 0.99

        result = {
            "prompt": ref["prompt"],
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "max_rel_error": float(rel_error),
            "cosine_similarity": float(cos_sim),
            "top5_match_count": top5_match,
            "passed": passed,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"   {status}: max_abs_err={max_abs_error:.6e}, "
              f"cos_sim={cos_sim:.8f}, top5_match={top5_match}/5")

        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print(f"[Qwen3-Validate] Summary: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status} | {r['prompt'][:50]:50s} | "
              f"max_err={r['max_abs_error']:.2e} | cos={r['cosine_similarity']:.6f}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Accuracy Validation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["reference", "compare"],
        help="'reference' to generate HF reference outputs, 'compare' to validate buddy outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_data",
        help="Directory for reference output data",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="./validation_data",
        help="Directory containing reference outputs (compare mode)",
    )
    parser.add_argument(
        "--buddy-output-dir",
        type=str,
        default="./build",
        help="Directory containing buddy-mlir compiled outputs (compare mode)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for test prompts",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum absolute error tolerance for validation",
    )
    args = parser.parse_args()

    if args.mode == "reference":
        generate_reference(args.output_dir, args.max_seq_len)
    elif args.mode == "compare":
        success = compare_outputs(
            args.reference_dir, args.buddy_output_dir, args.tolerance
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
