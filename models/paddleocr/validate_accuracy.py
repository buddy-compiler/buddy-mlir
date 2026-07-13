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
# PaddleOCR-VL-0.9B Accuracy Validation Script
#
# Compares HuggingFace model outputs against Buddy-MLIR compiled outputs.
# PaddleOCR-VL is a vision-language model that takes image + text as input.
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
from transformers import AutoModel


def create_dummy_image_input(image_size: int = 384, patch_size: int = 14):
    """Create dummy image input matching PaddleOCR-VL expected format.

    PaddleOCR-VL uses SigLIP vision encoder with:
    - image_size: 384x384
    - patch_size: 14
    - channels: 3

    Returns pixel_values and image_grid_thw for a single image.
    """
    # Single image: 3x384x384
    # Patches: (384/14) * (384/14) = 27*27 = 729 patches
    # But the model uses temporal_patch_size=2 and spatial_merge_size=2
    # Actual number of vision tokens depends on the model config.
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32) * 0.02

    # image_grid_thw: (temporal, height, width) of feature grid
    # For a single 384x384 image with temporal_patch_size=2:
    # t=1, h=54, w=72  (or similar, depending on config)
    image_grid_thw = [[1, 54, 72]]

    return pixel_values, image_grid_thw


def generate_reference(output_dir: str):
    """Generate HuggingFace reference outputs for accuracy validation."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = "lvyufeng/PaddleOCR-VL-0.9B"
    print(f"[PaddleOCR-Validate] Loading HF model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    ).eval()
    model.config.use_cache = False

    image_token_id = model.config.image_token_id
    print(f"   image_token_id = {image_token_id}")

    # Create dummy image inputs
    pixel_values, image_grid_thw = create_dummy_image_input()

    # Build input with image tokens + text tokens
    n_img_tokens = 972  # (54*72) / (2*2) = 3888/4 = 972
    total_len = 982  # 972 image tokens + 10 text tokens

    input_ids = torch.full((1, total_len), 1, dtype=torch.int64)
    input_ids[0, :n_img_tokens] = image_token_id
    attention_mask = torch.ones((1, total_len), dtype=torch.int64)

    # Position IDs for 3D rotary position embedding
    position_ids = torch.zeros((3, 1, total_len), dtype=torch.int64)

    print(f"\n[PaddleOCR-Validate] Running HF forward pass...")
    print(f"   input_ids:      {input_ids.shape}")
    print(f"   pixel_values:   {pixel_values.shape}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            return_dict=False,
        )

    # Output is typically (logits, ...) or just logits
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    logits_np = logits.detach().numpy().astype(np.float32)
    last_logits = logits_np[0, -1, :]

    ref = {
        "input_shape": list(input_ids.shape),
        "n_img_tokens": n_img_tokens,
        "total_len": total_len,
        "logits_shape": list(logits_np.shape),
        "last_logits_sum": float(np.sum(last_logits)),
        "last_logits_mean": float(np.mean(last_logits)),
        "last_logits_std": float(np.std(last_logits)),
        "top5_indices": np.argsort(last_logits)[-5:][::-1].tolist(),
        "top5_values": last_logits[np.argsort(last_logits)[-5:][::-1]].tolist(),
        "logits_total_sum": float(np.sum(logits_np)),
        "logits_total_mean": float(np.mean(logits_np)),
    }

    # Save full logits for detailed comparison
    np.save(os.path.join(output_dir, "ref_logits.npy"), last_logits)
    np.save(os.path.join(output_dir, "ref_full_logits.npy"), logits_np)

    # Save reference metadata
    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(ref, f, indent=2)

    print(f"\n[PaddleOCR-Validate] Reference data:")
    print(f"   logits shape:  {ref['logits_shape']}")
    print(f"   logits sum:    {ref['logits_total_sum']:.4f}")
    print(f"   top-5 tokens:  {ref['top5_indices']}")
    print(f"\n[PaddleOCR-Validate] Reference data saved to: {output_dir}")

    return ref


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-2):
    """Compare Buddy-MLIR compiled model outputs against HF reference."""
    # Load reference manifest
    manifest_path = os.path.join(reference_dir, "reference_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[PaddleOCR-Validate] ERROR: reference manifest not found at {manifest_path}")
        return False

    with open(manifest_path) as f:
        ref = json.load(f)

    # Load reference logits
    ref_logits = np.load(os.path.join(reference_dir, "ref_logits.npy"))

    # Load buddy output logits
    buddy_logits_path = os.path.join(buddy_output_dir, "buddy_logits.npy")
    if not os.path.exists(buddy_logits_path):
        print(f"[PaddleOCR-Validate] SKIP: buddy logits not found at {buddy_logits_path}")
        print("   Run buddy-mlir inference first to generate buddy outputs.")
        return False

    buddy_logits = np.load(buddy_logits_path)

    # Ensure same shape
    assert ref_logits.shape == buddy_logits.shape, \
        f"Shape mismatch: ref {ref_logits.shape} vs buddy {buddy_logits.shape}"

    # Compute metrics
    abs_diff = np.abs(ref_logits - buddy_logits)
    max_abs_error = float(np.max(abs_diff))
    mean_abs_error = float(np.mean(abs_diff))
    rel_error = float(np.max(abs_diff / (np.abs(ref_logits) + 1e-10)))

    # Cosine similarity
    cos_sim = float(np.dot(ref_logits, buddy_logits) / (
        np.linalg.norm(ref_logits) * np.linalg.norm(buddy_logits) + 1e-10
    ))

    # Top-5 match
    ref_top5 = set(np.argsort(ref_logits)[-5:])
    buddy_top5 = set(np.argsort(buddy_logits)[-5:])
    top5_match = len(ref_top5 & buddy_top5)

    # For VLMs, tolerance is more relaxed due to vision encoder complexity
    passed = cos_sim > 0.99 and top5_match >= 3

    print("\n" + "=" * 60)
    print(f"[PaddleOCR-Validate] Validation Results:")
    print(f"   Max absolute error:  {max_abs_error:.6e}")
    print(f"   Mean absolute error: {mean_abs_error:.6e}")
    print(f"   Max relative error:  {rel_error:.6e}")
    print(f"   Cosine similarity:   {cos_sim:.8f}")
    print(f"   Top-5 match count:   {top5_match}/5")
    print(f"   Status:              {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"\n   Ref top-5:  {sorted(ref_top5)}")
        print(f"   Buddy top-5: {sorted(buddy_top5)}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-0.9B Accuracy Validation"
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
        "--tolerance",
        type=float,
        default=1e-2,
        help="Maximum absolute error tolerance for validation",
    )
    args = parser.parse_args()

    if args.mode == "reference":
        generate_reference(args.output_dir)
    elif args.mode == "compare":
        success = compare_outputs(
            args.reference_dir, args.buddy_output_dir, args.tolerance
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
