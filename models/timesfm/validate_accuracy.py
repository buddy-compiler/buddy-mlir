#!/usr/bin/env python3
# ===- validate_accuracy.py -------------------------------------------------
#
# TimesFM 2.5 Accuracy Validation Script
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os
import sys
import numpy as np
import torch
import timesfm


def generate_reference(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[TimesFM-Validate] Loading TimesFM 2.5 model...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", backend="cpu"
    )
    model = tfm.model.cpu().eval()
    for p in model.parameters():
        p.data = p.data.cpu()

    test_inputs = [np.random.RandomState(i * 42).randn(1, 16, 32).astype(np.float32) for i in range(4)]
    references = {}
    for i, data in enumerate(test_inputs):
        dummy = torch.from_numpy(data)
        masks = torch.ones_like(dummy)
        with torch.no_grad():
            out, _ = model(inputs=dummy, masks=masks)
            _, _, point_forecast, _ = out
        fc = point_forecast.detach().cpu().numpy().astype(np.float32)
        ref = {
            "input_shape": list(dummy.shape),
            "forecast_shape": list(fc.shape),
            "forecast_mean": float(np.mean(fc)),
            "forecast_std": float(np.std(fc)),
            "forecast_sum": float(np.sum(fc)),
        }
        references[f"input_{i}"] = ref
        np.save(os.path.join(output_dir, f"ref_forecast_{i}.npy"), fc)
        print(f"   [{i}] input={dummy.shape}, forecast={fc.shape}, mean={ref['forecast_mean']:.6f}")

    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)
    print(f"\n[TimesFM-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-3):
    manifest_path = os.path.join(reference_dir, "reference_manifest.json")
    if not os.path.exists(manifest_path):
        print("[TimesFM-Validate] ERROR: reference manifest not found")
        return False
    with open(manifest_path) as f:
        references = json.load(f)
    all_passed = True
    for key, ref in references.items():
        idx = key.split("_")[1]
        buddy_path = os.path.join(buddy_output_dir, f"buddy_forecast_{idx}.npy")
        if not os.path.exists(buddy_path):
            print(f"[TimesFM-Validate] SKIP: {buddy_path} not found")
            continue
        buddy_fc = np.load(buddy_path)
        ref_fc = np.load(os.path.join(reference_dir, f"ref_forecast_{idx}.npy"))
        max_abs_err = float(np.max(np.abs(ref_fc - buddy_fc)))
        mae = float(np.mean(np.abs(ref_fc - buddy_fc)))
        cos_sim = float(np.dot(ref_fc.flatten(), buddy_fc.flatten()) /
                        (np.linalg.norm(ref_fc) * np.linalg.norm(buddy_fc) + 1e-10))
        passed = cos_sim > 0.99 and max_abs_err < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  {status} | cos={cos_sim:.6f} | MAE={mae:.2e} | max_err={max_abs_err:.2e}")
        if not passed:
            all_passed = False
    print(f"\n[TimesFM-Validate] {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="TimesFM 2.5 Accuracy Validation")
    parser.add_argument("--mode", type=str, required=True, choices=["reference", "compare"])
    parser.add_argument("--output-dir", type=str, default="./validation_data")
    parser.add_argument("--reference-dir", type=str, default="./validation_data")
    parser.add_argument("--buddy-output-dir", type=str, default="./build")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if args.mode == "reference":
        generate_reference(args.output_dir)
    elif args.mode == "compare":
        success = compare_outputs(args.reference_dir, args.buddy_output_dir, args.tolerance)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
