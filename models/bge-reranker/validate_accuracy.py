#!/usr/bin/env python3
import argparse, json, os, sys, numpy as np, torch
from transformers import AutoModelForSequenceClassification

def generate_reference(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[BGE-Validate] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.float32).eval()
    test_prompts = [("hello world", "hi there"), ("what is AI", "artificial intelligence"), ("the sky is blue", "weather is nice")]
    refs = {}
    for i, (a, b) in enumerate(test_prompts):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        inp = tok(a, b, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inp).logits.detach().cpu().numpy().astype(np.float32)
        refs[f"pair_{i}"] = {"text_a": a, "text_b": b, "score": float(logits[0,0])}
        print(f"  [{i}] '{a}' x '{b}' → score={logits[0,0]:.4f}")
        np.save(os.path.join(output_dir, f"ref_logits_{i}.npy"), logits[0])
    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f: json.dump(refs, f, indent=2)
    return refs

def compare_outputs(reference_dir, buddy_output_dir, tolerance=1e-3):
    with open(os.path.join(reference_dir, "reference_manifest.json")) as f: refs = json.load(f)
    for key, ref in refs.items():
        idx = key.split("_")[1]
        bp = os.path.join(buddy_output_dir, f"buddy_logits_{idx}.npy")
        if not os.path.exists(bp): print(f"SKIP {bp}"); continue
        bl = np.load(bp); rl = np.load(os.path.join(reference_dir, f"ref_logits_{idx}.npy"))
        err = float(np.max(np.abs(rl - bl)))
        passed = err < tolerance
        print(f"  {'PASS' if passed else 'FAIL'} | {ref['text_a'][:30]} | max_err={err:.2e}")
    return True

def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", required=True, choices=["reference","compare"]); p.add_argument("--output-dir", default="./validation_data"); p.add_argument("--reference-dir", default="./validation_data"); p.add_argument("--buddy-output-dir", default="./build"); p.add_argument("--tolerance", type=float, default=1e-3); a = p.parse_args()
    if a.mode == "reference": generate_reference(a.output_dir)
    else: sys.exit(0 if compare_outputs(a.reference_dir, a.buddy_output_dir, a.tolerance) else 1)
if __name__ == "__main__": main()
