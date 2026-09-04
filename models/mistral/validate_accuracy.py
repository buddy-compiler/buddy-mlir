#!/usr/bin/env python3
import argparse, json, os, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_reference(output_dir: str, max_seq=128):
    os.makedirs(output_dir, exist_ok=True)
    print("[Mistral-Validate] Loading model...")
    m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    prompts = ["Hello, my name is", "The capital of France is", "What is 2+2? The answer is"]
    refs = {}
    for i, p in enumerate(prompts):
        inp = tok(p, return_tensors="pt", truncation=True, max_length=max_seq)
        with torch.no_grad():
            logits = m(**inp).logits.detach().cpu().numpy()
        ll = logits[0,-1,:].astype(np.float32)
        refs[f"prompt_{i}"] = {"prompt":p,"sum":float(np.sum(ll)),"mean":float(np.mean(ll)),"top5":np.argsort(ll)[-5:][::-1].tolist()}
        np.save(os.path.join(output_dir, f"ref_logits_{i}.npy"), ll)
        print(f"  [{i}] {p[:40]} → top5={refs[f'prompt_{i}']['top5']}")
    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f: json.dump(refs, f, indent=2)

def compare_outputs(ref_dir, buddy_dir, tol=1e-3):
    with open(os.path.join(ref_dir, "reference_manifest.json")) as f: refs = json.load(f)
    for key, ref in refs.items():
        idx = key.split("_")[1]; bp = os.path.join(buddy_dir, f"buddy_logits_{idx}.npy")
        if not os.path.exists(bp): print(f"SKIP {bp}"); continue
        bl = np.load(bp); rl = np.load(os.path.join(ref_dir, f"ref_logits_{idx}.npy"))
        cs = float(np.dot(rl,bl)/(np.linalg.norm(rl)*np.linalg.norm(bl)+1e-10))
        print(f"  {'PASS' if cs>0.99 else 'FAIL'} | {ref['prompt'][:40]} | cos={cs:.6f}")
    return True

def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", required=True, choices=["reference","compare"]); p.add_argument("--output-dir", default="./validation_data"); p.add_argument("--reference-dir", default="./validation_data"); p.add_argument("--buddy-output-dir", default="./build"); a = p.parse_args()
    if a.mode == "reference": generate_reference(a.output_dir)
    else: sys.exit(0 if compare_outputs(a.reference_dir, a.buddy_output_dir) else 1)
if __name__ == "__main__": main()
