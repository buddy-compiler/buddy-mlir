#!/usr/bin/env python3
import argparse, json, os, sys, numpy as np, torch
from transformers import AutoModel

def generate_reference(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[SmolVLM2-Validate] Loading model...")
    m = AutoModel.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct", dtype=torch.float32).eval()
    import types
    for mod in m.modules():
        if hasattr(mod.forward, "__wrapped__"): mod.forward = types.MethodType(mod.forward.__wrapped__, mod)
    pix = torch.zeros((1, 1, 3, 512, 512)); ids = torch.ones((1, 64), dtype=torch.int64); am = torch.ones((1, 64), dtype=torch.int64)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=am, pixel_values=pix)
    logits = out.last_hidden_state.detach().cpu().numpy().astype(np.float32) if hasattr(out, 'last_hidden_state') else out[0].detach().cpu().numpy().astype(np.float32)
    ll = logits[0, -1, :]
    ref = {"shape": list(logits.shape), "sum": float(np.sum(ll)), "mean": float(np.mean(ll)), "top5": np.argsort(ll)[-5:][::-1].tolist()}
    np.save(os.path.join(output_dir, "ref_logits.npy"), ll)
    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f: json.dump(ref, f, indent=2)
    print(f"  logits={ref['shape']}, top5={ref['top5']}")

def compare_outputs(ref_dir, buddy_dir, tol=1e-2):
    with open(os.path.join(ref_dir, "reference_manifest.json")) as f: ref = json.load(f)
    bp = os.path.join(buddy_dir, "buddy_logits.npy")
    if not os.path.exists(bp): print(f"SKIP {bp}"); return False
    bl = np.load(bp); rl = np.load(os.path.join(ref_dir, "ref_logits.npy"))
    cs = float(np.dot(rl, bl) / (np.linalg.norm(rl) * np.linalg.norm(bl) + 1e-10))
    print(f"  cos_sim={cs:.6f} {'PASS' if cs > 0.99 else 'FAIL'}")
    return cs > 0.99

def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", required=True, choices=["reference", "compare"]); p.add_argument("--output-dir", default="./validation_data"); p.add_argument("--reference-dir", default="./validation_data"); p.add_argument("--buddy-output-dir", default="./build"); a = p.parse_args()
    if a.mode == "reference": generate_reference(a.output_dir)
    else: sys.exit(0 if compare_outputs(a.reference_dir, a.buddy_output_dir) else 1)
if __name__ == "__main__": main()
