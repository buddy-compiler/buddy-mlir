#!/usr/bin/env python3
import argparse, json, os, sys, numpy as np, torch
from kokoro import KModel

def generate_reference(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[Kokoro-Validate] Loading Kokoro-82M...")
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
    dummy_ids = torch.randint(0, 100, (1, 30), dtype=torch.int64)
    dummy_ref = torch.randn(1, 256, dtype=torch.float32)
    with torch.no_grad():
        audio, dur = model.forward_with_tokens(input_ids=dummy_ids, ref_s=dummy_ref, speed=1.0)
    a = audio.cpu().numpy().astype(np.float32)
    d = dur.cpu().numpy().astype(np.int64)
    ref = {"audio_shape":list(a.shape),"audio_mean":float(np.mean(a)),"audio_std":float(np.std(a)),"dur_shape":list(d.shape)}
    np.save(os.path.join(output_dir,"ref_audio.npy"), a)
    np.save(os.path.join(output_dir,"ref_dur.npy"), d)
    with open(os.path.join(output_dir,"reference_manifest.json"),"w") as f: json.dump(ref,f,indent=2)
    print(f"  audio={a.shape}, dur={d.shape}")
    return ref

def compare_outputs(ref_dir, buddy_dir, tol=1e-2):
    with open(os.path.join(ref_dir,"reference_manifest.json")) as f: ref = json.load(f)
    bp = os.path.join(buddy_dir,"buddy_audio.npy")
    if not os.path.exists(bp): print(f"SKIP {bp}"); return False
    bl = np.load(bp); rl = np.load(os.path.join(ref_dir,"ref_audio.npy"))
    cs = float(np.dot(rl.flatten(),bl.flatten())/(np.linalg.norm(rl)*np.linalg.norm(bl)+1e-10))
    print(f"  cos_sim={cs:.6f} {'PASS' if cs>0.99 else 'FAIL'}")
    return cs>0.99

def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode",required=True,choices=["reference","compare"]); p.add_argument("--output-dir",default="./validation_data"); p.add_argument("--reference-dir",default="./validation_data"); p.add_argument("--buddy-output-dir",default="./build"); a = p.parse_args()
    if a.mode=="reference": generate_reference(a.output_dir)
    else: sys.exit(0 if compare_outputs(a.reference_dir,a.buddy_output_dir) else 1)
if __name__=="__main__": main()
