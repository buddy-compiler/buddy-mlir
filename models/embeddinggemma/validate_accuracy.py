#!/usr/bin/env python3
# ===- validate_accuracy.py -------------------------------------------------
#
# embeddinggemma-300m Accuracy Validation Script
#
# Compares HuggingFace SentenceTransformer outputs against Buddy-MLIR
# compiled outputs.  The model produces 768-dim normalized embeddings.
#
# Validation metric: cosine similarity between reference and buddy embeddings.
#
# ===---------------------------------------------------------------------------

import argparse
import json
import os
import sys
import numpy as np
import torch


def generate_reference(output_dir: str):
    """Generate HF reference embeddings for validation."""
    os.makedirs(output_dir, exist_ok=True)

    print("[EmbeddingGemma-Validate] Loading HF model...")
    from import_model import EmbeddingGemmaWrapper
    model = EmbeddingGemmaWrapper("google/embeddinggemma-300m")
    model.eval()

    test_sentences = [
        "hello world",
        "what is the weather like today",
        "artificial intelligence and machine learning",
        "the quick brown fox jumps over the lazy dog",
    ]

    references = {}
    for i, text in enumerate(test_sentences):
        print(f"\n[EmbeddingGemma-Validate] Processing: {text!r}")

        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
        tokens = st_model.tokenize([text])
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        with torch.no_grad():
            embedding = model(input_ids, attention_mask)

        emb_np = embedding.detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb_np[0]))

        ref = {
            "text": text,
            "embedding_dim": int(emb_np.shape[1]),
            "l2_norm": norm,
            "embedding_mean": float(np.mean(emb_np)),
            "embedding_std": float(np.std(emb_np)),
            "top5_indices": np.argsort(emb_np[0])[-5:][::-1].tolist(),
            "top5_values": emb_np[0][np.argsort(emb_np[0])[-5:][::-1]].tolist(),
        }
        references[f"text_{i}"] = ref
        np.save(os.path.join(output_dir, f"ref_embedding_{i}.npy"), emb_np)
        print(f"   dim={ref['embedding_dim']}, L2 norm={norm:.6f}")

    # Pairwise cosine similarities
    print("\n[EmbeddingGemma-Validate] Computing pairwise cosine similarities...")
    cos_sims = {}
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            ei = np.load(os.path.join(output_dir, f"ref_embedding_{i}.npy"))[0]
            ej = np.load(os.path.join(output_dir, f"ref_embedding_{j}.npy"))[0]
            cs = float(np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej)))
            cos_sims[f"{i}_{j}"] = cs
            print(f"   cos_sim({i},{j}): {cs:.6f}")
    references["pairwise_cosine"] = cos_sims

    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)
    print(f"\n[EmbeddingGemma-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(reference_dir: str, buddy_output_dir: str, tolerance: float = 1e-3):
    """Compare Buddy-MLIR embeddings against HF reference."""
    manifest_path = os.path.join(reference_dir, "reference_manifest.json")
    if not os.path.exists(manifest_path):
        print("[EmbeddingGemma-Validate] ERROR: reference manifest not found")
        return False

    with open(manifest_path) as f:
        references = json.load(f)

    all_passed = True
    for key, ref in references.items():
        if not key.startswith("text_"):
            continue
        idx = key.split("_")[1]
        buddy_path = os.path.join(buddy_output_dir, f"buddy_embedding_{idx}.npy")
        if not os.path.exists(buddy_path):
            print(f"[EmbeddingGemma-Validate] SKIP: {buddy_path} not found")
            continue

        buddy_emb = np.load(buddy_path)[0]
        ref_emb = np.load(os.path.join(reference_dir, f"ref_embedding_{idx}.npy"))[0]

        cos_sim = float(np.dot(ref_emb, buddy_emb) /
                        (np.linalg.norm(ref_emb) * np.linalg.norm(buddy_emb) + 1e-10))
        max_abs_err = float(np.max(np.abs(ref_emb - buddy_emb)))

        passed = cos_sim > 0.99 and max_abs_err < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  {status} | {ref['text'][:40]:40s} | cos={cos_sim:.6f} | max_err={max_abs_err:.2e}")

        if not passed:
            all_passed = False

    print(f"\n[EmbeddingGemma-Validate] {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="embeddinggemma-300m Accuracy Validation")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["reference", "compare"],
                       help="'reference' to generate HF outputs, 'compare' to validate buddy outputs")
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
