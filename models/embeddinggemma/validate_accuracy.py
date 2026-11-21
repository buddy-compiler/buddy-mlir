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

DEFAULT_SPEC = os.path.join(os.path.dirname(__file__), "specs", "f32.json")
DEFAULT_TOLERANCE = 1e-5


def generate_reference(
    output_dir: str,
    spec_path: str = DEFAULT_SPEC,
    texts_file: str | None = None,
):
    """Generate HF reference embeddings for validation."""
    import torch

    os.makedirs(output_dir, exist_ok=True)

    print("[EmbeddingGemma-Validate] Loading HF model...")
    from sentence_transformers import SentenceTransformer

    with open(spec_path) as f:
        spec = json.load(f)
    model_path = (
        os.environ.get("EMBEDDINGGEMMA_MODEL_PATH")
        or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
        or spec["hf_model_path"]
    )
    model = SentenceTransformer(model_path, device="cpu").float()
    model.eval()

    test_sentences = [
        "hello world",
        "what is the weather like today",
        "artificial intelligence and machine learning",
        "the quick brown fox jumps over the lazy dog",
    ]
    if texts_file is not None:
        with open(texts_file) as f:
            test_sentences = json.load(f)
    batch_size = int(spec.get("batch_size", 1))
    if (
        batch_size < 1
        or not isinstance(test_sentences, list)
        or not test_sentences
        or not all(isinstance(text, str) for text in test_sentences)
        or len(test_sentences) % batch_size
    ):
        raise ValueError(
            "Texts must contain complete, nonempty batches of strings"
        )

    references = {}
    for i, start in enumerate(range(0, len(test_sentences), batch_size)):
        texts = test_sentences[start : start + batch_size]
        print(
            f"\n[EmbeddingGemma-Validate] Processing batch {i}, B={batch_size}"
        )

        tokens = model.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=int(spec["max_seq_len"]),
            return_tensors="pt",
        )

        with torch.no_grad():
            embedding = model(tokens)["sentence_embedding"]

        emb_np = embedding.detach().cpu().numpy().astype(np.float32)
        if (
            emb_np.shape != (batch_size, int(spec["hidden_size"]))
            or not np.isfinite(emb_np).all()
        ):
            raise ValueError("Invalid HuggingFace reference embedding")
        norm = float(np.linalg.norm(emb_np[0]))

        ref = {
            "text": texts[0],
            "texts": texts,
            "batch_size": batch_size,
            "sequence_length": int(spec["max_seq_len"]),
            "valid_tokens": tokens["attention_mask"].sum(dim=1).tolist(),
            "embedding_dim": int(emb_np.shape[1]),
            "l2_norm": norm,
        }
        references[f"text_{i}"] = ref
        np.save(os.path.join(output_dir, f"ref_embedding_{i}.npy"), emb_np)
        print(f"   dim={ref['embedding_dim']}, L2 norm={norm:.6f}")

    with open(os.path.join(output_dir, "reference_manifest.json"), "w") as f:
        json.dump(references, f, indent=2)
    print(f"\n[EmbeddingGemma-Validate] Reference data saved to: {output_dir}")
    return references


def compare_outputs(
    reference_dir: str,
    buddy_output_dir: str,
    tolerance: float = DEFAULT_TOLERANCE,
):
    """Compare Buddy-MLIR embeddings against HF reference."""
    manifest_path = os.path.join(reference_dir, "reference_manifest.json")
    if not os.path.exists(manifest_path):
        print("[EmbeddingGemma-Validate] ERROR: reference manifest not found")
        return False

    with open(manifest_path) as f:
        references = json.load(f)

    all_passed = True
    compared_count = 0
    for key, ref in references.items():
        if not key.startswith("text_"):
            continue
        idx = key.split("_")[1]
        buddy_path = os.path.join(
            buddy_output_dir, f"buddy_embedding_{idx}.npy"
        )
        ref_path = os.path.join(reference_dir, f"ref_embedding_{idx}.npy")
        missing = [p for p in (buddy_path, ref_path) if not os.path.isfile(p)]
        if missing:
            print(
                f"[EmbeddingGemma-Validate] ERROR: missing embeddings: {missing}"
            )
            all_passed = False
            continue

        buddy_array = np.load(buddy_path, allow_pickle=False)
        ref_array = np.load(ref_path, allow_pickle=False)
        expected_shape = (
            int(ref.get("batch_size", 1)),
            int(ref["embedding_dim"]),
        )
        if (
            buddy_array.shape != expected_shape
            or ref_array.shape != expected_shape
            or buddy_array.size == 0
            or not np.isfinite(buddy_array).all()
            or not np.isfinite(ref_array).all()
        ):
            print(
                f"[EmbeddingGemma-Validate] ERROR: invalid embeddings for {key}"
            )
            all_passed = False
            continue
        texts = ref.get("texts", [ref["text"]])
        for row, (ref_emb, buddy_emb) in enumerate(zip(ref_array, buddy_array)):
            compared_count += 1
            cos_sim = float(
                np.dot(ref_emb, buddy_emb)
                / (np.linalg.norm(ref_emb) * np.linalg.norm(buddy_emb) + 1e-10)
            )
            max_abs_err = float(np.max(np.abs(ref_emb - buddy_emb)))
            passed = cos_sim > 0.99 and max_abs_err < tolerance
            status = "PASS" if passed else "FAIL"
            print(
                f"  {status} | row={row} {texts[row][:40]:40s} | cos={cos_sim:.6f} | max_err={max_abs_err:.2e}"
            )
            all_passed = passed and all_passed

    if compared_count == 0:
        print("[EmbeddingGemma-Validate] ERROR: no embeddings compared")
        all_passed = False

    print(
        f"\n[EmbeddingGemma-Validate] {'ALL PASSED' if all_passed else 'SOME FAILED'}"
    )
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="embeddinggemma-300m Accuracy Validation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["reference", "compare"],
        help="'reference' to generate HF outputs, 'compare' to validate buddy outputs",
    )
    parser.add_argument("--output-dir", type=str, default="./validation_data")
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC,
        help="Variant spec used to generate reference inputs",
    )
    parser.add_argument(
        "--texts-file",
        help="JSON list of texts, grouped by the spec batch_size",
    )
    parser.add_argument(
        "--reference-dir", type=str, default="./validation_data"
    )
    parser.add_argument("--buddy-output-dir", type=str, default="./build")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if args.mode == "reference":
        generate_reference(args.output_dir, args.spec, args.texts_file)
    elif args.mode == "compare":
        success = compare_outputs(
            args.reference_dir, args.buddy_output_dir, args.tolerance
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
