#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare full-length texts and fixed-batch specs without running a model."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def exact_length_text(tokenizer, text, length):
    content_length = length - tokenizer.num_special_tokens_to_add()
    if content_length < 1:
        raise ValueError(
            "Sequence length must leave room for text and special tokens"
        )
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    pivot = encoded["offset_mapping"][content_length - 1][1]
    for end in range(max(1, pivot - 128), min(len(text), pivot + 128) + 1):
        candidate = text[:end].rstrip()
        if len(tokenizer(candidate)["input_ids"]) == length:
            return candidate
    raise ValueError(f"Cannot construct an exact {length}-token prefix")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=here / "specs/f32.json")
    parser.add_argument(
        "--corpus", type=Path, default=here / "workload-texts.json"
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 256, 512]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    if not corpus or not all(isinstance(text, str) and text for text in corpus):
        parser.error("Corpus must contain nonempty text strings")
    if min(args.batch_sizes) < 1 or max(args.batch_sizes) > len(corpus):
        parser.error("Batch sizes must fit the corpus")
    base = json.loads(args.spec.read_text())
    if (
        min(args.lengths) < 1
        or max(args.lengths) > base["max_position_embeddings"]
    ):
        parser.error("Invalid sequence length")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    matrix = []
    for length in args.lengths:
        texts = []
        for row in range(max(args.batch_sizes)):
            blocks = corpus[row:] + corpus[:row]
            document = " ".join(blocks)
            while len(tokenizer(document)["input_ids"]) < length:
                document += " " + " ".join(blocks)
            texts.append(exact_length_text(tokenizer, document, length))
        for batch_size in args.batch_sizes:
            name = f"f32-b{batch_size}-s{length}"
            selected = texts[:batch_size]
            tokens = tokenizer(
                selected,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="np",
            )
            if (
                tokens["input_ids"].shape != (batch_size, length)
                or not tokens["attention_mask"].all()
            ):
                raise ValueError("Prepared workload contains padding")
            spec = dict(base, batch_size=batch_size, max_seq_len=length)
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(spec, indent=2) + "\n"
            )
            (args.output_dir / f"{name}-texts.json").write_text(
                json.dumps(selected, indent=2) + "\n"
            )
            matrix.append(
                {
                    "name": name,
                    "spec": f"{name}.json",
                    "texts": f"{name}-texts.json",
                    "valid_tokens": [length] * batch_size,
                }
            )
            print(f"{name}: {batch_size} sequences, {length} valid tokens each")
    (args.output_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
