#!/usr/bin/env python3
"""Pack official BPE resources for bounded, filesystem-free access on NR.

This is resource conversion, not host-side inference/tokenization. The C reader
and encoder consume this format without a filesystem. Unicode tables for the
encoder are generated separately by gen_unicode_tables.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

MAGIC = b"QBPTOK1\0"
EXPECTED_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped = dict(zip(values, values))
    for value in range(256):
        if value not in mapped:
            mapped[value] = 256 + len(mapped) - len(values)
    return {chr(unicode): byte for byte, unicode in mapped.items()}


def pack(directory, output):
    tokenizer_path = directory / "tokenizer.json"
    doc = json.loads(tokenizer_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    settings = json.loads((directory / "tokenizer_config.json").read_text())
    bpe = doc["model"]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False,
                  "trim_offsets": False, "use_regex": False}
    expected_pre = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": EXPECTED_REGEX},
         "behavior": "Isolated", "invert": False}, byte_level]}
    pattern = EXPECTED_REGEX
    if (bpe["type"] != "BPE" or doc["normalizer"] != {"type": "NFC"}
            or doc["pre_tokenizer"] != expected_pre
            or doc["post_processor"] != byte_level
            or doc.get("truncation") is not None or doc.get("padding") is not None):
        raise ValueError("unsupported tokenizer normalizer or pre-tokenizer; do not silently approximate")
    if doc["decoder"] != byte_level or any(bpe.get(k) for k in
            ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "byte_fallback", "ignore_merges")):
        raise ValueError("unsupported BPE/decoder configuration")
    vocab, added = bpe["vocab"], doc["added_tokens"]
    slots = config["vocab_size"]
    decode_byte = byte_alphabet()
    entries = [(0, 0, 0)] * slots
    pieces = bytearray()
    def append(token_id, raw, flags):
        if not 0 <= token_id < slots or entries[token_id][2]:
            raise ValueError("invalid or duplicate token ID")
        entries[token_id] = (len(pieces), len(raw), flags)
        pieces.extend(raw)
    for spelling, token_id in vocab.items():
        append(token_id, bytes(decode_byte[c] for c in spelling), 1)
    for token in added:
        if any(token[k] for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("added-token matching flags require an additional implementation")
        append(token["id"], token["content"].encode(), 2 | (4 if token["special"] else 0))
    merges = []
    seen = set()
    for rank, pair in enumerate(bpe["merges"]):
        left, right = pair.split(" ") if isinstance(pair, str) else pair
        key = (vocab[left], vocab[right])
        if key in seen:
            raise ValueError("duplicate merge pair")
        seen.add(key)
        merges.append((*key, vocab[left + right], rank))
    merges.sort(key=lambda row: row[:2])
    header_size = 64
    records_off = header_size
    pieces_off = records_off + slots * 12
    merge_off = (pieces_off + len(pieces) + 3) & ~3
    added_off = merge_off + len(merges) * 16
    byte_off = added_off + len(added) * 4
    total = byte_off + 256 * 4
    blob = bytearray(total)
    struct.pack_into("<8s14I", blob, 0, MAGIC, 1, total, slots, len(vocab), len(merges), len(added),
                     records_off, pieces_off, len(pieces), merge_off, added_off, byte_off, 0, 0)
    for index, record in enumerate(entries):
        struct.pack_into("<III", blob, records_off + index * 12, *record)
    blob[pieces_off:pieces_off + len(pieces)] = pieces
    for index, record in enumerate(merges):
        struct.pack_into("<IIII", blob, merge_off + index * 16, *record)
    # Longest-first order makes exact added-token matching deterministic.
    for index, token in enumerate(sorted(added, key=lambda x: (-len(x["content"].encode()), x["id"]))):
        struct.pack_into("<I", blob, added_off + index * 4, token["id"])
    inverse = {value: key for key, value in decode_byte.items()}
    for byte in range(256):
        struct.pack_into("<I", blob, byte_off + byte * 4, vocab[inverse[byte]])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    generation = json.loads((directory / "generation_config.json").read_text())
    report = {"format": "QBPTOK1", "bytes": total, "sha256": hashlib.sha256(blob).hexdigest(),
              "model_vocab_slots": slots, "base_vocab_entries": len(vocab), "added_token_entries": len(added),
              "unassigned_model_slots": [i for i, row in enumerate(entries) if not row[2]],
              "merges": len(merges), "normalizer": doc["normalizer"], "pretokenizer_regex": pattern,
              "automatic_bos": settings.get("add_bos_token"), "tokenizer_bos": settings.get("bos_token"),
              "model_bos_id": config["bos_token_id"], "model_eos_id": config["eos_token_id"],
              "generation_eos_ids": generation["eos_token_id"], "special_tokens": added,
              "sources": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
                          ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json")},
              "encoder_status": "implemented in text/tokenizer_encode.c; host and FPGA validation are separate evidence"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = pack(args.assets, args.output)
    print(f"{report['bytes']} bytes, {report['base_vocab_entries']} BPE + {report['added_token_entries']} added tokens; {args.output}")
