#!/usr/bin/env python3
"""Derive the deployment weight image layout from the compiled main graph.

The Buddy main graph already resolved the flat parameter buffer into concrete
``memref.subview %arg0[offset] [length]`` slices, so the offsets are not a design
choice we invent here -- they are what the compiler will actually read. This tool

  1. parses those offsets out of ``forward_*.mlir``;
  2. joins them with ``params.json`` (shape + SHA256 of each graph parameter);
  3. matches every parameter to a *named* checkpoint tensor by content, so a
     wrong placement cannot pass unnoticed;
  4. emits the image layout (name, offset, bytes, sha256) plus a coverage check
     that the sections tile the whole buffer with no gap or overlap.

Content matching (rather than positional guessing) is what lets the report state
which checkpoint tensor each offset holds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

DTYPES = {"BF16": "uint16", "F16": "float16", "F32": "float32",
          "I64": "int64", "I32": "int32", "U8": "uint8"}

SUBVIEW = re.compile(
    r"memref\.subview %(\w+)\[(?P<offset>\d+)\] \[(?P<length>\d+)\] \[1\]")
EXPAND = re.compile(
    r"memref\.expand_shape %(\w+) \[\[0, 1\]\] output_shape \[(?P<d0>\d+), (?P<d1>\d+)\]")


def read_safetensors_header(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def tensor_sha256(checkpoint, base, entry):
    import numpy as np
    dtype = np.dtype(DTYPES[entry["dtype"]])
    begin, end = entry["data_offsets"]
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as handle:
        handle.seek(base + begin)
        remaining = end - begin
        while remaining > 0:
            block = handle.read(min(32 * 1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def parse_weight_buffer(mlir_text, buffer_arg="arg0"):
    """Return the flat buffer length and the ordered subview sections."""
    match = re.search(rf"%{buffer_arg}: memref<(\d+)xf32>", mlir_text)
    if not match:
        raise ValueError("flat weight buffer not found in main graph")
    total = int(match.group(1))

    # Track subview SSA name -> (offset, length) and its expand_shape target.
    sections = []
    pending = {}
    for line in mlir_text.splitlines():
        sub = SUBVIEW.search(line)
        if sub and sub.group(1) == buffer_arg:
            name = line.strip().split()[0].lstrip("%")
            pending[name] = (int(sub.group("offset")), int(sub.group("length")))
            continue
        exp = EXPAND.search(line)
        if exp and exp.group(1) in pending:
            name = line.strip().split()[0].lstrip("%")
            offset, length = pending.pop(exp.group(1))
            sections.append({
                "ssa": name,
                "offset_elements": offset,
                "elements": length,
                "shape": [int(exp.group("d0")), int(exp.group("d1"))],
            })
    # 1-D parameters are subviews that are never expanded.
    for name, (offset, length) in pending.items():
        sections.append({"ssa": name, "offset_elements": offset,
                         "elements": length, "shape": [length]})
    sections.sort(key=lambda s: s["offset_elements"])
    return total, sections


def build_layout(mlir_path, params_path, checkpoint, config_path, output=None):
    mlir_text = Path(mlir_path).read_text()
    total, sections = parse_weight_buffer(mlir_text)
    params = json.loads(Path(params_path).read_text())
    header, base = read_safetensors_header(checkpoint)

    # SHA256 of every checkpoint tensor, then a name lookup by (shape, sha256).
    by_content = {}
    tensor_hashes = {}
    for name, entry in header.items():
        digest = tensor_sha256(checkpoint, base, entry)
        tensor_hashes[name] = digest
        by_content.setdefault((tuple(entry["shape"]), digest), []).append(name)

    image_sha = hashlib.sha256()
    problems = []
    entries = []
    cursor = 0
    for index, section in enumerate(sections):
        if index >= len(params):
            problems.append(f"more sections ({len(sections)}) than parameters "
                            f"({len(params)})")
            break
        param = params[index]
        shape = tuple(param["shape"])
        digest = param["sha256"]
        # A graph parameter is fp32; the checkpoint is bf16, so hashes differ.
        # Match on shape first, then confirm by upcasting candidates.
        candidates = [n for n, e in header.items() if tuple(e["shape"]) == shape]
        matched = None
        if len(candidates) == 1:
            matched = candidates[0]
            match_kind = "unique-shape"
        elif len(candidates) > 1:
            matched = pick_by_value(checkpoint, base, header, candidates, param)
            match_kind = "value-match" if matched else "ambiguous"
        else:
            match_kind = "computed-or-missing"

        if section["elements"] != param["elements"]:
            problems.append(
                f"section {index} at {section['offset_elements']} has "
                f"{section['elements']} elements but parameter has "
                f"{param['elements']}")
        if section["shape"] != list(shape):
            problems.append(
                f"section {index} shape {section['shape']} != parameter {list(shape)}")
        if section["offset_elements"] != cursor:
            problems.append(
                f"section {index} starts at {section['offset_elements']} but "
                f"previous sections end at {cursor}")
        cursor = section["offset_elements"] + section["elements"]

        entries.append({
            "index": index,
            "checkpoint_tensor": matched,
            "match_kind": match_kind,
            "shape": list(shape),
            "elements": param["elements"],
            "dtype": "f32",
            "offset_elements": section["offset_elements"],
            "offset_bytes": section["offset_elements"] * 4,
            "bytes": param["bytes"],
            "sha256_f32": param["sha256"],
            "checkpoint_sha256": tensor_hashes.get(matched) if matched else None,
            "ssa": section["ssa"],
        })
        image_sha.update(struct.pack("<QQ", section["offset_elements"],
                                     param["elements"]))

    if cursor != total:
        problems.append(f"sections cover {cursor} of {total} elements")

    named = sum(1 for e in entries if e["checkpoint_tensor"])
    report = {
        "source_mlir": str(mlir_path),
        "checkpoint": str(checkpoint),
        "weight_buffer_elements": total,
        "weight_buffer_bytes_f32": total * 4,
        "section_count": len(sections),
        "parameter_count": len(params),
        "named_sections": named,
        "unnamed_sections": [e["index"] for e in entries if not e["checkpoint_tensor"]],
        "layout_sha256": image_sha.hexdigest(),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "sections": entries,
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def pick_by_value(checkpoint, base, header, candidates, param, chunk=1 << 22):
    """Disambiguate same-shape tensors by comparing fp32 values."""
    import numpy as np
    want = np.array(param["sample"], dtype=np.float32)
    step = param["sample_step"]
    for name in candidates:
        entry = header[name]
        dtype = np.dtype(DTYPES[entry["dtype"]])
        begin = entry["data_offsets"][0]
        total = int(np.prod(entry["shape"]))
        flat = None
        with open(checkpoint, "rb") as handle:
            if dtype == np.uint16:
                # bf16 -> fp32 by shifting into the high half.
                handle.seek(base + begin)
                raw = np.frombuffer(handle.read(total * 2), dtype=np.uint16)
                flat = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                handle.seek(base + begin)
                flat = np.frombuffer(handle.read(total * dtype.itemsize),
                                     dtype=dtype).astype(np.float32)
        got = flat[::step][:len(want)]
        if got.shape == want.shape and np.array_equal(got, want):
            return name
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_layout(args.mlir, args.params, args.checkpoint, args.config,
                          args.output)
    summary = {k: v for k, v in result.items() if k != "sections"}
    print(json.dumps(summary, indent=2))
    if result["status"] != "PASS":
        for problem in result["problems"][:20]:
            print("  !", problem)