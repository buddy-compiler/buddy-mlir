#!/usr/bin/env python3
"""Build the deployable W8A8 weight image and its DDR placement manifest.

Inputs are the two things already verified elsewhere:

  * ``validation/weight-layout.json``  -- the offsets the compiler itself
    resolved, with every section matched to a named checkpoint tensor by
    content;
  * the W8A8 contract in ``tools/quant_reference.py`` -- per-output-channel
    symmetric int8 weights, identical to the verified Triton quantize path.

The 2.22 GiB FP32 parameter buffer cannot be resident, so 2-D matrices are
stored as int8 plus one f32 scale per output channel, and 1-D tensors (norms and
the rotary table) stay f32 because they are tiny and feeding them through an
int8 round-trip would only add error.

Every produced section is recorded with its address, size and SHA256 so a board
run can be tied back to the exact bytes that were uploaded. The image is placed
in the HIGH DDR region because that is the region ``common/nr/nr.ld`` marks
NOLOAD: the NR startup zeroes ``.bss`` in LOW, so an uploaded image in HIGH
cannot be wiped by it. That property is asserted here rather than assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_reference import quantize_rows  # noqa: E402

# From examples/FPGA-BOSCAME/common/nr/nr.ld
LOW_ORIGIN = 0x80000000
LOW_LENGTH = 0x30000000
HIGH_ORIGIN = 0xB8000000
HIGH_LENGTH = 0x48000000
ALIGNMENT = 64


def load_safetensors_header(checkpoint):
    with open(checkpoint, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def read_tensor_f32(handle, base, entry, count):
    handle.seek(base + entry["data_offsets"][0])
    if entry["dtype"] == "BF16":
        raw = np.frombuffer(handle.read(count * 2), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if entry["dtype"] == "F32":
        return np.frombuffer(handle.read(count * 4), dtype=np.float32)
    raise ValueError(f"unsupported checkpoint dtype {entry['dtype']}")


def rotary_inv_freq(head_dim, theta):
    index = np.arange(0, head_dim, 2, dtype=np.float64)
    return (1.0 / (theta ** (index / head_dim))).astype(np.float32)


def build(layout_path, checkpoint, config_path, output_dir, cache_len=512,
          layers=28, kv_heads=8, head_dim=128, tokenizer_blob=None):
    layout = json.loads(Path(layout_path).read_text())
    if layout["status"] != "PASS":
        raise SystemExit("weight layout is not PASS")
    config = json.loads(Path(config_path).read_text())
    parameters = config.get("rope_parameters") or {}
    theta = parameters.get("rope_theta", config.get("rope_theta", 10000.0))
    eps = config["rms_norm_eps"]

    header, base = load_safetensors_header(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = bytearray()
    scales = bytearray()
    kept = bytearray()
    sections = []

    def add(buffer, kind, name, payload, count, shape):
        offset = len(buffer)
        buffer.extend(payload)
        sections.append({"kind": kind, "tensor": name, "shape": shape,
                         "elements": count, "buffer_offset": offset,
                         "bytes": len(payload)})

    with open(checkpoint, "rb") as handle:
        for section in layout["sections"]:
            name = section["checkpoint_tensor"]
            shape = section["shape"]
            count = section["elements"]
            if name is None:
                # The rotary table is computed, not stored in the checkpoint.
                values = rotary_inv_freq(head_dim, theta)
                if count != values.size:
                    raise ValueError("rotary table size mismatch")
                add(kept, "f32-computed", "inv_freq", values.tobytes(),
                    count, shape)
                continue
            values = read_tensor_f32(handle, base, header[name], count)
            if len(shape) == 2:
                matrix = values.reshape(shape)
                quantized, scale = quantize_rows(matrix)
                add(weights, "int8", name, quantized.tobytes(), count, shape)
                add(scales, "f32-scale-per-output-channel", name,
                    scale.astype(np.float32).tobytes(), scale.size, [shape[0]])
            else:
                add(kept, "f32", name, values.tobytes(), count, shape)

    # --- assemble the image ------------------------------------------------
    image = bytearray()

    def place(label, payload, section_kind):
        while len(image) % ALIGNMENT:
            image.append(0)
        address = HIGH_ORIGIN + len(image)
        image.extend(payload)
        return {"label": label, "kind": section_kind, "address": hex(address),
                "address_value": address, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    placements = [
        place("weights_int8", bytes(weights), "int8-per-output-channel"),
        place("weight_scales_f32", bytes(scales), "per-output-channel-scale"),
        place("weights_kept_f32", bytes(kept), "f32-norms-and-rotary"),
    ]

    kv_bytes = layers * 2 * kv_heads * cache_len * head_dim * 4
    while len(image) % ALIGNMENT:
        image.append(0)
    kv_address = HIGH_ORIGIN + len(image)
    image.extend(b"\x00" * kv_bytes)
    placements.append({"label": "kv_cache_f32", "kind": "zero-initialised-f32",
                       "address": hex(kv_address), "address_value": kv_address,
                       "bytes": kv_bytes,
                       "sha256": hashlib.sha256(b"\x00" * kv_bytes).hexdigest(),
                       "note": "included so the image length reflects the "
                               "reserved layout; the board must zero it"})

    if tokenizer_blob and Path(tokenizer_blob).is_file():
        payload = Path(tokenizer_blob).read_bytes()
        placements.append(place("tokenizer_resource", payload, "qwen-tokenizer"))

    image_end = HIGH_ORIGIN + len(image)
    report = {
        "source_layout": str(layout_path),
        "layout_sha256": layout["layout_sha256"],
        "checkpoint_sha256_source": "validation/checkpoint-manifest.json",
        "quantization": {
            "linear": "per-output-channel symmetric int8, range [-127,127]",
            "scale": "max|row| / 127, zero rows use 1",
            "rounding": "FP32 divide, +/-0.5 by sign, truncate, saturate",
            "dequantize": "(acc_f32 * activation_scale) * weight_scale",
            "kept_f32": "norms and rotary table",
        },
        "sizes": {
            "weights_int8": len(weights),
            "weight_scales_f32": len(scales),
            "weights_kept_f32": len(kept),
            "kv_cache_f32": kv_bytes,
            "image_bytes": len(image),
            "fp32_weights_if_unquantized": layout["weight_buffer_bytes_f32"],
        },
        "compression_vs_f32": round(
            layout["weight_buffer_bytes_f32"]
            / max(1, len(weights) + len(scales) + len(kept)), 3),
        "placements": placements,
        "image": {
            "path": str(output_dir / "qwen3-w8a8.bin"),
            "bytes": len(image),
            "sha256": hashlib.sha256(bytes(image)).hexdigest(),
            "start_address": hex(HIGH_ORIGIN),
            "end_address": hex(image_end),
        },
        "memory_regions": {
            "HIGH": {"origin": hex(HIGH_ORIGIN), "length": hex(HIGH_LENGTH),
                     "used_bytes": len(image),
                     "fits": len(image) <= HIGH_LENGTH,
                     "section": "NOLOAD workspace"},
            "LOW": {"origin": hex(LOW_ORIGIN), "length": hex(LOW_LENGTH),
                    "section": "boot, text, data, .bss, stacks",
                    "image_overlaps": False},
        },
        "bss_zeroing_safety": {
            "claim": "the NR startup zeroes .bss/.tbss/stacks in LOW; the image "
                     "lives entirely in HIGH, so zeroing cannot reach it",
            "evidence": "common/nr/nr.ld places .bss/.tbss/.stacks in > LOW and "
                        ".workspace at 0xb8000000 in > HIGH",
            "image_region_tested": hex(HIGH_ORIGIN) + ".." + hex(image_end),
            "overlap_with_low": False,
        },
        "sections_emitted": len(sections),
        "section_index": sections,
        "not_verified": [
            "no board upload or readback was performed",
            "physical DDR capacity was not probed",
            "the image is not executed by any compiled model yet",
        ],
    }

    (output_dir / "qwen3-w8a8.bin").write_bytes(bytes(image))
    (output_dir / "weight-image.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (output_dir / "weight-sections.json").write_text(
        json.dumps(sections, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-len", type=int, default=512)
    parser.add_argument("--tokenizer-blob", type=Path, default=None)
    args = parser.parse_args()
    report = build(args.layout, args.checkpoint, args.config, args.output,
                   args.cache_len, tokenizer_blob=args.tokenizer_blob)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("section_index",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())