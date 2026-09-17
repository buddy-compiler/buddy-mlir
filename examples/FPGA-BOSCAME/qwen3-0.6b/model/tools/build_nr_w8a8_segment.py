#!/usr/bin/env python3
"""Emit the fully-quantised W8A8 parameter segment the compiled graph expects.

The graph's parameter list *is* the layout: the compiled entry takes its
parameters positionally, so the segment has to be one flat buffer in exactly that
order, with every parameter 64-byte aligned. This tool walks ``graph.params`` from
the rewrite report, obtains each parameter's bytes --

  * int8 weight matrices and their per-output-channel scales, quantised from the
    checkpoint with the contract the Triton kernels implement,
  * the f32 RMSNorm weights and the rotary table, which stay in FP32,

-- and records where each one landed so the bare-metal main can build a
descriptor for it.

It also writes the DDR load plan. Unlike the FP32 segment this one is 570 MiB
rather than 2.2 GiB, which is the whole reason the full model can be placed at all
(validation/board/fp32-28layer-does-not-fit.json).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALIGN = 64


def checkpoint_tensor_map(layout, original_parameters):
    """original parameter name -> checkpoint tensor name (or None for computed)."""
    sections = layout["sections"]
    if len(sections) != len(original_parameters):
        raise SystemExit(
            f"the rewrite reports {len(original_parameters)} original parameters "
            f"but the layout has {len(sections)} sections")
    return {name: section["checkpoint_tensor"]
            for name, section in zip(original_parameters, sections)}


def source_values(flat, layout, original_parameters):
    """original parameter name -> its slice of the FP32 buffer."""
    sections = layout["sections"]
    out = {}
    for name, section in zip(original_parameters, sections):
        start = section["offset_elements"]
        out[name] = flat[start:start + section["elements"]]
    return out


def build(args):
    import run_graph_host as R
    import quant_model_reference as Q
    from quant_reference import quantize_rows

    report = json.loads(args.report.read_text())
    layout = json.loads(args.layout.read_text())
    config = json.loads(args.config.read_text())

    # Which graph's parameter list to materialise. prefill and decode describe the
    # same weights (only the row counts differ), so either serves; the caller
    # passes the one the image was built from.
    graph = report["graphs"][args.graph]
    parameters = graph["parameters"]

    flat, _, computed = R.build_weight_buffer(layout, args.checkpoint, np.float32)
    # The rotary table is not in the checkpoint; it is derived from rope_theta.
    # run_graph_host.rotary_inv_freq expects a config object, so the formula is
    # applied here directly from the JSON config.
    head_dim = config["head_dim"]
    rope = config.get("rope_parameters") or {}
    theta = rope.get("rope_theta", config.get("rope_theta", 10000.0))
    for section in computed:
        if section["shape"] != [head_dim // 2]:
            raise SystemExit(f"unexpected computed section {section['shape']}")
        index = np.arange(0, head_dim, 2, dtype=np.float64)
        values = (1.0 / (theta ** (index / head_dim))).astype(np.float32)
        start = section["offset_elements"]
        flat[start:start + section["elements"]] = values
    original = report["original_parameters"]
    sources = source_values(flat, layout, original)
    tensors = checkpoint_tensor_map(layout, original)

    weights = Q.QuantizedWeights(args.checkpoint.parent, layout, config,
                                 quantize=True)

    # parameter name -> (dtype, bytes)
    payload = {}
    for entry in report.get("w8a8_linears") or []:
        source = sources[entry["weight_param"]]
        values = source.reshape(entry["n"], entry["k"])
        quantised, scale = quantize_rows(values)
        if entry["weight_param_name"] in payload:
            # The tied embedding/lm_head matrix is one parameter shared by two
            # call sites, so it appears twice in the rewrite report.
            continue
        payload[entry["weight_param_name"]] = ("i8", quantised.tobytes())
        payload[entry["scale_param_name"]] = ("f32", scale.astype(np.float32).tobytes())
    for entry in report.get("w8a8_embeddings") or []:
        source = sources[entry["source_param"]]
        width = entry["width"]
        values = source.reshape(len(source) // width, width)
        quantised, scale = quantize_rows(values)
        payload[entry["weights_param"]] = ("i8", quantised.tobytes())
        payload[entry["scale_param"]] = ("f32", scale.astype(np.float32).tobytes())

    # Everything else is FP32: the norm weights and the rotary table. The table
    # has no checkpoint tensor -- its values were just computed into `flat` -- so
    # the buffer is the source for every one of these, not the checkpoint.
    for name in original:
        if name in payload:
            continue
        payload[name] = ("f32", sources[name].astype(np.float32).tobytes())

    offsets = {}
    blob = bytearray()
    placement = []
    for parameter in parameters:
        name = parameter["name"]
        if name not in payload:
            raise SystemExit(f"no value for parameter {name}")
        dtype, raw = payload[name]
        while len(blob) % ALIGN:
            blob.append(0)
        offsets[name] = len(blob)
        blob.extend(raw)
        placement.append({"name": name, "dtype": dtype, "shape": parameter["shape"],
                          "offset_bytes": offsets[name], "bytes": len(raw)})

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    segment = output / args.name
    segment.write_bytes(bytes(blob))
    digest = hashlib.sha256(segment.read_bytes()).hexdigest()

    image = args.image.resolve() if args.image else None
    image_size = image.stat().st_size if image else 0
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest() if image else None
    if image and image_size % ALIGN:
        raise SystemExit(f"model image size {image_size} is not {ALIGN}-byte aligned")
    if image and segment.parent.resolve() != image.parent:
        raise SystemExit("the image and the segment must share a directory so the "
                         "plan can reference them relatively")
    if len(blob) % ALIGN or args.address % ALIGN:
        raise SystemExit("segment size and address must be aligned")

    plan = output / "ddr-load.plan"
    if image:
        plan.write_text("\n".join([
        "# Generated by model/tools/build_nr_w8a8_segment.py -- do not edit.",
        "version = 1",
        "ddr_base = 0x80000000",
        "ddr_size = 0x400000000",
        f"alignment = {ALIGN}",
        "",
        "[[segments]]",
        'name = "model"',
        'file = "image.bin"',
        "address = 0x80000000",
        f"size = {image_size}",
        f'sha256 = "{image_sha}"',
        "",
        "[[segments]]",
        'name = "weights"',
        f'file = "{segment.name}"',
        f"address = 0x{args.address:08x}",
        f"size = {len(blob)}",
        f'sha256 = "{digest}"',
        "",
        ]))

    record = {
        "stage": "W8A8 parameter segment for the compiled graph",
        "segment": str(segment), "bytes": len(blob), "sha256": digest,
        "address": hex(args.address), "alignment": ALIGN,
        "parameters": len(placement), "placement": placement,
        "plan": str(plan) if image else None,
        "image": {"file": image.name, "bytes": image_size, "sha256": image_sha,
                  "address": "0x80000000"} if image else None,
        "total_ddr_bytes": image_size + len(blob),
        "dtype_totals": {
            dtype: sum(p["bytes"] for p in placement if p["dtype"] == dtype)
            for dtype in ("i8", "f32")},
    }
    (output / "w8a8-segment.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="triton-call-replacement.json from the W8A8 rewrite")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="model.safetensors")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path,
                        help="optional legacy load plan; prefer preparing weights first, "
                             "then prepare_model_run.py with the final ELF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", default="prefill", choices=("prefill", "decode"))
    parser.add_argument("--address", type=lambda v: int(v, 0), default=0xB8000000)
    parser.add_argument("--name", default="weights-w8a8.bin")
    args = parser.parse_args()
    record = build(args)
    print(json.dumps({k: v for k, v in record.items() if k != "placement"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
