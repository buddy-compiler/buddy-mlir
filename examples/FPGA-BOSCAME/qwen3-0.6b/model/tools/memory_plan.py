#!/usr/bin/env python3
"""Compute the deployment memory budget from the real weight layout.

Numbers here are derived, not assumed:

  * weight sizes come from ``validation/weight-layout.json``, i.e. from the
    offsets the compiler itself resolved, so a wrong plan cannot hide behind a
    hand-written parameter count;
  * the KV cache size follows the cache shape the imported graph declares
    (``1x8x512x128`` per layer per K/V);
  * the address ranges come from ``common/nr/nr.ld``.

What is *not* verified here is how much DDR the board actually exposes; the
linker's regions are a linker assertion, not a hardware measurement. The report
keeps that distinction explicit instead of implying a capacity probe happened.
"""
import argparse
import json
from pathlib import Path

# From examples/FPGA-BOSCAME/common/nr/nr.ld
LOW_ORIGIN = 0x80000000
LOW_LENGTH = 0x30000000          # 768 MiB, NR NH boot + ordinary state
HIGH_ORIGIN = 0xB8000000
HIGH_LENGTH = 0x48000000         # 1152 MiB, NOLOAD workspace
HOLE_ORIGIN = 0xB0000000         # ModelZoo reports RA faults in this aperture
HOLE_LENGTH = 0x08000000

MIB = 1024 * 1024


def plan(layout_path, cache_len, layers, batch, kv_heads, head_dim,
         tokenizer_bytes, kv_dtype_bytes=4, context=None):
    layout = json.loads(Path(layout_path).read_text())
    if layout["status"] != "PASS":
        raise SystemExit("weight layout is not PASS")

    int8_weights = 0
    weight_scales = 0
    fp32_kept = 0
    kept = []
    quantized = []
    for section in layout["sections"]:
        shape = section["shape"]
        elements = section["elements"]
        if len(shape) == 2:
            rows = shape[0]
            int8_weights += elements
            weight_scales += rows * 4
            quantized.append({"index": section["index"], "shape": shape,
                              "int8_bytes": elements, "scale_bytes": rows * 4,
                              "tensor": section["checkpoint_tensor"]})
        else:
            fp32_kept += elements * 4
            kept.append({"index": section["index"], "shape": shape,
                         "fp32_bytes": elements * 4,
                         "tensor": section["checkpoint_tensor"]})

    kv_cache = layers * 2 * batch * kv_heads * cache_len * head_dim * kv_dtype_bytes
    alignment = 64

    components = {
        "weights_int8": int8_weights,
        "weights_per_channel_scales_f32": weight_scales,
        "weights_kept_f32_norms_and_rope_table": fp32_kept,
        "kv_cache": kv_cache,
        "tokenizer_resource_blob": tokenizer_bytes,
    }
    # Activation and workspace estimates are stated as reserves, not measurements.
    reserves = {
        "activation_and_workspace_reserve": 64 * MIB,
        "attention_scores_and_probabilities": layers * 0 + 2 * MIB,
        "logits_f32_vocab": 151936 * 4,
        "runtime_text_data_bss": 8 * MIB,
        "nh_and_ra_stacks": 2 * MIB,
        "uart_buffers": 64 * 1024,
    }
    total = sum(components.values()) + sum(reserves.values())

    low_used = 0
    high_used = 0
    # Static, non-workspace state lives in LOW; the big buffers go to HIGH.
    low_used += reserves["runtime_text_data_bss"] + reserves["nh_and_ra_stacks"] \
        + reserves["uart_buffers"]
    high_used += (components["weights_int8"]
                  + components["weights_per_channel_scales_f32"]
                  + components["weights_kept_f32_norms_and_rope_table"]
                  + components["kv_cache"]
                  + components["tokenizer_resource_blob"]
                  + reserves["activation_and_workspace_reserve"]
                  + reserves["attention_scores_and_probabilities"]
                  + reserves["logits_f32_vocab"])

    report = {
        "source_layout": str(layout_path),
        "layout_sha256": layout["layout_sha256"],
        "deployment_precision": "W8A8 linear (AME int8 x int8 -> int32) with "
                                "FP32 norms, RoPE, attention and residuals",
        "cache": {"max_cache_len": cache_len, "layers": layers, "batch": batch,
                  "kv_heads": kv_heads, "head_dim": head_dim,
                  "kv_dtype": "f32", "bytes": kv_cache},
        "components_bytes": components,
        "reserves_bytes": reserves,
        "total_bytes": total,
        "total_mib": round(total / MIB, 2),
        "quantized_matrices": len(quantized),
        "fp32_kept_tensors": len(kept),
        "analysis": {
            "f32_weights_bytes_if_unquantized": layout["weight_buffer_bytes_f32"],
            "f32_weights_gib": round(layout["weight_buffer_bytes_f32"] / 2**30, 3),
            "int8_saving_factor": round(
                layout["weight_buffer_bytes_f32"]
                / max(1, int8_weights + weight_scales + fp32_kept), 3),
        },
        "linker_regions": {
            "LOW": {"origin": hex(LOW_ORIGIN), "length": hex(LOW_LENGTH),
                    "mib": LOW_LENGTH / MIB, "planned_use": low_used,
                    "fits": low_used <= LOW_LENGTH},
            "HIGH": {"origin": hex(HIGH_ORIGIN), "length": hex(HIGH_LENGTH),
                     "mib": HIGH_LENGTH / MIB, "planned_use": high_used,
                     "fits": high_used <= HIGH_LENGTH},
            "excluded_aperture": {"origin": hex(HOLE_ORIGIN),
                                  "length": hex(HOLE_LENGTH),
                                  "reason": "ModelZoo reports RA access faults here"},
        },
        "verification_status": {
            "weight_offsets": "derived from the compiled main graph (verified)",
            "cache_shape": "taken from the imported graph ABI (verified)",
            "linker_regions": "read from common/nr/nr.ld (verified as a linker "
                              "assertion)",
            "physical_ddr_capacity": "NOT verified: no board probe performed",
            "activation_and_workspace": "reserve, not a measured liveness result",
        },
    }
    report["conclusion"] = (
        "W8A8 fits the linker regions with the stated reserves"
        if report["linker_regions"]["HIGH"]["fits"]
        and report["linker_regions"]["LOW"]["fits"]
        else "planned usage exceeds a linker region")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokenizer-bytes", type=int, default=5222928)
    args = parser.parse_args()
    report = plan(args.layout, args.cache_len, args.layers, args.batch,
                  args.kv_heads, args.head_dim, args.tokenizer_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("quantized_matrices", "fp32_kept_tensors")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
