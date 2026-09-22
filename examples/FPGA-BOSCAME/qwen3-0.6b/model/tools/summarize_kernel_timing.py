#!/usr/bin/env python3
"""Summarize an accepted FPGA kernel profile by specialization and family.

Reads the strict model archive, checks the hashes of all consumed evidence, and
emits CSV/JSON/Markdown. Times are aggregates across layers; this does not invent
per-layer, individual-call, or graph-internal operator timings.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

PREFIX = "_mlir_ciface_kernel_"
LABELS = {
    "matmul_i8": "INT8 linear / AME",
    "per_token_quantization": "Activation Quantize / RVV",
    "dequantization": "Dequantize / RVV",
    "rmsnorm": "RMSNorm (hidden + Q/K norm)",
    "attention_qk": "Attention QK",
    "attention_pv": "Attention PV",
    "attention_scale_mask_position": "Attention scale + causal mask",
    "softmax": "Softmax",
    "silu": "SiLU",
    "kv_cache_update_position": "KV cache update",
    "layout_transpose": "Layout transpose",
    "embedding_w8a8": "Embedding W8A8",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(stages, cases, hz):
    require(hz > 0, "clock must be positive")
    require(len(stages) == 9 and
            [(s["kind"], s["graph_start_position"]) for s in stages] ==
            [("prefill", 0)] + [("decode", p) for p in range(16, 24)],
            "expected exactly one 16-token prefill and eight sequential decode steps")
    kernels, families, detail = {}, {}, []
    for step, stage in enumerate(stages):
        require(sum(k["cycles"] for k in stage["kernels"]) == stage["kernel_cycles"] and
                stage["compute_cycles"] == stage["kernel_cycles"] + stage["graph_cycles_outside_kernel_measurements"],
                "stage cycle totals disagree")
        for kernel in stage["kernels"]:
            symbol = kernel["symbol"]
            require(symbol.startswith(PREFIX), "unknown kernel ABI prefix")
            name = symbol[len(PREFIX):]
            require(name in cases, "kernel absent from linked archive: " + name)
            front = cases[name]
            family = front["family"]
            require(kernel["calls"] > 0 and kernel["cycles"] > 0, "nonpositive kernel count/time")
            for table, key in ((kernels, name), (families, family)):
                row = table.setdefault(key, {"name": key, "family": family,
                                             "calls_by_stage": [0] * 9, "cycles_by_stage": [0] * 9})
                row["calls_by_stage"][step] += kernel["calls"]
                row["cycles_by_stage"][step] += kernel["cycles"]
            detail.append({"stage": stage["kind"], "cache_position": stage["graph_start_position"],
                           "output_position": stage["position"], "case": name, "family": family,
                           "calls": kernel["calls"], "cycles": kernel["cycles"],
                           "milliseconds": kernel["cycles"] * 1000 / hz,
                           "mean_call_milliseconds": kernel["cycles"] * 1000 / (hz * kernel["calls"]),
                           "percent_graph": 100 * kernel["cycles"] / stage["compute_cycles"]})
    decode_graph = sum(s["compute_cycles"] for s in stages[1:])
    graph_total = sum(s["compute_cycles"] for s in stages)
    for table in (kernels, families):
        for row in table.values():
            c, t = row["calls_by_stage"], row["cycles_by_stage"]
            row.update(prefill_calls=c[0], decode_calls_per_step=sum(c[1:]) / 8,
                       prefill_milliseconds=t[0] * 1000 / hz,
                       decode_mean_milliseconds=sum(t[1:]) * 1000 / (8 * hz),
                       prefill_percent_graph=100 * t[0] / stages[0]["compute_cycles"],
                       decode_percent_graph=100 * sum(t[1:]) / decode_graph,
                       total_cycles=sum(t), total_percent_graph=100 * sum(t) / graph_total,
                       prefill_mean_call_milliseconds=t[0] * 1000 / (hz * c[0]) if c[0] else None,
                       decode_mean_call_milliseconds=sum(t[1:]) * 1000 / (hz * sum(c[1:])) if sum(c[1:]) else None)
    phases = []
    for stage in stages:
        row = {k: v for k, v in stage.items() if k != "kernels"}
        for name, value in list(row.items()):
            if name.endswith("cycles") or name == "graph_cycles_outside_kernel_measurements":
                row[name + "_milliseconds"] = value * 1000 / hz
        phases.append(row)
    return {"kernels": sorted(kernels.values(), key=lambda x: -x["total_cycles"]),
            "families": sorted(families.values(), key=lambda x: -x["total_cycles"]),
            "per_stage_kernel": detail, "stages": phases,
            "graph_outside_kernels": {
                "prefill_milliseconds": stages[0]["graph_cycles_outside_kernel_measurements"] * 1000 / hz,
                "decode_mean_milliseconds": sum(s["graph_cycles_outside_kernel_measurements"] for s in stages[1:]) * 1000 / (8 * hz),
                "prefill_percent_graph": 100 * stages[0]["graph_cycles_outside_kernel_measurements"] / stages[0]["compute_cycles"],
                "decode_percent_graph": 100 * sum(s["graph_cycles_outside_kernel_measurements"] for s in stages[1:]) / decode_graph}}


def csv_text(rows, fields):
    out = io.StringIO()
    writer = csv.DictWriter(out, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clock-hz", type=int, required=True,
                   help="configured FPGA clock, explicitly supplied; not inferred from wall time")
    args = p.parse_args()
    verification = json.loads((args.archive / "verification.json").read_text())
    require(verification["status"] == "MODEL_RUN_NUMERIC_PASS" and
            verification["profile_status"] == "KERNEL_PROFILE_PASS", "requires numerical AND profile acceptance")
    require(verification["scope"]["layers"] == 28, "requires complete 28-layer model")
    consumed = {}

    def read(relative):
        path = args.archive / relative
        digest = sha(path)
        require(verification["archive_sha256"].get(relative) == digest, "archive hash mismatch: " + relative)
        consumed[relative] = digest
        return json.loads(path.read_text())

    profile = read("kernel-profile-verification.json")
    manifest = read("build/model-lib/archive.json")
    configuration = read("image/kernel-profile.json")
    plan = read("image/w8a8-image-plan.json")
    reference = read("references/quant-reference.json")
    require(reference["capacity"] == 128 and plan["kv_bytes"] == 28 * 2 * 8 * 128 * 128 * 4,
            "requires capacity 128 and full 28-layer FP32 K/V cache")
    require(not configuration["progress_uart"] and not configuration.get("phase_probe") and
            verification.get("intermediate_status") == "NOT_INSTRUMENTED",
            "diagnostic work inside graph timing is not supported")
    require(profile["status"] == "KERNEL_PROFILE_PASS" and not profile["errors"], "invalid profile")
    cases = {c["case"]: c["frontend"] for c in manifest["cases"]}
    report = aggregate(profile["stages"], cases, args.clock_hz)
    report.update(run_id=verification["run_id"], status="VERIFIED_PROFILE_SUMMARY",
                  configured_clock_hz=args.clock_hz, layers=28, max_sequence_length=128,
                  prefill_tokens=16, decode_steps=8, evidence_sha256=consumed,
                  archive_verification_sha256=sha(args.archive / "verification.json"),
                  limits=profile["limits"] + [
                      "Per-kernel rows aggregate equal specializations across all layers; per-call numbers are means, not individual measurements.",
                      "Decode averages cover all eight real decode steps; per-stage-kernel.csv retains every measured step.",
                      "Graph remainder is unclassified: copies, views, RoPE/residual/scalar work, graph adapters, allocation, wrapper bookkeeping and synchronization cannot be separated by this profile.",
                      "Kernel cycles include the descriptor/grid adapter and an added completion fence; interpret as instrumented costs, not standalone instruction latency."])
    require(not args.output.exists(), "choose a fresh output to preserve earlier evidence")
    args.output.mkdir(parents=True)
    (args.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["name", "family", "prefill_calls", "decode_calls_per_step", "prefill_milliseconds",
              "decode_mean_milliseconds", "prefill_percent_graph", "decode_percent_graph",
              "prefill_mean_call_milliseconds", "decode_mean_call_milliseconds", "total_cycles", "total_percent_graph"]
    for kind in ("kernels", "families"):
        (args.output / (kind + ".csv")).write_text(csv_text(report[kind], fields))
    detail = report["per_stage_kernel"]
    (args.output / "per-stage-kernel.csv").write_text(csv_text(detail, list(detail[0])))
    lines = ["# 28-layer / max seq 128 / 16-token prefill + 8-step decode", "",
             f"Run: `{report['run_id']}`. Clock: {args.clock_hz} Hz. Numeric and kernel profile checks: PASS.", "",
             "Times below are milliseconds, aggregated across all layers. Decode is the mean of eight real steps.", "",
             "## Operator families", "",
             "| Family | Prefill calls | Decode calls/step | Prefill ms | Decode ms/step | Prefill graph % | Decode graph % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["families"]:
        lines.append(f"| {LABELS.get(r['name'], r['name'])} | {r['prefill_calls']} | {r['decode_calls_per_step']:g} | {r['prefill_milliseconds']:.3f} | {r['decode_mean_milliseconds']:.3f} | {r['prefill_percent_graph']:.2f} | {r['decode_percent_graph']:.2f} |")
    r = report["graph_outside_kernels"]
    lines += [f"| Graph outside measured kernels | — | — | {r['prefill_milliseconds']:.3f} | {r['decode_mean_milliseconds']:.3f} | {r['prefill_percent_graph']:.2f} | {r['decode_percent_graph']:.2f} |", "",
              "## Every kernel specialization", "",
              "| Kernel | Prefill calls | Decode calls/step | Prefill ms | Decode ms/step |",
              "|---|---:|---:|---:|---:|"]
    for r in report["kernels"]:
        lines.append(f"| `{r['name']}` | {r['prefill_calls']} | {r['decode_calls_per_step']:g} | {r['prefill_milliseconds']:.3f} | {r['decode_mean_milliseconds']:.3f} |")
    lines += ["", "## Scope", ""] + ["- " + limit for limit in report["limits"]]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "kernels": len(report["kernels"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
