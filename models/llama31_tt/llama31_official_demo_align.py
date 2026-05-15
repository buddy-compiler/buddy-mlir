#!/usr/bin/env python3
# ===- llama31_official_demo_align.py -----------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Helpers for aligning Buddy Llama runs with the official tt_transformers demo.
#
# ===---------------------------------------------------------------------------

"""Helpers for aligning Buddy Llama runs with the official tt_transformers demo.

This file deliberately does not run HF as a golden model. It reads the same
reference files and baseline logs consumed by Tenstorrent's
``models/tt_transformers/demo/simple_text_demo.py`` so Buddy can replay the
official demo protocol directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
TT_METAL_HOME = Path(
    os.environ.get(
        "TT_METAL_HOME",
        _REPO_ROOT / "thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal",
    )
)
OFFICIAL_BASELINE_DIR = Path(
    os.environ.get(
        "TT_OFFICIAL_DEMO_LOG_DIR", _THIS_DIR / "official_demo_artifacts"
    )
)
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"
DEFAULT_REFPT = (
    TT_METAL_HOME
    / "models/tt_transformers/tests/reference_outputs"
    / f"{DEFAULT_MODEL_NAME}.refpt"
)
DEFAULT_PERF_MD = TT_METAL_HOME / "models/tt_transformers/PERF.md"
DEFAULT_SUMMARY_CSV = OFFICIAL_BASELINE_DIR / "logs/summary.csv"
DEFAULT_OUT = _THIS_DIR / "official_demo_artifacts"
DEFAULT_TRACE = OFFICIAL_BASELINE_DIR / "logs/ci_token_matching_trace.json"


def _read_perf_row(perf_md: Path) -> dict[str, str] | None:
    in_performance = False
    for raw in perf_md.read_text().splitlines():
        line = raw.strip()
        if line == "## Performance":
            in_performance = True
            continue
        if in_performance and line.startswith("## "):
            break
        if not in_performance or not line.startswith("| Llama-3.1-8B"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) >= 6:
            return {
                "model": cols[0],
                "device": cols[1],
                "top1_pct": cols[2],
                "top5_pct": cols[3],
                "speed_t_s_u": cols[4],
                "ttft_ms": cols[5],
            }
    return None


def _read_latest_rows(
    summary_csv: Path, *, require_passed: bool
) -> dict[str, dict[str, str]]:
    wanted = {
        "batch-1": "batch-1",
        "ci-token-matching": "ci-token-matching",
        "ci-token-matching-Instruct": "ci-token-matching",
    }
    latest: dict[str, dict[str, str]] = {}
    if not summary_csv.exists():
        return latest
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if require_passed and row.get("passed") != "1":
                continue
            case = row.get("case", "")
            key = wanted.get(case)
            if key is None:
                continue
            latest[key] = row
    return latest


def cmd_baseline(args: argparse.Namespace) -> None:
    perf_row = _read_perf_row(args.perf_md)
    latest_passed = _read_latest_rows(args.summary_csv, require_passed=True)
    latest_any = _read_latest_rows(args.summary_csv, require_passed=False)
    payload: dict[str, Any] = {
        "official_perf_md_tenstorrent": perf_row,
        "local_reproduced_official_demo_passed": latest_passed,
        "local_reproduced_official_demo_latest": latest_any,
        "paths": {
            "perf_md": str(args.perf_md),
            "summary_csv": str(args.summary_csv),
            "official_demo": str(
                TT_METAL_HOME
                / "models/tt_transformers/demo/simple_text_demo.py"
            ),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_refpt(refpt: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = torch.load(refpt, map_location="cpu")
    reference_tokens = data["reference_tokens"][0].to(torch.int64).cpu()
    split_point = reference_tokens.shape[-1] // 2
    prompt_tokens = reference_tokens[:split_point]
    forced_reference_tokens = reference_tokens[split_point:]
    top5_tokens = (
        data["top5_tokens"][split_point - 1 :, :].to(torch.int64).cpu()
    )
    if top5_tokens.shape[0] != forced_reference_tokens.shape[0]:
        raise RuntimeError(
            "official reference split mismatch: "
            f"top5={tuple(top5_tokens.shape)}, "
            f"forced={tuple(forced_reference_tokens.shape)}"
        )
    return (
        prompt_tokens.numpy(),
        forced_reference_tokens.numpy(),
        top5_tokens.numpy(),
    )


def cmd_export_reference(args: argparse.Namespace) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens, forced_tokens, top5_tokens = _load_refpt(args.refpt)
    npz_path = out_dir / "ci_token_matching_reference.npz"
    json_path = out_dir / "ci_token_matching_reference_summary.json"
    np.savez(
        npz_path,
        prompt_tokens=prompt_tokens,
        forced_reference_tokens=forced_tokens,
        official_top5_tokens=top5_tokens,
    )
    summary = {
        "source_refpt": str(args.refpt),
        "model_name": DEFAULT_MODEL_NAME,
        "prompt_length": int(prompt_tokens.shape[0]),
        "forced_decode_length": int(forced_tokens.shape[0]),
        "top5_shape": list(top5_tokens.shape),
        "first_prompt_tokens": prompt_tokens[:16].tolist(),
        "first_forced_tokens": forced_tokens[:16].tolist(),
        "first_top5_rows": top5_tokens[:4].tolist(),
        "npz": str(npz_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _accuracy_from_predictions(
    predicted_tokens: np.ndarray, top5_tokens: np.ndarray
) -> tuple[float, float]:
    n = min(predicted_tokens.shape[0], top5_tokens.shape[0])
    predicted_tokens = predicted_tokens[:n]
    top5_tokens = top5_tokens[:n]
    top1 = predicted_tokens == top5_tokens[:, 0]
    top5 = (predicted_tokens[:, None] == top5_tokens).any(axis=1)
    return float(top1.mean() * 100.0), float(top5.mean() * 100.0)


def cmd_summarize_trace(args: argparse.Namespace) -> None:
    trace = json.loads(args.trace.read_text())
    prompt_tokens, forced_tokens, top5_tokens = _load_refpt(args.refpt)
    predicted_tokens = np.asarray(
        trace["accuracy_predicted_tokens"], dtype=np.int64
    )
    decoded_input_tokens = np.asarray(
        [step["input_tokens"][0] for step in trace["decode_steps"]],
        dtype=np.int64,
    )
    recomputed_top1, recomputed_top5 = _accuracy_from_predictions(
        predicted_tokens, top5_tokens
    )
    n = int(min(predicted_tokens.shape[0], forced_tokens.shape[0]))
    # TokenAccuracy.collect_predicted_tokens records the current prediction,
    # then replaces it with reference_tokens[gt_pos] before decode_forward.
    forced_input_expected = forced_tokens[:n]
    forced_input_match = bool(
        decoded_input_tokens[: forced_input_expected.shape[0]].shape
        == forced_input_expected.shape
        and np.array_equal(
            decoded_input_tokens[: forced_input_expected.shape[0]],
            forced_input_expected,
        )
    )
    payload = {
        "trace": str(args.trace),
        "model_name": trace.get("model_name"),
        "test_id": trace.get("test_id"),
        "prompt_length": len(trace["prompt_tokens"][0]),
        "reference_prompt_length": int(prompt_tokens.shape[0]),
        "decode_steps": len(trace["decode_steps"]),
        "predicted_tokens": int(predicted_tokens.shape[0]),
        "forced_input_match": forced_input_match,
        "trace_top1_pct": trace.get("top1_pct"),
        "trace_top5_pct": trace.get("top5_pct"),
        "recomputed_top1_pct": recomputed_top1,
        "recomputed_top5_pct": recomputed_top5,
        "first_prefill_tokens": trace["prefill_tokens"][:8],
        "first_predicted_tokens": predicted_tokens[:16].tolist(),
        "first_forced_reference_tokens": forced_tokens[:16].tolist(),
        "first_official_top5_rows": top5_tokens[:4].tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Official tt_transformers demo alignment helper."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser(
        "baseline", help="Print official PERF.md and local run summary."
    )
    p_base.add_argument("--perf-md", type=Path, default=DEFAULT_PERF_MD)
    p_base.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p_base.set_defaults(func=cmd_baseline)

    p_ref = sub.add_parser(
        "export-reference",
        help="Export the official ci-token-matching reference token split.",
    )
    p_ref.add_argument("--refpt", type=Path, default=DEFAULT_REFPT)
    p_ref.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_ref.set_defaults(func=cmd_export_reference)

    p_trace = sub.add_parser(
        "summarize-trace",
        help="Summarize and validate an official ci-token-matching token trace.",
    )
    p_trace.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    p_trace.add_argument("--refpt", type=Path, default=DEFAULT_REFPT)
    p_trace.set_defaults(func=cmd_summarize_trace)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
