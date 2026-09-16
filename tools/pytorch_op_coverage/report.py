# ===- report.py - JSON + Markdown coverage report emitters ----------------===
#
# Licensed under the Apache License, Version 2.0 (the "License").
# ===----------------------------------------------------------------------===
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classify import OpCoverageRecord, summarize


def build_report_payload(
    records: list[OpCoverageRecord],
    *,
    mode: str,
    repo_rev: str | None,
    target_set_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize(records)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "issue": "https://github.com/buddy-compiler/buddy-mlir/issues/911",
        "mode": mode,
        "repo_rev": repo_rev,
        "target_set": target_set_path,
        "summary": summary,
        "operators": [r.to_dict() for r in records],
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = payload["summary"]
    lines: list[str] = []
    lines.append("# Buddy-MLIR PyTorch Operator Coverage Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{payload['generated_at']}`")
    lines.append(f"- Issue: {payload['issue']}")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Repo rev: `{payload.get('repo_rev')}`")
    lines.append(f"- Target set: `{payload['target_set']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Denominator: **{s['denominator']}**")
    lines.append(f"- Total ops: **{s['total_ops']}**")
    lines.append(
        f"- Frontend recognized: **{s['frontend_recognized']}** "
        f"({s['frontend_recognized_pct']}%)"
    )
    lines.append(
        f"- Has Buddy lowering: **{s['has_buddy_lowering']}** "
        f"({s['has_buddy_lowering_pct']}%)"
    )
    lines.append(
        f"- Fully supported (static): **{s['fully_supported_static']}** "
        f"({s['fully_supported_static_pct']}%)"
    )
    lines.append(f"- Partial / limited: **{s['partial']}** ({s['partial_pct']}%)")
    lines.append(f"- Frontend only (no lowering): **{s['frontend_only']}**")
    lines.append(f"- Unsupported: **{s['unsupported']}** ({s['unsupported_pct']}%)")
    lines.append("")
    lines.append(f"> {s['disclaimer']}")
    lines.append("")
    moe = s["moe_critical"]
    lines.append("## MoE-critical subset")
    lines.append("")
    lines.append(f"- Total: **{moe['total']}**")
    lines.append(
        f"- Fully supported (static): **{moe['fully_supported_static']}** "
        f"({moe['fully_supported_static_pct']}%)"
    )
    lines.append(f"- Partial: **{moe['partial']}**")
    lines.append(f"- Unsupported: **{moe['unsupported']}**")
    lines.append("")
    lines.append("## Unsupported operators")
    lines.append("")
    unsupported = [r for r in payload["operators"] if r["status"] == "unsupported"]
    if not unsupported:
        lines.append("_None in this target set._")
    else:
        lines.append("| ATen | Family | Notes |")
        lines.append("| --- | --- | --- |")
        for r in unsupported:
            notes = (r.get("notes") or "").replace("|", "\\|")
            lines.append(f"| `{r['aten']}` | {r['family']} | {notes} |")
    lines.append("")
    lines.append("## Partial / limited operators")
    lines.append("")
    partial = [r for r in payload["operators"] if r["status"] == "partial"]
    if not partial:
        lines.append("_None flagged._")
    else:
        lines.append("| ATen | Buddy op | Dialects | Limitation |")
        lines.append("| --- | --- | --- | --- |")
        for r in partial:
            lim = (r.get("known_limitations") or "").replace("|", "\\|")
            dialects = ", ".join(r.get("lowering_dialects") or [])
            lines.append(
                f"| `{r['aten']}` | `{r.get('buddy_op')}` | {dialects} | {lim} |"
            )
    lines.append("")
    lines.append("## Frontend-only (mapped, no lowering found)")
    lines.append("")
    frontend_only = [r for r in payload["operators"] if r["status"] == "frontend_only"]
    if not frontend_only:
        lines.append("_None._")
    else:
        lines.append("| ATen | Buddy op |")
        lines.append("| --- | --- |")
        for r in frontend_only:
            lines.append(f"| `{r['aten']}` | `{r.get('buddy_op')}` |")
    lines.append("")
    lines.append("## High-priority follow-ups")
    lines.append("")
    lines.append("1. Enable live mode (import→lower→compile→correctness) on Linux/WSL CI.")
    lines.append("2. Trace real MoE workloads to expand the denominator beyond v0 seed.")
    lines.append("3. Prioritize unsupported MoE ops that block expert dispatch/combine.")
    lines.append("4. Add regression microtests for each newly supported MoE op.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
