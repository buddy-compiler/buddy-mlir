"""Emit JSON and Markdown coverage reports."""

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
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "issue": "https://github.com/buddy-compiler/buddy-mlir/issues/911",
        "mode": mode,
        "repo_rev": repo_rev,
        "target_set": target_set_path,
        "summary": summarize(records),
        "operators": [r.to_dict() for r in records],
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Buddy-MLIR PyTorch Operator Coverage Report",
        "",
        f"- Generated (UTC): `{payload['generated_at']}`",
        f"- Issue: {payload['issue']}",
        f"- Mode: `{payload['mode']}`",
        f"- Repo rev: `{payload.get('repo_rev')}`",
        f"- Target set: `{payload['target_set']}`",
        "",
        "## Summary",
        "",
        f"- Denominator: **{summary['denominator']}**",
        f"- Total ops: **{summary['total_ops']}**",
        (
            f"- Frontend recognized: **{summary['frontend_recognized']}** "
            f"({summary['frontend_recognized_pct']}%)"
        ),
        (
            f"- Has Buddy lowering: **{summary['has_buddy_lowering']}** "
            f"({summary['has_buddy_lowering_pct']}%)"
        ),
        (
            f"- Fully supported (static): **{summary['fully_supported_static']}** "
            f"({summary['fully_supported_static_pct']}%)"
        ),
        (
            f"- Partial / limited: **{summary['partial']}** "
            f"({summary['partial_pct']}%)"
        ),
        f"- Frontend only (no lowering): **{summary['frontend_only']}**",
        (
            f"- Unsupported: **{summary['unsupported']}** "
            f"({summary['unsupported_pct']}%)"
        ),
        "",
        f"> {summary['disclaimer']}",
        "",
    ]

    moe = summary["moe_critical"]
    lines.extend(
        [
            "## MoE-critical subset",
            "",
            f"- Total: **{moe['total']}**",
            (
                f"- Fully supported (static): **{moe['fully_supported_static']}** "
                f"({moe['fully_supported_static_pct']}%)"
            ),
            f"- Partial: **{moe['partial']}**",
            f"- Unsupported: **{moe['unsupported']}**",
            "",
            "## Unsupported operators",
            "",
        ]
    )

    unsupported = [r for r in payload["operators"] if r["status"] == "unsupported"]
    if not unsupported:
        lines.append("_None in this target set._")
    else:
        lines.extend(["| ATen | Family | Notes |", "| --- | --- | --- |"])
        for record in unsupported:
            notes = (record.get("notes") or "").replace("|", "\\|")
            lines.append(
                f"| `{record['aten']}` | {record['family']} | {notes} |"
            )

    lines.extend(["", "## Partial / limited operators", ""])
    partial = [r for r in payload["operators"] if r["status"] == "partial"]
    if not partial:
        lines.append("_None flagged._")
    else:
        lines.extend(
            [
                "| ATen | Buddy op | Dialects | Limitation |",
                "| --- | --- | --- | --- |",
            ]
        )
        for record in partial:
            lim = (record.get("known_limitations") or "").replace("|", "\\|")
            dialects = ", ".join(record.get("lowering_dialects") or [])
            lines.append(
                f"| `{record['aten']}` | `{record.get('buddy_op')}` | "
                f"{dialects} | {lim} |"
            )

    lines.extend(["", "## Frontend-only (mapped, no lowering found)", ""])
    frontend_only = [
        r for r in payload["operators"] if r["status"] == "frontend_only"
    ]
    if not frontend_only:
        lines.append("_None._")
    else:
        lines.extend(["| ATen | Buddy op |", "| --- | --- |"])
        for record in frontend_only:
            lines.append(f"| `{record['aten']}` | `{record.get('buddy_op')}` |")

    lines.extend(
        [
            "",
            "## Follow-ups",
            "",
            "1. Enable live compile and numerical checks in CI.",
            "2. Expand the target set from MoE / Transformer workload traces.",
            "3. Add regression tests for high-priority unsupported MoE ops.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
