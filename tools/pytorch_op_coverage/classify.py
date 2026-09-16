# ===- classify.py - Coverage status classification -------------------------===
#
# Licensed under the Apache License, Version 2.0 (the "License").
# ===----------------------------------------------------------------------===
"""Classify each target ATen op into coverage buckets for issue #911."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Status = Literal[
    "fully_supported_static",
    "frontend_only",
    "partial",
    "unsupported",
    "live_passed",
    "live_failed",
    "live_skipped",
]


@dataclass
class OpCoverageRecord:
    aten: str
    family: str
    families: list[str] = field(default_factory=list)
    pytorch_schema: str | None = None
    buddy_op: str | None = None
    frontend_recognized: bool = False
    has_buddy_lowering: bool = False
    lowering_dialects: list[str] = field(default_factory=list)
    lowered: str = "not_run"  # yes|no|not_run|error
    compiled: str = "not_run"
    correctness: str = "not_run"
    status: Status = "unsupported"
    known_limitations: str | None = None
    notes: str = ""
    live_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_static(
    aten: str,
    family: str,
    ops_map: dict[str, str],
    lowering_by_op: dict[str, list[str]],
    known_partial: dict[str, str],
    decomp_aliases: dict[str, list[str]] | None = None,
    families: list[str] | None = None,
) -> OpCoverageRecord:
    buddy_op = ops_map.get(aten)
    frontend = buddy_op is not None
    dialects = lowering_by_op.get(buddy_op or "", [])
    has_lowering = bool(dialects)
    limitation = known_partial.get(aten)
    aliases = (decomp_aliases or {}).get(aten, [])
    alias_hits = [a for a in aliases if a in ops_map]

    if not frontend and alias_hits:
        # Count as recognized via documented decomp/alias path, but partial.
        alias = alias_hits[0]
        buddy_op = ops_map[alias]
        dialects = lowering_by_op.get(buddy_op or "", [])
        has_lowering = bool(dialects)
        status: Status = "partial"
        notes = (
            f"No direct _ops_map entry; expected via decomp/alias → `{alias}` "
            f"→ `{buddy_op}`."
        )
        limitation = limitation or f"Alias/decomp of {alias}"
        frontend = True
    elif not frontend:
        status = "unsupported"
        notes = "No DynamoCompiler._ops_map entry."
    elif not has_lowering:
        status = "frontend_only"
        notes = (
            f"Mapped to {buddy_op} but no ops_registry lowering found "
            f"in tosa/linalg/math/func/ttir."
        )
    elif limitation:
        status = "partial"
        notes = "Frontend + lowering present; flagged as limited/partial in target set."
    else:
        status = "fully_supported_static"
        notes = "Frontend map + at least one dialect lowering (static evidence only)."

    return OpCoverageRecord(
        aten=aten,
        family=family,
        families=list(families or [family]),
        pytorch_schema=f"aten::{aten}",
        buddy_op=buddy_op,
        frontend_recognized=frontend,
        has_buddy_lowering=has_lowering,
        lowering_dialects=dialects,
        status=status,
        known_limitations=limitation,
        notes=notes,
    )


def summarize(records: list[OpCoverageRecord]) -> dict[str, Any]:
    total = len(records)
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    frontend_n = sum(1 for r in records if r.frontend_recognized)
    lowering_n = sum(1 for r in records if r.has_buddy_lowering)
    # Static "coverage" used for MVP denominator progress (NOT the final 90% claim).
    static_full = sum(1 for r in records if r.status == "fully_supported_static")
    partial_n = sum(1 for r in records if r.status == "partial")
    unsupported_n = sum(1 for r in records if r.status == "unsupported")
    frontend_only_n = sum(1 for r in records if r.status == "frontend_only")

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2) if total else 0.0

    moe = [r for r in records if "moe_critical" in (r.families or [r.family])]
    moe_total = len(moe)
    moe_static_full = sum(1 for r in moe if r.status == "fully_supported_static")
    moe_partial = sum(1 for r in moe if r.status == "partial")
    moe_unsupported = sum(1 for r in moe if r.status == "unsupported")

    def moe_pct(n: int) -> float:
        return round(100.0 * n / moe_total, 2) if moe_total else 0.0

    return {
        "denominator": "Buddy Target Op Set v0 (unique aten keys)",
        "total_ops": total,
        "frontend_recognized": frontend_n,
        "frontend_recognized_pct": pct(frontend_n),
        "has_buddy_lowering": lowering_n,
        "has_buddy_lowering_pct": pct(lowering_n),
        "fully_supported_static": static_full,
        "fully_supported_static_pct": pct(static_full),
        "partial": partial_n,
        "partial_pct": pct(partial_n),
        "frontend_only": frontend_only_n,
        "unsupported": unsupported_n,
        "unsupported_pct": pct(unsupported_n),
        "by_status": by_status,
        "moe_critical": {
            "total": moe_total,
            "fully_supported_static": moe_static_full,
            "fully_supported_static_pct": moe_pct(moe_static_full),
            "partial": moe_partial,
            "unsupported": moe_unsupported,
        },
        "disclaimer": (
            "fully_supported_static is NOT live compile/correctness coverage. "
            "Do not claim issue #911 90% until live mode measures compiled+correct."
        ),
    }
