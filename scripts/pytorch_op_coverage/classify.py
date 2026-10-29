"""Keep source evidence independent from execution results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STAGES = (
    "exported",
    "imported",
    "lowered",
    "compiled",
    "executed",
    "correctness",
)


@dataclass
class OpCoverageRecord:
    operator: str
    families: list[str]
    pytorch_schema: str | None = None
    buddy_op: str | None = None
    frontend_recognized: bool = False
    lowering_dialects: list[str] = field(default_factory=list)
    alias_candidates: list[str] = field(default_factory=list)
    static_status: str = "unmapped"
    known_limitations: str | None = None
    required_cases: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)

    def validated(self) -> bool:
        # Missing cases and known limitations keep the op out of the numerator.
        if not self.required_cases or self.known_limitations:
            return False
        by_id = {case["case_id"]: case for case in self.cases}
        if len(by_id) != len(self.cases):
            return False
        return all(
            case_id in by_id
            and by_id[case_id].get("status") == "passed"
            and all(by_id[case_id].get(s) == "passed" for s in STAGES)
            for case_id in self.required_cases
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "validated_for_profile": self.validated()}


def classify_static(row, ops_map, lowering_by_op, target):
    operator = row["operator"]
    # Buddy's map currently drops the namespace. Preserve it in our identity.
    key = operator.split("::", 1)[1]
    buddy_op = ops_map.get(key)
    dialects = lowering_by_op.get(buddy_op, [])
    aliases = [
        alias
        for alias in target.get("decomp_aliases", {}).get(operator, [])
        if alias.split("::", 1)[1] in ops_map
    ]
    limitation = target.get("known_partial_or_limited", {}).get(operator)
    if buddy_op and dialects:
        status = "registered_lowering"
    elif buddy_op:
        status = "frontend_only"
    elif aliases:
        status = "alias_candidate"
    else:
        status = "unmapped"
    return OpCoverageRecord(
        operator=operator,
        families=row["families"],
        pytorch_schema=target.get("schemas", {}).get(operator),
        buddy_op=buddy_op,
        frontend_recognized=buddy_op is not None,
        lowering_dialects=dialects,
        alias_candidates=aliases,
        static_status=status,
        known_limitations=limitation,
    )


def summarize(records: list[OpCoverageRecord]) -> dict[str, Any]:
    total = len(records)
    counts = {
        "frontend_recognized": sum(r.frontend_recognized for r in records),
        "registered_lowering": sum(bool(r.lowering_dialects) for r in records),
        "alias_candidate": sum(bool(r.alias_candidates) for r in records),
        "unmapped": sum(r.static_status == "unmapped" for r in records),
        "known_limited": sum(bool(r.known_limitations) for r in records),
        "validated_for_profile": sum(r.validated() for r in records),
    }
    cases = [c for r in records for c in r.cases]
    result = {
        "total_ops": total,
        **counts,
        "percentages": {
            k: round(100 * v / total, 2) if total else 0.0
            for k, v in counts.items()
        },
        "case_counts": {
            status: sum(c.get("status") == status for c in cases)
            for status in ("passed", "failed", "skipped", "blocked", "timeout")
        },
        "stage_passed_cases": {
            stage: sum(c.get(stage) == "passed" for c in cases)
            for stage in STAGES
        },
        "required_case_count": sum(len(r.required_cases) for r in records),
        "operators_without_cases": sum(not r.required_cases for r in records),
    }
    moe = [r for r in records if "moe_critical" in r.families]
    if moe and len(moe) != total:
        result["moe_critical"] = summarize(moe)
    return result
