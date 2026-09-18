"""JSON and Markdown share the same evidence, failures and denominator."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from classify import STAGES, summarize
from probes import PROFILES


def workload_gaps(workloads, records):
    by_name = {r.operator: r for r in records}
    gaps = {}
    for workload in workloads:
        for case in workload["cases"]:
            for op, count in case.get("observed_ops", {}).items():
                entry = gaps.setdefault(
                    op,
                    {
                        "operator": op,
                        "workloads": {},
                        "in_target": op in by_name,
                    },
                )
                entry["workloads"][workload["name"]] = max(
                    count, entry["workloads"].get(workload["name"], 0)
                )
    for name, entry in gaps.items():
        record = by_name.get(name)
        entry["static_status"] = (
            record.static_status if record else "outside_target"
        )
        entry["validated_for_profile"] = record.validated() if record else False
    return sorted(
        gaps.values(),
        key=lambda x: (
            "moe_block" not in x["workloads"],
            -sum(x["workloads"].values()),
            x["operator"],
        ),
    )


def build_report_payload(
    records,
    *,
    target,
    mode,
    provenance,
    environment,
    workloads,
    exit_code,
    threshold=None,
):
    return {
        "report_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "run_status": "blocked"
        if exit_code == 2
        else "failed"
        if exit_code
        else "completed",
        "exit_code": exit_code,
        "minimum_coverage": threshold,
        "target": {
            k: target.get(k)
            for k in (
                "name",
                "version",
                "denominator_rule",
                "schema_source",
                "migration_from_v0",
            )
        },
        "provenance": provenance,
        "environment": environment,
        "profile": {
            "name": "cpu-export-v1",
            "cases": PROFILES,
            "seed": 0,
            "rtol": 1e-4,
            "atol": 1e-5,
            "external_calls": False,
            "path": "strict torch.export -> Buddy _compile_fx -> tosa-priority -> JIT",
            "scope": "Explicit small contiguous CPU inputs; no claim for arbitrary shapes/dtypes or full models",
        },
        "summary": summarize(records),
        "operators": [r.to_dict() for r in records],
        "workloads": workloads,
        "workload_operator_inventory": workload_gaps(workloads, records),
    }


def write_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def cell(value):
    return (
        str(value if value is not None else "—")
        .replace("|", "\\|")
        .replace("\n", "<br>")
        .replace("\r", "")
    )


def write_markdown(payload, path: Path):
    summary, target = payload["summary"], payload["target"]
    provenance = payload["provenance"]
    measured = payload["mode"] == "live"
    lines = [
        "# PyTorch operator coverage",
        "",
        f"- Mode: **{payload['mode']}**; run: **{payload['run_status']}**; exit: **{payload['exit_code']}**",
        f"- Generated (UTC): `{payload['generated_at']}`",
        f"- Target: **{target['name']}** / `{target['version']}`; **{summary['total_ops']}** unique operators",
        f"- Source: `{provenance['git_revision']}`; dirty: `{provenance['dirty']}`",
        f"- Source SHA-256: `{provenance['source_sha256']}`",
        f"- Profile: `{payload['profile']['name']}`; seed 0; rtol 1e-4; atol 1e-5; external calls disabled",
        "",
        "> Registration and export are not compile/correctness evidence. Untested, skipped, failed and limited operators stay in the denominator.",
        "",
    ]
    if target.get("migration_from_v0"):
        lines += [
            "> v1 has 106 entries versus v0's 108: two Buddy cache helpers were removed; Prim namespace and softmax overload were corrected. Percentages are not directly comparable.",
            "",
        ]
    if provenance.get("changed_during_run"):
        lines += [
            "**Source changed during this run; regenerate before using these results.**",
            "",
        ]
    if payload["environment"].get("error"):
        lines += [
            f"Stack diagnostic: {cell(payload['environment']['error'])}",
            "",
        ]
    env = payload["environment"]
    lines += [
        f"Python: `{provenance['python']}`; measured torch: `{env.get('torch', 'not loaded')}`; schema snapshot torch: `{(target.get('schema_source') or {}).get('torch_version', 'unknown')}`.",
        "",
    ]
    lines += [
        "## Coverage",
        "",
        "| Evidence | Count | % of fixed denominator |",
        "| --- | ---: | ---: |",
    ]
    for key in (
        "frontend_recognized",
        "registered_lowering",
        "alias_candidate",
        "unmapped",
        "known_limited",
        "validated_for_profile",
    ):
        lines.append(
            f"| {key} | {summary[key]} | {summary['percentages'][key]}% |"
        )
    lines += [
        "",
        f"Live validation: **{'requested' if measured else 'not measured'}**. Confirmed end-to-end numerator: **{summary['validated_for_profile']}**.",
        f"Operators without an input contract: **{summary['operators_without_cases']}**.",
        "A completed run is not the 90% gate; use `--mode live --min-coverage 90` for that gate.",
        "",
    ]
    if summary.get("moe_critical"):
        moe = summary["moe_critical"]
        lines += [
            f"MoE: **{moe['validated_for_profile']}/{moe['total_ops']}** validated for profile ({moe['percentages']['validated_for_profile']}%); **{moe['known_limited']}** known limited.",
            "",
        ]
    lines += [
        "## Execution evidence",
        "",
        "| Stage | Passed cases |",
        "| --- | ---: |",
    ]
    for stage, count in summary["stage_passed_cases"].items():
        lines.append(f"| {stage} | {count} |")
    lines += [
        "",
        f"Case outcomes: `{json.dumps(summary['case_counts'], sort_keys=True)}`",
        "",
        "## Operator details",
        "",
        "| Operator | Source evidence | Required/passed cases | Validated | Limitation / alias candidates |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for r in payload["operators"]:
        passed = sum(c.get("correctness") == "passed" for c in r["cases"])
        notes = r["known_limitations"] or ", ".join(r["alias_candidates"])
        lines.append(
            f"| `{r['operator']}` | {r['static_status']} | {len(r['required_cases'])}/{passed} | {r['validated_for_profile']} | {cell(notes)} |"
        )
    problems = [
        (r["operator"], c)
        for r in payload["operators"]
        for c in r["cases"]
        if c.get("status") != "passed"
    ]
    if problems:
        lines += [
            "",
            "## Failed, blocked and untested cases",
            "",
            "| Operator | Case | Status | Stage | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, c in problems:
            lines.append(
                f"| `{name}` | {c['case_id']} | {c['status']} | {cell(c.get('active_stage'))} | {cell(c.get('reason'))} |"
            )
    if env.get("schema_errors"):
        lines += ["", "## Schema lookup errors", ""]
        lines += [
            f"- `{op}`: {cell(reason)}"
            for op, reason in env["schema_errors"].items()
        ]
    if payload["workloads"]:
        lines += [
            "",
            "## Representative block workloads",
            "",
            "These are small fixed-shape blocks, not full-model acceptance tests. Trace success only validates eager execution and export.",
            "",
            "| Workload | Case | Status | Stage results | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for w in payload["workloads"]:
            for c in w["cases"]:
                stages = ", ".join(f"{s}={c.get(s, 'not_run')}" for s in STAGES)
                lines.append(
                    f"| {w['name']} | {c['case_id']} | {c['status']} | {stages} | {cell(c.get('reason'))} |"
                )
    if payload["workload_operator_inventory"]:
        lines += [
            "",
            "## Observed workload operators and gaps",
            "",
            "Counts are maximum FX node counts per workload across profiles, not runtime invocation frequencies. MoE entries are listed first. Outside-target operators do not change the denominator.",
            "",
            "| Operator | Workload: FX nodes | In target | Source evidence |",
            "| --- | --- | --- | --- |",
        ]
        for entry in payload["workload_operator_inventory"]:
            counts = ", ".join(
                f"{k}: {v}" for k, v in entry["workloads"].items()
            )
            lines.append(
                f"| `{entry['operator']}` | {counts} | {entry['in_target']} | {entry['static_status']} |"
            )
    lines += [
        "",
        "## Remaining work",
        "",
        "- Validate all configured cases on a built Buddy CPU runtime; retain native failures and timeouts.",
        "- Add input contracts for untested operators; expand shapes, dtypes and attributes with regression tests.",
        "- Prioritize MoE workload blockers; review trace-derived additions as a new target-set version.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
