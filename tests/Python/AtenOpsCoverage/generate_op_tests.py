#!/usr/bin/env python3
"""Generate batched test files from an operator catalog.

Each generated test file declares an explicit OPS list. Remove a name to skip
that operator in the batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TEMPLATE = """# RUN: %PYTHON %s 2>&1 | FileCheck %s
from {runner_module} import {runner_func}

# Edit OPS to add or remove operators (format: "op.overload" by default).
OPS = [
{ops}
]

if __name__ == "__main__":
    {runner_func}(OPS, batch_label="{label}", max_fails={max_fails})
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
"""


def chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_namespaces(raw: Sequence[str]) -> List[str]:
    namespaces: List[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                namespaces.append(part)
    return namespaces


def _parse_runner(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise ValueError("Runner must be in module:func format")
    module, func = value.split(":", 1)
    module = module.strip()
    func = func.strip()
    if not module or not func:
        raise ValueError("Runner must include both module and function")
    return module, func


def _format_op_name(
    entry: Dict[str, str], name_format: str, namespace: str
) -> str:
    try:
        return name_format.format(
            namespace=entry.get("namespace", namespace),
            op=entry["op"],
            overload=entry["overload"],
        )
    except KeyError as exc:
        raise ValueError(f"Missing field in name format: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate batched operator tests"
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("tests/Python/AtenOpsCoverage/op_catalog.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/Python/AtenOpsCoverage"),
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-fails", type=int, default=20)
    parser.add_argument(
        "--prefix",
        type=str,
        default="test_{namespace}_op_batch",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting batch index",
    )
    parser.add_argument(
        "--namespace",
        action="append",
        default=[],
        help="Namespace filter (repeatable or comma-separated)",
    )
    parser.add_argument(
        "--name-format",
        type=str,
        default="{op}.{overload}",
        help="Format for OPS entries; fields: namespace, op, overload",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="aten_op_batch_runner:run_aten_op_batch",
        help="Runner import in module:func form",
    )
    args = parser.parse_args()

    with args.coverage.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    namespaces = _normalize_namespaces(args.namespace)
    default_namespace = namespaces[0] if len(namespaces) == 1 else None

    def resolve_entry_namespace(entry: Dict[str, str]) -> str:
        if "namespace" in entry:
            return entry["namespace"]
        if default_namespace:
            return default_namespace
        if namespaces:
            raise ValueError(
                "Coverage entries missing 'namespace'; pass a single --namespace"
            )
        return "aten"

    if namespaces:
        entries = [
            e for e in entries if resolve_entry_namespace(e) in namespaces
        ]

    if not entries:
        raise ValueError("No entries matched the requested filters")

    by_namespace: Dict[str, List[Dict[str, str]]] = {}
    for entry in entries:
        by_namespace.setdefault(resolve_entry_namespace(entry), []).append(
            entry
        )

    runner_module, runner_func = _parse_runner(args.runner)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for namespace, ns_entries in sorted(by_namespace.items()):
        if len(by_namespace) > 1 and "{namespace}" not in args.prefix:
            raise ValueError(
                "prefix must include {namespace} when generating multiple namespaces"
            )
        prefix = (
            args.prefix.format(namespace=namespace)
            if "{namespace}" in args.prefix
            else args.prefix
        )
        if not prefix.endswith("_"):
            prefix = f"{prefix}_"

        names = [
            _format_op_name(entry, args.name_format, namespace)
            for entry in ns_entries
        ]
        for idx, batch in enumerate(
            chunk(names, args.batch_size), start=args.start_index
        ):
            ops_body = "\n".join(f'    "{name}",' for name in batch)
            content = TEMPLATE.format(
                ops=ops_body,
                label=f"{prefix}{idx}",
                max_fails=args.max_fails,
                runner_module=runner_module,
                runner_func=runner_func,
            )
            path = output_dir / f"{prefix}{idx:03d}.py"
            path.write_text(content, encoding="utf-8")
            print(f"Wrote {path} with {len(batch)} operators")


if __name__ == "__main__":
    main()
