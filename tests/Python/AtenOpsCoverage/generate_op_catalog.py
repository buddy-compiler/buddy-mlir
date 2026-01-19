#!/usr/bin/env python3
"""Generate a PyTorch operator coverage catalog and minimal-input plan.

Outputs:
  - JSON: op, overload, has_decomp, devices, dtypes, notes, schema
  - notes mark special traits (inplace/view/meta/random/sparse/quantized/cuda_only)
    and whether minimal inputs can be constructed.

Notes:
  - has_decomp is based on torch._decomp.decomposition_table plus optional extras.
  - Static analysis only; operators are not executed.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch._decomp import decomposition_table as torch_decomposition_table


# Keyword sets for tagging operator traits used by downstream filters.
RANDOM_KEYWORDS = {
    "random",
    "rand",
    "bernoulli",
    "normal",
    "poisson",
    "multinomial",
    "dropout",
    "rrelu",
}
SPARSE_KEYWORDS = {"sparse", "csr", "csc", "coo"}
QUANT_KEYWORDS = {"quant", "q_scale", "qscheme", "fake_quant", "int8"}
CUDA_KEYWORDS = {"cuda", "cudnn", "cublas"}
VIEW_KEYWORDS = {
    "view",
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
    "narrow",
    "slice",
    "select",
    "permute",
    "transpose",
    "t",
    "alias",
    "expand",
    "broadcast",
}


@dataclass
class OperatorRow:
    op: str
    overload: str
    has_decomp: bool
    devices: str
    dtypes: str
    notes: str
    schema: str


def load_extra_decompositions(extra_paths: Sequence[str]) -> Dict:
    """Load custom decomposition tables from modules or module:attr paths."""
    merged: Dict = {}
    for path in extra_paths:
        module_path, attr = (path.split(":", 1) + ["decompositions"])[:2]
        module = importlib.import_module(module_path)
        table = getattr(module, attr, None)
        if table is None:
            raise ValueError(f"Missing {attr} in {module_path}")
        merged.update(table)
    return merged


def resolve_ops_namespace(namespace: str):
    if not hasattr(torch.ops, namespace):
        raise ValueError(f"Unknown torch.ops namespace: {namespace}")
    return getattr(torch.ops, namespace)


def iter_ops(
    namespace: str,
) -> Iterable[Tuple[str, str, torch._ops.OpOverload]]:
    """Iterate all ops and overloads under torch.ops.<namespace>."""
    ops_root = resolve_ops_namespace(namespace)
    for op_name in sorted(dir(ops_root)):
        if op_name.startswith("_"):
            continue
        try:
            packet = getattr(ops_root, op_name)
            overloads = packet.overloads()
            if overloads:
                for overload in overloads:
                    overload_obj = getattr(packet, overload)
                    yield op_name, overload, overload_obj
            else:
                overload_obj = packet.default
                yield op_name, "default", overload_obj
        except Exception:
            continue


def infer_special_tags(
    namespace: str, op_name: str, overload: str, schema_str: str
) -> Set[str]:
    """Tag op traits based on name and schema patterns."""
    full_name = f"{namespace}.{op_name}.{overload}"
    lowered = full_name.lower()
    tags: Set[str] = set()

    if op_name.endswith("_") or overload == "inplace":
        tags.add("inplace")
    if any(k in lowered for k in VIEW_KEYWORDS):
        tags.add("view_like")
    if any(k in lowered for k in RANDOM_KEYWORDS):
        tags.add("random")
    if any(k in lowered for k in SPARSE_KEYWORDS) or (
        "Tensor?[]" in schema_str and "Sparse" in schema_str
    ):
        tags.add("sparse")
    if any(k in lowered for k in QUANT_KEYWORDS):
        tags.add("quantized")
    if any(k in lowered for k in CUDA_KEYWORDS):
        tags.add("cuda_only")
    if "meta" in lowered:
        tags.add("meta")
    if namespace in {"prim", "prims"} or "prim" in lowered:
        tags.add("prim")
    return tags


def infer_devices(tags: Set[str]) -> str:
    if "cuda_only" in tags:
        return "cuda"
    return "unspecified"


def infer_dtypes(schema_str: str) -> str:
    if "ScalarType" in schema_str or "dtype" in schema_str:
        return "by-arg"
    if "Tensor" in schema_str:
        return "tensor"
    return "unknown"


def guess_arg_value(type_str: str):
    """Generate a minimal placeholder value from a type string."""
    t = type_str.replace(" ", "").lower()
    try:
        if "tensor" in t:
            return torch.ones(1, dtype=torch.float32)
        if t.startswith("int") or t.startswith("symint"):
            return 1
        if t.startswith("float") or t.startswith("double"):
            return 1.0
        if t.startswith("bool"):
            return True
        if "scalar" in t:
            return 1.0
        if "device" in t:
            return torch.device("cpu")
        if "memoryformat" in t:
            return torch.contiguous_format
        if "layout" in t:
            return torch.strided
    except Exception:
        return None
    return None


def build_minimal_inputs(schema: torch._C.FunctionSchema) -> Tuple[bool, str]:
    """Check whether minimal inputs can be constructed for a schema."""
    failures: List[str] = []
    for arg in schema.arguments:
        type_str = str(arg.type)
        if arg.default_value is not None:
            continue
        val = guess_arg_value(type_str)
        if val is None:
            failures.append(arg.name)
    if failures:
        return False, f"input_missing:{'|'.join(failures)}"
    return True, "input_ok"


def _normalize_namespaces(raw: Sequence[str]) -> List[str]:
    namespaces: List[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                namespaces.append(part)
    return namespaces


def collect_rows(
    namespaces: Sequence[str],
    extra_decompositions: Dict,
    limit: Optional[int] = None,
) -> List[OperatorRow]:
    """Collect coverage rows across namespaces."""
    rows: List[OperatorRow] = []
    combined_decomp = dict(torch_decomposition_table)
    combined_decomp.update(extra_decompositions)

    total = 0
    for namespace in namespaces:
        for op_name, overload, overload_obj in iter_ops(namespace):
            schema = overload_obj._schema  # type: ignore[attr-defined]
            schema_str = str(schema)
            tags = infer_special_tags(namespace, op_name, overload, schema_str)
            devices = infer_devices(tags)
            dtypes = infer_dtypes(schema_str)
            has_decomp = overload_obj in combined_decomp

            input_ok, input_note = build_minimal_inputs(schema)
            if not input_ok:
                tags.add(input_note)
            else:
                tags.add("input_ready")

            notes = ",".join(sorted(tags)) if tags else ""
            rows.append(
                OperatorRow(
                    op=op_name,
                    overload=overload,
                    has_decomp=has_decomp,
                    devices=devices,
                    dtypes=dtypes,
                    notes=notes,
                    schema=schema_str,
                )
            )
            total += 1
            if limit and total >= limit:
                return rows
    return rows


def write_json(rows: List[OperatorRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a torch.ops coverage catalog"
    )
    parser.add_argument(
        "--namespace",
        action="append",
        default=[],
        help="torch.ops namespace (repeatable or comma-separated)",
    )
    parser.add_argument(
        "--extra-decomp",
        action="append",
        default=[],
        help="Custom decomposition table, format module[:attr]",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("tests/Python/AtenOpsCoverage/op_catalog.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of operators (debug only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra = (
        load_extra_decompositions(args.extra_decomp)
        if args.extra_decomp
        else {}
    )
    namespaces = _normalize_namespaces(args.namespace)
    if not namespaces:
        namespaces = ["aten"]
    rows = collect_rows(namespaces, extra, limit=args.limit)
    write_json(rows, args.json)
    print(f"Wrote {len(rows)} entries to {args.json}")


if __name__ == "__main__":
    main()
