"""Parse DynamoCompiler._ops_map and dialect ops_registry tables from source.

Static mode only reads the frontend tree; it does not import buddy or torch.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def default_frontend_py(repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root_from_here()
    return root / "frontend" / "Python" / "frontend.py"


def default_ops_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root_from_here()
    return root / "frontend" / "Python" / "ops"


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def parse_ops_map(frontend_py: Path) -> dict[str, str]:
    """Return aten_symbol -> BuddyOpClassName from DynamoCompiler._ops_map."""
    tree = ast.parse(
        frontend_py.read_text(encoding="utf-8"), filename=str(frontend_py)
    )
    ops_map: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_ops_map"
            ):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if key is None:
                    continue
                aten = _literal_str(key)
                buddy_op = _name_of(value)
                if aten is not None and buddy_op is not None:
                    ops_map[aten] = buddy_op

    if not ops_map:
        raise RuntimeError(f"Failed to parse _ops_map from {frontend_py}")
    return ops_map


def parse_ops_registry_file(path: Path) -> dict[str, str]:
    """Parse `ops_registry` / `llm_ops_registry` dicts from a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in ("ops_registry", "llm_ops_registry"):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key, val in zip(node.value.keys, node.value.values):
                if key is None:
                    continue
                op_name = _literal_str(key)
                if op_name is None:
                    continue
                out[op_name] = _name_of(val) or "<expr>"
    return out


def parse_all_registries(ops_dir: Path) -> dict[str, dict[str, str]]:
    """Return dialect -> {BuddyOpClassName -> lowering_fn}."""
    dialects: dict[str, dict[str, str]] = {}
    for name in (
        "tosa.py",
        "linalg.py",
        "math.py",
        "func.py",
        "ttir.py",
        "ttir_llm.py",
    ):
        path = ops_dir / name
        if not path.exists():
            continue
        registry = parse_ops_registry_file(path)
        if registry:
            dialects[path.stem] = registry
    return dialects


def merge_registry_keys(
    dialects: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """BuddyOpClassName -> dialects that register a lowering."""
    merged: dict[str, list[str]] = {}
    for dialect, registry in dialects.items():
        for op_name in registry:
            merged.setdefault(op_name, []).append(dialect)
    return merged


def load_target_op_set(path: Path) -> dict[str, Any]:
    target = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(target, dict) or not all(
        target.get(key) for key in ("name", "version", "families")
    ):
        raise ValueError("Target set needs name, version and nonempty families")
    if not isinstance(target["families"], dict):
        raise ValueError("families must be an object")
    for meta in target["families"].values():
        if not isinstance(meta, dict) or not isinstance(meta.get("ops"), list):
            raise ValueError("Each family needs an ops list")
        if not all(isinstance(op, str) for op in meta["ops"]):
            raise ValueError("Operator identities must be strings")
    rows = flatten_target_ops(target)
    if not rows:
        raise ValueError("Target set must contain operators")
    valid = re.compile(r"^(aten|prims)::[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_0-9]+$")
    for row in rows:
        if not valid.fullmatch(row["operator"]):
            raise ValueError(f"Use namespace::op.overload: {row['operator']}")
    keys = {r["operator"] for r in rows}
    for field in ("known_partial_or_limited", "decomp_aliases", "schemas"):
        if not isinstance(target.get(field, {}), dict):
            raise ValueError(f"{field} must be an object")
        if set(target.get(field, {})) - keys:
            raise ValueError(f"{field} contains keys outside the denominator")
    for aliases in target.get("decomp_aliases", {}).values():
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and valid.fullmatch(alias)
            for alias in aliases
        ):
            raise ValueError(
                "Alias candidates must be qualified operator lists"
            )
    return target


def flatten_target_ops(target: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten families into unique ops; keep all family memberships."""
    order: list[str] = []
    first_family: dict[str, str] = {}
    all_families: dict[str, list[str]] = {}
    for family, meta in target.get("families", {}).items():
        for op in meta.get("ops", []):
            if op not in first_family:
                first_family[op] = family
                order.append(op)
                all_families[op] = [family]
            elif family not in all_families[op]:
                all_families[op].append(family)
    return [{"operator": op, "families": all_families[op]} for op in order]
