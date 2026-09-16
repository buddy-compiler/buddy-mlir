# ===- parse_frontend.py - Static parsers for Buddy PyTorch frontend -------===
#
# Licensed under the Apache License, Version 2.0 (the "License").
# ===----------------------------------------------------------------------===
"""
Parse DynamoCompiler._ops_map and dialect ops_registry tables from source.

Static mode does not import buddy/torch; it only reads the frontend tree so the
tool can run before a full MLIR Python build exists.
"""

from __future__ import annotations

import ast
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
        # e.g. module.FooOp -> FooOp
        return node.attr
    return None


def parse_ops_map(frontend_py: Path) -> dict[str, str]:
    """Return aten_symbol -> BuddyOpClassName from DynamoCompiler._ops_map."""
    src = frontend_py.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(frontend_py))

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


_REGISTRY_ASSIGN = re.compile(
    r"^(?:llm_)?ops_registry\s*=\s*\{", re.MULTILINE
)


def parse_ops_registry_file(path: Path) -> dict[str, str]:
    """
    Parse a Python file containing `ops_registry = { "OpName": fn, ... }`.

    Returns BuddyOpClassName -> lowering_function_name (best effort).
    """
    src = path.read_text(encoding="utf-8")
    # Prefer AST on the whole file; registries can be large but are valid Python.
    tree = ast.parse(src, filename=str(path))
    out: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in ("ops_registry", "llm_ops_registry"):
                continue
            value = node.value
            # Handle `ops_registry = { **other, "X": y }`
            dict_nodes: list[ast.Dict] = []
            if isinstance(value, ast.Dict):
                dict_nodes.append(value)
            elif isinstance(value, ast.Call):
                continue
            for d in dict_nodes:
                for key, val in zip(d.keys, d.values):
                    if key is None:
                        continue
                    op_name = _literal_str(key)
                    if op_name is None:
                        continue
                    fn = _name_of(val) or "<expr>"
                    out[op_name] = fn
            # Starred merges: ops_registry = { **foo.llm_ops_registry, ... }
            if isinstance(value, ast.Dict):
                for key, val in zip(value.keys, value.values):
                    if key is None and isinstance(val, ast.Name):
                        # cannot resolve here
                        continue
    return out


def parse_all_registries(ops_dir: Path) -> dict[str, dict[str, str]]:
    """
    Return dialect -> {BuddyOpClassName -> lowering_fn}.

    Files: tosa.py, linalg.py, math.py, func.py, ttir.py, ttir_llm.py
    """
    dialects: dict[str, dict[str, str]] = {}
    for name in ("tosa.py", "linalg.py", "math.py", "func.py", "ttir.py", "ttir_llm.py"):
        path = ops_dir / name
        if not path.exists():
            continue
        dialect = path.stem
        registry = parse_ops_registry_file(path)
        if registry:
            dialects[dialect] = registry
    return dialects


def merge_registry_keys(dialects: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """BuddyOpClassName -> list of dialects that register a lowering."""
    merged: dict[str, list[str]] = {}
    for dialect, registry in dialects.items():
        for op_name in registry:
            merged.setdefault(op_name, []).append(dialect)
    return merged


def load_target_op_set(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def flatten_target_ops(target: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten families into unique op records.

    Each op keeps `family` (first-seen) and `families` (all memberships) so
    MoE subset stats can include ops also listed under transformer_core.
    """
    order: list[str] = []
    first_family: dict[str, str] = {}
    all_families: dict[str, list[str]] = {}
    families = target.get("families", {})
    for family, meta in families.items():
        for op in meta.get("ops", []):
            if op not in first_family:
                first_family[op] = family
                order.append(op)
                all_families[op] = [family]
            elif family not in all_families[op]:
                all_families[op].append(family)
    return [
        {
            "aten": op,
            "family": first_family[op],
            "families": all_families[op],
        }
        for op in order
    ]
