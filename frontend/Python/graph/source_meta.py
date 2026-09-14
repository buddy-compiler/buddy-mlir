import re
from collections.abc import Mapping
from dataclasses import dataclass

_ATEN_NAME = re.compile(
    r"aten\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z"
)


@dataclass(frozen=True)
class SourceMeta:
    module_path: str | None = None
    module_class: str | None = None
    original_aten: str | None = None


def _clean_path(value) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    prefix = "L['self']."
    return value[len(prefix) :] if value.startswith(prefix) else value


def _qualified_name(value) -> str | None:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    if not module or not qualname:
        return None
    return f"{module}.{qualname}"


def _stable_string(value) -> str | None:
    if value is None:
        return None
    try:
        result = str(value)
    except Exception:
        return None
    return result if _ATEN_NAME.fullmatch(result) else None


def extract_source_meta(gm_node) -> tuple[SourceMeta, ...]:
    try:
        meta = gm_node.meta
        if not isinstance(meta, Mapping):
            meta = {}
    except Exception:
        meta = {}

    module_path = None
    module_class = None
    try:
        stack = meta.get("nn_module_stack")
        candidates = []
        class_candidates = []
        if isinstance(stack, Mapping):
            for value in stack.values():
                try:
                    candidate_class = _qualified_name(value[1])
                except Exception:
                    candidate_class = None
                if candidate_class is not None:
                    class_candidates.append(candidate_class)
                try:
                    path = _clean_path(value[0])
                except Exception:
                    path = None
                if path is not None:
                    candidates.append((len(path.split(".")), path, value))
        if candidates:
            _, module_path, selected = max(candidates, key=lambda item: item[0])
            try:
                module_class = _qualified_name(selected[1])
            except Exception:
                module_class = None
        elif class_candidates:
            module_class = class_candidates[0]
    except Exception:
        module_path = None

    original_aten = None
    try:
        original_aten = _stable_string(meta.get("original_aten"))
    except Exception:
        original_aten = None

    result = SourceMeta(module_path, module_class, original_aten)
    return (
        (result,)
        if result.module_path is not None
        or result.module_class is not None
        or result.original_aten is not None
        else ()
    )


def merge_source_meta(
    *groups: tuple[SourceMeta, ...],
) -> tuple[SourceMeta, ...]:
    merged = []
    seen = set()
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return tuple(merged)
