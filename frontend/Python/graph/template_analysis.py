# ===- template_analysis.py --------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ===-------------------------------------------------------------------------

"""Deterministic, graph-independent fingerprints for LayerRegion templates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .operation import Op, TensorConstantOp
from .region_analysis import (
    LayerRegion,
    RegionInterface,
    _operand_dict_items,
    _resolve_operand_node_reference,
)
from .type import TensorMeta

if TYPE_CHECKING:
    from .graph import Graph
    from .structure_analysis import NodeAnnotation


@dataclass(frozen=True)
class FingerprintSummary:
    node_count: int
    internal_dependency_count: int
    data_input_count: int
    parameter_count: int
    constant_count: int
    data_output_count: int


@dataclass(frozen=True)
class RegionFingerprint:
    digest: str
    summary: FingerprintSummary


@dataclass
class TemplateGroup:
    fingerprint: RegionFingerprint
    representative: LayerRegion
    instances: list[LayerRegion]
    canonical_form: bytes


@dataclass
class TemplateIndex:
    region_fingerprints: dict[LayerRegion, RegionFingerprint]
    template_groups: list[TemplateGroup]
    non_reusable_regions: list[LayerRegion]


@dataclass
class _Candidate:
    canonical_form: bytes
    fingerprint: RegionFingerprint
    instances: list[LayerRegion]


@dataclass(frozen=True)
class _NodeRef:
    op: Op


class _UnsupportedFingerprintValue(Exception):
    pass


_LAYER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])layers\.([0-9]+)(?=\.|$)")
_ENCODER_LAYER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])encoder\.layer\."
    r"((?:0|[1-9][0-9]*))(?=\.|$)"
)


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _normalize(value: Any) -> Any:
    """Return a JSON-safe value without unstable repr or object addresses."""
    if value is None:
        return value
    if isinstance(value, Enum):
        return {"enum": _qualified_type(value), "name": value.name}
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, complex):
        return {"complex": [_normalize(value.real), _normalize(value.imag)]}
    if isinstance(value, slice):
        return {
            "slice": [
                _normalize(value.start),
                _normalize(value.stop),
                _normalize(value.step),
            ]
        }
    if isinstance(value, range):
        return {"range": [value.start, value.stop, value.step]}
    if isinstance(value, TensorMeta):
        return {
            "shape": _normalize(value.shape),
            "dtype": _normalize(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return {
            "sequence": "tuple" if isinstance(value, tuple) else "list",
            "items": [_normalize(item) for item in value],
        }
    if isinstance(value, dict):
        items = [(_normalize(key), _normalize(item)) for key, item in value.items()]
        items.sort(key=lambda item: _json_bytes(item[0]))
        return {"dict": items}

    qualified_type = _qualified_type(value)
    if qualified_type in {
        "torch.dtype",
        "torch.device",
        "torch.layout",
        "torch.memory_format",
    } or qualified_type.startswith("numpy.dtypes."):
        return {"scalar_type": qualified_type, "value": str(value)}

    # torch/numpy tensors and arrays are compile-time literals only when they
    # reach this function. Parameter operations are represented by slots.
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        descriptor = {
            "type": qualified_type,
            "shape": _normalize(tuple(shape)),
            "dtype": str(dtype),
        }
        try:
            count = int(value.numel())
        except Exception:
            try:
                count = int(value.size)
            except Exception:
                count = None
        if count is not None and count <= 64:
            try:
                descriptor["value"] = _normalize(value.tolist())
                return {"tensor": descriptor}
            except Exception:
                pass
        try:
            raw = value.detach().cpu().contiguous().numpy().tobytes()
        except Exception:
            try:
                raw = value.tobytes()
            except Exception:
                raw = None
        if raw is not None:
            descriptor["content_sha256"] = hashlib.sha256(raw).hexdigest()
            return {"tensor": descriptor}
        raise _UnsupportedFingerprintValue

    # Symbolic dimensions commonly expose stable node expressions. Never fall
    # back to repr(value), which may contain an address.
    node = getattr(value, "node", None)
    if node is not None:
        try:
            expression = str(node)
        except Exception:
            expression = None
        if expression is not None and "0x" not in expression:
            return {"symbolic": expression, "type": qualified_type}
    raise _UnsupportedFingerprintValue


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _shape_dtype(op: Op) -> dict[str, Any]:
    meta = op.tensor_meta
    if isinstance(meta, TensorMeta):
        shape, dtype = meta.shape, meta.dtype
    elif isinstance(meta, dict):
        shape, dtype = meta.get("shape"), meta.get("dtype")
    else:
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
    return {"shape": _normalize(shape), "dtype": _normalize(dtype)}


def _semantic_attributes(op: Op) -> dict[str, Any]:
    attributes = {}
    layout = getattr(op, "_layout", None)
    if layout is not None:
        attributes["layout"] = _normalize(layout)
    if hasattr(op, "call_func_name"):
        attributes["callee"] = _normalize(op.call_func_name)
        attributes["argument_result_indices"] = _normalize(
            getattr(op, "_args_index", [])
        )
    return attributes


class LayerFingerprintBuilder:
    """Build one canonical form while its region is scanned for interfaces."""

    def __init__(
        self,
        graph: Graph,
        region: LayerRegion,
        annotations: dict[Op, NodeAnnotation],
        node_to_region: dict[Op, Any],
        params: set[Op],
    ) -> None:
        self._graph = graph
        self._region = region
        self._annotations = annotations
        self._node_to_region = node_to_region
        self._local_ids: dict[Op, int] = {}
        self._params = params
        self._slots: dict[str, dict[Op, int]] = {
            "input": {},
            "param": {},
            "const": {},
        }
        self._slot_descriptors: dict[str, list[Any]] = {
            "input": [],
            "param": [],
            "const": [],
        }
        self._nodes: list[Any] = []
        self._internal_dependencies = 0
        self._unsupported = False

    def consume(self, op: Op) -> None:
        if self._unsupported:
            return
        self._local_ids[op] = len(self._nodes)
        annotation = self._annotations.get(op)
        sources = set()
        for source in op._source_meta:
            path = source.module_path
            if path is not None:
                expected = self._region.layer_index
                path = _LAYER_TOKEN.sub(
                    lambda match: "layers.{L}"
                    if int(match.group(1)) == expected
                    else match.group(0),
                    path,
                )
                path = _ENCODER_LAYER_TOKEN.sub(
                    lambda match: "encoder.layer.{L}"
                    if int(match.group(1)) == expected
                    else match.group(0),
                    path,
                )
            sources.add((path, source.module_class, source.original_aten))
        normalized_sources = [list(item) for item in sorted(sources, key=_json_bytes)]
        original_aten = sorted(
            {item[2] for item in sources if item[2] is not None}
        )
        try:
            self._nodes.append(
                {
                    "op": _qualified_type(op),
                    "original_aten": original_aten,
                    "source_meta": normalized_sources,
                    "component": annotation.component if annotation else None,
                    "subcomponent": annotation.subcomponent if annotation else None,
                    "result": _shape_dtype(op),
                    "attributes": _semantic_attributes(op),
                    "args": self._operand(op.args),
                    "kwargs": self._operand(op.kwargs),
                }
            )
        except _UnsupportedFingerprintValue:
            self._unsupported = True
            self._nodes.clear()
            self._local_ids.clear()
            for slots in self._slots.values():
                slots.clear()
            for descriptors in self._slot_descriptors.values():
                descriptors.clear()

    def _operand(self, value: Any) -> Any:
        referenced = _resolve_operand_node_reference(
            value, self._graph.node_table
        )
        if referenced is not None:
            if self._node_to_region.get(referenced) is self._region:
                self._internal_dependencies += 1
                return _NodeRef(referenced)
            if referenced in self._params:
                kind = "param"
            elif isinstance(referenced, TensorConstantOp):
                kind = "const"
            else:
                kind = "input"
            slots = self._slots[kind]
            slot = slots.get(referenced)
            if slot is None:
                slot = len(slots)
                slots[referenced] = slot
                descriptor: Any = _shape_dtype(referenced)
                if kind == "const":
                    descriptor = {
                        **descriptor,
                        "value": _normalize(referenced.args),
                    }
                self._slot_descriptors[kind].append(descriptor)
            return [kind, slot]
        if isinstance(value, list):
            return ["list", [self._operand(item) for item in value]]
        if isinstance(value, tuple):
            return ["tuple", [self._operand(item) for item in value]]
        if isinstance(value, dict):
            items = [
                [self._operand(key), self._operand(item)]
                for key, item in _operand_dict_items(value)
            ]
            items.sort(key=lambda item: _json_bytes(item[0]))
            return ["dict", items]
        return ["literal", _normalize(value)]

    def _resolve_node_refs(self, value: Any) -> Any:
        if isinstance(value, _NodeRef):
            return ["node", self._local_ids[value.op]]
        if isinstance(value, list):
            return [self._resolve_node_refs(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._resolve_node_refs(item) for key, item in value.items()
            }
        return value

    def finish(
        self, interface: RegionInterface
    ) -> tuple[bytes, FingerprintSummary] | None:
        if self._unsupported:
            return None
        summary = FingerprintSummary(
            node_count=len(self._region.nodes),
            internal_dependency_count=self._internal_dependencies,
            data_input_count=len(interface.data_inputs),
            parameter_count=len(interface.parameters),
            constant_count=len(interface.constants),
            data_output_count=len(interface.data_outputs),
        )
        canonical = _json_bytes(
            {
                "version": 1,
                "nodes": self._resolve_node_refs(self._nodes),
                "external_slots": self._slot_descriptors,
                "interface": {
                    "data_inputs": summary.data_input_count,
                    "parameters": summary.parameter_count,
                    "constants": summary.constant_count,
                    "data_outputs": [
                        self._local_ids[op] for op in interface.data_outputs
                    ],
                },
            }
        )
        return canonical, summary


class TemplateRecognizer:
    """Collision-safe grouping with one retained canonical form per template."""

    def __init__(self) -> None:
        self._by_digest: dict[str, list[_Candidate]] = {}
        self._region_fingerprints: dict[LayerRegion, RegionFingerprint] = {}
        self._non_reusable_regions: list[LayerRegion] = []

    def _mark_non_reusable(self, region: LayerRegion) -> None:
        self._non_reusable_regions.append(region)

    def make_builder(
        self,
        graph: Graph,
        region: LayerRegion,
        annotations: dict[Op, NodeAnnotation],
        node_to_region: dict[Op, Any],
        params: set[Op],
    ) -> LayerFingerprintBuilder:
        return LayerFingerprintBuilder(
            graph, region, annotations, node_to_region, params
        )

    def add(
        self,
        region: LayerRegion,
        canonical_form: bytes,
        summary: FingerprintSummary,
    ) -> None:
        digest = hashlib.sha256(canonical_form).hexdigest()
        candidates = self._by_digest.setdefault(digest, [])
        for candidate in candidates:
            if candidate.canonical_form == canonical_form:
                candidate.instances.append(region)
                self._region_fingerprints[region] = candidate.fingerprint
                return
        fingerprint = RegionFingerprint(digest, summary)
        candidates.append(_Candidate(canonical_form, fingerprint, [region]))
        self._region_fingerprints[region] = fingerprint

    def finish(self) -> TemplateIndex:
        groups = []
        non_reusable = list(self._non_reusable_regions)
        for candidates in self._by_digest.values():
            for candidate in candidates:
                candidate.instances.sort(key=lambda region: region.layer_index)
                if len(candidate.instances) == 1:
                    non_reusable.append(candidate.instances[0])
                    continue
                groups.append(
                    TemplateGroup(
                        fingerprint=candidate.fingerprint,
                        representative=candidate.instances[0],
                        instances=candidate.instances,
                        canonical_form=candidate.canonical_form,
                    )
                )
        groups.sort(key=lambda group: group.representative.layer_index)
        non_reusable = sorted(
            set(non_reusable), key=lambda region: region.layer_index
        )
        return TemplateIndex(self._region_fingerprints, groups, non_reusable)


def build_template_index(graph: Graph, structure_index) -> TemplateIndex:
    """Supplement an existing structure index without rebuilding any Region."""
    recognizer = TemplateRecognizer()
    params = set(graph.params)
    layers = sorted(
        (
            region
            for region in structure_index.regions
            if isinstance(region, LayerRegion)
        ),
        key=lambda region: region.layer_index,
    )
    for region in layers:
        builder = recognizer.make_builder(
            graph,
            region,
            structure_index.annotations,
            structure_index.node_to_region,
            params,
        )
        for op in region.nodes:
            builder.consume(op)
        fingerprint = builder.finish(region.interface)
        if fingerprint is None:
            recognizer._mark_non_reusable(region)
        else:
            recognizer.add(region, *fingerprint)
    return recognizer.finish()
