# ===- transformer_partition.py ----------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===-------------------------------------------------------------------------

"""Deterministic Transformer partition analysis over an existing Buddy Graph.

The module owns the complete pure-analysis pipeline: structural Region
recognition, Region interface construction, template fingerprint grouping,
instance binding, materialization sequencing, and final plan verification.
It never rewrites the source Graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from . import operation as operation
from .operation import Op, OutputOp, TensorConstantOp
from .structure_analysis import (
    ModuleStructureAnalyzer,
    NodeAnnotation,
    _layer_key,
)
from .type import TensorMeta

if TYPE_CHECKING:
    from .graph import Graph


class RegionKind(Enum):
    PRELUDE = auto()
    LAYER = auto()
    EPILOGUE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class GraphValueRef:
    op: Op
    result_index: int = 0


class RegionInputKind(Enum):
    DATA = auto()
    PARAMETER = auto()
    CONSTANT = auto()
    STATE = auto()


@dataclass(frozen=True)
class RegionInputRef:
    kind: RegionInputKind
    value: GraphValueRef


@dataclass
class RegionInterface:
    data_inputs: list[Op] = field(default_factory=list)
    data_outputs: list[Op] = field(default_factory=list)
    parameters: list[Op] = field(default_factory=list)
    constants: list[Op] = field(default_factory=list)
    state_inputs: list[Op] = field(default_factory=list)
    state_outputs: list[Op] = field(default_factory=list)
    ordered_inputs: list[RegionInputRef] = field(default_factory=list)
    ordered_outputs: list[GraphValueRef] = field(default_factory=list)


@dataclass(eq=False)
class GraphRegion:
    kind: RegionKind
    nodes: list[Op]
    interface: RegionInterface = field(default_factory=RegionInterface)


@dataclass(eq=False)
class LayerRegion(GraphRegion):
    layer_index: int = 0
    component_nodes: dict[str, list[Op]] = field(default_factory=dict)
    subcomponent_nodes: dict[str, list[Op]] = field(default_factory=dict)
    layer_container: str | None = None


@dataclass
class GraphStructureIndex:
    annotations: dict[Op, NodeAnnotation]
    regions: list[GraphRegion]
    node_to_region: dict[Op, GraphRegion]


def _stable_operand_dict_key(value) -> tuple[str, str]:
    value_type = type(value)
    qualified_type = f"{value_type.__module__}.{value_type.__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return qualified_type, repr(value)
    return qualified_type, ""


def _operand_dict_items(value: dict):
    return sorted(
        value.items(), key=lambda item: _stable_operand_dict_key(item[0])
    )


def _resolve_operand_node_reference(
    value, node_table: dict[str, Op]
) -> Op | None:
    if isinstance(value, str):
        return node_table.get(value)
    return None


def _iter_operand_value_references(
    value,
    node_table: dict[str, Op],
    result_index: int = 0,
    path: tuple[int | str, ...] = (),
    with_paths: bool = False,
):
    referenced = _resolve_operand_node_reference(value, node_table)
    if referenced is not None:
        reference = GraphValueRef(referenced, result_index)
        yield (reference, path) if with_paths else reference
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_operand_value_references(
                item,
                node_table,
                result_index,
                path + (index,),
                with_paths,
            )
    elif isinstance(value, dict):
        for key, item in _operand_dict_items(value):
            yield from _iter_operand_value_references(
                key,
                node_table,
                result_index,
                path + ("key", key),
                with_paths,
            )
            yield from _iter_operand_value_references(
                item,
                node_table,
                result_index,
                path + (key,),
                with_paths,
            )


def iter_op_input_references(
    op: Op, node_table: dict[str, Op], *, with_paths: bool = False
):
    """Iterate operand value references, optionally retaining their paths.

    Paths use ``("args", index, ...)`` and ``("kwargs", key, ...)`` and are
    derived by this same traversal; planning therefore cannot disagree with
    Region/template analysis about what constitutes an operand reference.
    """
    for index, value in enumerate(op.args):
        result_index = (
            op._args_index[index] if index < len(op._args_index) else 0
        )
        yield from _iter_operand_value_references(
            value,
            node_table,
            result_index,
            ("args", index),
            with_paths,
        )
    for key, value in _operand_dict_items(op.kwargs):
        yield from _iter_operand_value_references(
            key,
            node_table,
            path=("kwargs", "key", key),
            with_paths=with_paths,
        )
        yield from _iter_operand_value_references(
            value,
            node_table,
            path=("kwargs", key),
            with_paths=with_paths,
        )


class RegionBuilder:
    """Build a deterministic structural index without mutating its graph."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def build(
        self, template_recognizer: TemplateRecognizer | None = None
    ) -> GraphStructureIndex:
        graph = self._graph
        analyzer = ModuleStructureAnalyzer()
        annotations = analyzer.analyze(graph).node_annotations
        body_positions: dict[Op, int] = {}
        body_nodes: set[Op] = set()
        eligible_set: set[Op] = set()
        params = set(graph.params)
        excluded = set(graph.inputs) | params

        regions: list[GraphRegion] = []
        node_to_region: dict[Op, GraphRegion] = {}
        current_region: GraphRegion | None = None
        current_identity: tuple[Any, ...] | None = None
        pending_unknown_ops: list[Op] = []
        seen_layer = False

        def append_node(region: GraphRegion, op: Op) -> None:
            region.nodes.append(op)
            if not isinstance(region, LayerRegion):
                return
            annotation = annotations.get(op, NodeAnnotation())
            if annotation.component is not None:
                region.component_nodes.setdefault(
                    annotation.component, []
                ).append(op)
            if annotation.subcomponent is not None:
                region.subcomponent_nodes.setdefault(
                    annotation.subcomponent, []
                ).append(op)

        def finish_current_region() -> None:
            nonlocal current_region, current_identity
            if current_region is None:
                return
            self._assign(current_region, node_to_region)
            regions.append(current_region)
            current_region = None
            current_identity = None

        def finish_pending_unknown() -> None:
            if not pending_unknown_ops:
                return
            region = GraphRegion(RegionKind.UNKNOWN, list(pending_unknown_ops))
            self._assign(region, node_to_region)
            regions.append(region)
            pending_unknown_ops.clear()

        # Classifications are already complete, including topology refinements.
        # Build Regions as a forward-only stream so a finalized identity is
        # never reopened later in graph.body.
        for position, op in enumerate(graph.body):
            body_positions[op] = position
            body_nodes.add(op)
            annotation = annotations.get(op, NodeAnnotation())
            if op in excluded or isinstance(op, (TensorConstantOp, OutputOp)):
                continue
            eligible_set.add(op)

            if annotation.layer_index is not None:
                layer_container, layer_index = _layer_key(annotation)
                identity = (RegionKind.LAYER, layer_container, layer_index)
            elif annotation.component == "embedding":
                identity = (RegionKind.PRELUDE,)
            elif (
                annotation.component == "lm_head"
                or annotation.subcomponent == "final_norm"
            ):
                identity = (RegionKind.EPILOGUE,)
            else:
                pending_unknown_ops.append(op)
                continue

            if (
                current_region is None
                and identity == (RegionKind.PRELUDE,)
                and pending_unknown_ops
                and not seen_layer
            ):
                current_identity = identity
                current_region = GraphRegion(RegionKind.PRELUDE, [])
                for pending_op in pending_unknown_ops:
                    append_node(current_region, pending_op)
                pending_unknown_ops.clear()
                append_node(current_region, op)
                continue

            if current_region is not None and identity == current_identity:
                for pending_op in pending_unknown_ops:
                    append_node(current_region, pending_op)
                pending_unknown_ops.clear()
                append_node(current_region, op)
                continue

            if (
                current_region is not None
                and current_identity == (RegionKind.PRELUDE,)
                and identity[0] is RegionKind.LAYER
                and not seen_layer
            ):
                for pending_op in pending_unknown_ops:
                    append_node(current_region, pending_op)
                pending_unknown_ops.clear()

            finish_current_region()
            finish_pending_unknown()
            current_identity = identity
            if identity[0] is RegionKind.LAYER:
                seen_layer = True
                current_region = LayerRegion(
                    kind=RegionKind.LAYER,
                    nodes=[],
                    layer_index=layer_index,
                    layer_container=layer_container,
                )
            else:
                current_region = GraphRegion(identity[0], [])
            append_node(current_region, op)

        finish_current_region()
        finish_pending_unknown()

        ordered_region_outputs = {region: [] for region in regions}
        seen_region_outputs = {region: set() for region in regions}
        for consumer in graph.body:
            consumer_region = node_to_region.get(consumer)
            for value in iter_op_input_references(consumer, graph.node_table):
                producer_region = node_to_region.get(value.op)
                if (
                    producer_region is None
                    or producer_region is consumer_region
                    or value in seen_region_outputs[producer_region]
                ):
                    continue
                seen_region_outputs[producer_region].add(value)
                ordered_region_outputs[producer_region].append(value)
        # Pass 2: each Region node is visited once for strict edge validation,
        # interface construction, and optional canonical fingerprint tokens.
        for region in regions:
            fingerprint_builder = None
            if template_recognizer is not None and isinstance(
                region, LayerRegion
            ):
                fingerprint_builder = template_recognizer.make_builder(
                    graph, region, annotations, node_to_region
                )
            region.interface = self._build_interface(
                region,
                node_to_region,
                body_positions,
                params,
                ordered_region_outputs[region],
                fingerprint_builder,
            )
            self._validate_interface(region, body_nodes)
            if fingerprint_builder is not None:
                fingerprint = fingerprint_builder.finish(region.interface)
                if fingerprint is None:
                    template_recognizer._mark_non_reusable(region)
                else:
                    template_recognizer.add(region, *fingerprint)

        index = GraphStructureIndex(annotations, regions, node_to_region)
        if len(node_to_region) != len(eligible_set):
            raise ValueError(
                "not every region-eligible node belongs to exactly one region"
            )
        return index

    @staticmethod
    def _assign(
        region: GraphRegion, node_to_region: dict[Op, GraphRegion]
    ) -> None:
        for op in region.nodes:
            previous = node_to_region.get(op)
            if previous is not None:
                raise ValueError(
                    f"node {op.name!r} belongs to both "
                    f"{previous.kind.name} and {region.kind.name} regions"
                )
            node_to_region[op] = region

    def _resolve_edge(self, owner: Op, name: str, edge_kind: str) -> Op:
        try:
            return self._graph.node_table[name]
        except KeyError as error:
            raise RuntimeError(
                f"node {owner.name!r} has {edge_kind} {name!r} that is not "
                "present in graph.node_table"
            ) from error

    def _build_interface(
        self,
        region: GraphRegion,
        node_to_region: dict[Op, GraphRegion],
        body_positions: dict[Op, int],
        params: set[Op],
        ordered_outputs: list[GraphValueRef],
        fingerprint_builder=None,
    ) -> RegionInterface:
        interface = RegionInterface()
        seen_inputs: set[GraphValueRef] = set()
        local_seen: set[Op] = set()
        previous_position = -1

        for op in region.nodes:
            position = body_positions.get(op)
            if position is None:
                raise ValueError(
                    f"region {region.kind.name} references node {op.name!r} "
                    "that is not an original graph node"
                )
            if op in local_seen:
                raise ValueError(
                    f"region {region.kind.name} contains duplicate node "
                    f"{op.name!r}"
                )
            if position < previous_position:
                raise ValueError(
                    f"region {region.kind.name} nodes do not preserve "
                    "graph.body order"
                )
            if node_to_region.get(op) is not region:
                raise ValueError(
                    f"node_to_region is inconsistent for node {op.name!r}"
                )
            local_seen.add(op)
            previous_position = position
            # Parent/child names remain the authoritative use-def validation.
            for parent_name in op.parents:
                self._resolve_edge(op, parent_name, "parent")
            for value in iter_op_input_references(op, self._graph.node_table):
                dependency = value.op
                if (
                    node_to_region.get(dependency) is region
                    or value in seen_inputs
                ):
                    continue
                seen_inputs.add(value)
                if dependency in params:
                    kind = RegionInputKind.PARAMETER
                    interface.parameters.append(dependency)
                elif isinstance(dependency, TensorConstantOp):
                    kind = RegionInputKind.CONSTANT
                    interface.constants.append(dependency)
                else:
                    kind = RegionInputKind.DATA
                    interface.data_inputs.append(dependency)
                interface.ordered_inputs.append(RegionInputRef(kind, value))
            for child_name in op._children:
                self._resolve_edge(op, child_name, "child")
            if fingerprint_builder is not None:
                fingerprint_builder.consume(op)

        interface.ordered_outputs.extend(ordered_outputs)
        interface.data_outputs.extend(value.op for value in ordered_outputs)

        return interface

    @staticmethod
    def _validate_interface(region: GraphRegion, body_nodes: set[Op]) -> None:
        interface = region.interface
        categories = (
            ("data_inputs", interface.data_inputs),
            ("data_outputs", interface.data_outputs),
            ("parameters", interface.parameters),
            ("constants", interface.constants),
            ("state_inputs", interface.state_inputs),
            ("state_outputs", interface.state_outputs),
        )
        for category, nodes in categories:
            for op in nodes:
                if op not in body_nodes:
                    raise ValueError(
                        f"region {region.kind.name} {category} references node "
                        f"{op.name!r} outside the original graph"
                    )

        ordered_values = [item.value for item in interface.ordered_inputs]
        if len(ordered_values) != len(set(ordered_values)):
            raise ValueError(
                f"region {region.kind.name} ordered_inputs contains a duplicate "
                "GraphValueRef"
            )
        expected_inputs = {
            RegionInputKind.DATA: interface.data_inputs,
            RegionInputKind.PARAMETER: interface.parameters,
            RegionInputKind.CONSTANT: interface.constants,
            RegionInputKind.STATE: interface.state_inputs,
        }
        for kind, classified in expected_inputs.items():
            ordered = [
                item.value.op
                for item in interface.ordered_inputs
                if item.kind is kind
            ]
            if ordered != classified:
                raise ValueError(
                    f"region {region.kind.name} {kind.name.lower()} inputs do "
                    "not match ordered_inputs"
                )
        if len(interface.ordered_outputs) != len(
            set(interface.ordered_outputs)
        ):
            raise ValueError(
                f"region {region.kind.name} ordered_outputs contains a duplicate "
                "GraphValueRef"
            )
        if [value.op for value in interface.ordered_outputs] != (
            interface.data_outputs + interface.state_outputs
        ):
            raise ValueError(
                f"region {region.kind.name} outputs do not match ordered_outputs"
            )
        for op in interface.data_outputs:
            if op not in region.nodes:
                raise ValueError(
                    f"region {region.kind.name} data output {op.name!r} is not "
                    "an internal producer"
                )


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
    result_index: int = 0


@dataclass(frozen=True)
class _ExternalRef:
    value: GraphValueRef


class _UnsupportedFingerprintValue(Exception):
    pass


_FINGERPRINT_CONVERSION_ERRORS = (
    AttributeError,
    NotImplementedError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
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
        return {
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
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
        items = [
            (_normalize(key), _normalize(item)) for key, item in value.items()
        ]
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
        except _FINGERPRINT_CONVERSION_ERRORS:
            try:
                count = int(value.size)
            except _FINGERPRINT_CONVERSION_ERRORS:
                count = None
        if count is not None and count <= 64:
            try:
                descriptor["value"] = _normalize(value.tolist())
                return {"tensor": descriptor}
            except _FINGERPRINT_CONVERSION_ERRORS:
                pass
        try:
            raw = value.detach().cpu().contiguous().numpy().tobytes()
        except _FINGERPRINT_CONVERSION_ERRORS:
            try:
                raw = value.tobytes()
            except _FINGERPRINT_CONVERSION_ERRORS:
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
        except _FINGERPRINT_CONVERSION_ERRORS:
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


def _value_shape_dtype(value: GraphValueRef) -> dict[str, Any]:
    meta = value.op.tensor_meta
    if isinstance(meta, TensorMeta):
        shape, dtype = meta.shape, meta.dtype
    elif isinstance(meta, dict):
        shape, dtype = meta.get("shape"), meta.get("dtype")
    else:
        shape = getattr(meta, "shape", None)
        dtype = getattr(meta, "dtype", None)
    is_multi_result = (
        isinstance(shape, (list, tuple))
        and bool(shape)
        and isinstance(shape[0], (list, tuple))
    )
    if is_multi_result:
        shape = shape[value.result_index]
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[value.result_index]
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
    ) -> None:
        self._graph = graph
        self._region = region
        self._annotations = annotations
        self._node_to_region = node_to_region
        self._local_ids: dict[Op, int] = {}
        self._nodes: list[Any] = []
        self._internal_dependencies = 0
        self._unsupported = False

    def consume(self, op: Op) -> None:
        if self._unsupported:
            return
        self._local_ids[op] = len(self._nodes)
        annotation = self._annotations.get(op)
        sources = set()
        layer_resolutions = (
            annotation.layer_resolutions if annotation is not None else ()
        )
        if len(layer_resolutions) != len(op._source_meta):
            layer_resolutions = (None,) * len(op._source_meta)
        for source, layer_resolution in zip(
            op._source_meta, layer_resolutions, strict=True
        ):
            path = source.module_path
            if (
                layer_resolution is not None
                and layer_resolution.layer_container
                == self._region.layer_container
                and layer_resolution.layer_index == self._region.layer_index
            ):
                path = layer_resolution.canonical_module_path
            sources.add((path, source.module_class, source.original_aten))
        normalized_sources = [
            list(item) for item in sorted(sources, key=_json_bytes)
        ]
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
                    "subcomponent": annotation.subcomponent
                    if annotation
                    else None,
                    "result": _shape_dtype(op),
                    "attributes": _semantic_attributes(op),
                    "args": [
                        self._operand(
                            value,
                            op._args_index[index]
                            if index < len(op._args_index)
                            else 0,
                        )
                        for index, value in enumerate(op.args)
                    ],
                    "kwargs": self._operand(op.kwargs),
                }
            )
        except _UnsupportedFingerprintValue:
            self._unsupported = True
            self._nodes.clear()
            self._local_ids.clear()

    def _operand(self, value: Any, result_index: int = 0) -> Any:
        referenced = _resolve_operand_node_reference(
            value, self._graph.node_table
        )
        if referenced is not None:
            if self._node_to_region.get(referenced) is self._region:
                self._internal_dependencies += 1
                return _NodeRef(referenced, result_index)
            return _ExternalRef(GraphValueRef(referenced, result_index))
        if isinstance(value, list):
            return [
                "list",
                [self._operand(item, result_index) for item in value],
            ]
        if isinstance(value, tuple):
            return [
                "tuple",
                [self._operand(item, result_index) for item in value],
            ]
        if isinstance(value, dict):
            items = [
                [
                    self._operand(key, result_index),
                    self._operand(item, result_index),
                ]
                for key, item in _operand_dict_items(value)
            ]
            items.sort(key=lambda item: _json_bytes(item[0]))
            return ["dict", items]
        return ["literal", _normalize(value)]

    def _resolve_node_refs(
        self,
        value: Any,
        external_slots: dict[GraphValueRef, tuple[str, int]],
    ) -> Any:
        if isinstance(value, _NodeRef):
            token = ["node", self._local_ids[value.op]]
            if value.result_index:
                token.append(value.result_index)
            return token
        if isinstance(value, _ExternalRef):
            return list(external_slots[value.value])
        if isinstance(value, list):
            return [
                self._resolve_node_refs(item, external_slots) for item in value
            ]
        if isinstance(value, dict):
            return {
                key: self._resolve_node_refs(item, external_slots)
                for key, item in value.items()
            }
        return value

    def finish(
        self, interface: RegionInterface
    ) -> tuple[bytes, FingerprintSummary] | None:
        if self._unsupported:
            return None
        kind_tokens = {
            RegionInputKind.DATA: "input",
            RegionInputKind.PARAMETER: "param",
            RegionInputKind.CONSTANT: "const",
            RegionInputKind.STATE: "state",
        }
        external_slots: dict[GraphValueRef, tuple[str, int]] = {}
        slot_descriptors: dict[str, list[Any]] = {
            token: [] for token in kind_tokens.values()
        }
        for input_ref in interface.ordered_inputs:
            token = kind_tokens[input_ref.kind]
            slot = len(slot_descriptors[token])
            external_slots[input_ref.value] = (token, slot)
            descriptor: Any = _value_shape_dtype(input_ref.value)
            if input_ref.kind is RegionInputKind.CONSTANT:
                descriptor = {
                    **descriptor,
                    "value": _normalize(input_ref.value.op.args),
                }
            slot_descriptors[token].append(descriptor)
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
                "version": 2,
                "nodes": self._resolve_node_refs(self._nodes, external_slots),
                "external_slots": slot_descriptors,
                "interface": {
                    "data_inputs": summary.data_input_count,
                    "parameters": summary.parameter_count,
                    "constants": summary.constant_count,
                    "ordered_inputs": [
                        kind_tokens[item.kind]
                        for item in interface.ordered_inputs
                    ],
                    "data_outputs": [
                        self._local_ids[value.op]
                        if value.result_index == 0
                        else [self._local_ids[value.op], value.result_index]
                        for value in interface.ordered_outputs
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
    ) -> LayerFingerprintBuilder:
        return LayerFingerprintBuilder(
            graph, region, annotations, node_to_region
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
        )
        for op in region.nodes:
            builder.consume(op)
        fingerprint = builder.finish(region.interface)
        if fingerprint is None:
            recognizer._mark_non_reusable(region)
        else:
            recognizer.add(region, *fingerprint)
    return recognizer.finish()


@dataclass(frozen=True)
class TemplateUnit:
    template_id: int
    representative: GraphRegion
    instances: tuple[GraphRegion, ...]


@dataclass
class TemplateMaterializationPlan:
    templates: list[TemplateUnit]
    region_to_template_id: dict[GraphRegion, int]
    parameter_indices: dict[Op, int]


@dataclass(frozen=True)
class TemplateInstanceBinding:
    """Stable external-value bindings for one Region instance."""

    region: GraphRegion
    template_id: int
    ordered_inputs: tuple[RegionInputRef, ...]
    ordered_outputs: tuple[GraphValueRef, ...]
    parameter_indices: tuple[int, ...]
    data_inputs: tuple[GraphValueRef, ...]
    state_inputs: tuple[GraphValueRef, ...]


@dataclass
class TransformerPartitionPlan(TemplateMaterializationPlan):
    """Complete analysis result consumed by template partitioning callers."""

    structure_index: GraphStructureIndex
    template_index: TemplateIndex
    instance_bindings: list[TemplateInstanceBinding]
    partition_sequence: tuple[GraphRegion, ...]


def graph_value_tensor_meta(value: GraphValueRef) -> TensorMeta:
    meta = value.op.tensor_meta
    shape = meta.shape if isinstance(meta, TensorMeta) else meta["shape"]
    dtype = meta.dtype if isinstance(meta, TensorMeta) else meta["dtype"]
    is_multi_result = (
        isinstance(shape, (list, tuple))
        and bool(shape)
        and isinstance(shape[0], (list, tuple))
    )
    if is_multi_result:
        try:
            shape = shape[value.result_index]
        except IndexError as error:
            raise ValueError(
                f"result index {value.result_index} is out of range for "
                f"{value.op.name!r}"
            ) from error
        if isinstance(dtype, (list, tuple)):
            try:
                dtype = dtype[value.result_index]
            except IndexError as error:
                raise ValueError(
                    f"dtype result index {value.result_index} is out of range "
                    f"for {value.op.name!r}"
                ) from error
    elif value.result_index != 0:
        raise ValueError(
            f"operation {value.op.name!r} has no metadata for result "
            f"{value.result_index}"
        )
    elif isinstance(dtype, (list, tuple)):
        if len(dtype) != 1:
            raise ValueError(
                f"operation {value.op.name!r} has ambiguous result dtype"
            )
        dtype = dtype[0]
    return TensorMeta(shape, dtype)


def _validate_v1_region_interface(region: GraphRegion) -> None:
    region_name = f"region {region.kind.name}"
    if any(
        input_ref.kind is RegionInputKind.CONSTANT
        for input_ref in region.interface.ordered_inputs
    ):
        raise ValueError(
            f"{region_name}: external TensorConstant inputs are not supported "
            "by template materialization V1"
        )
    if (
        any(
            input_ref.kind is RegionInputKind.STATE
            for input_ref in region.interface.ordered_inputs
        )
        or region.interface.state_inputs
        or region.interface.state_outputs
    ):
        raise ValueError(
            f"{region_name}: explicit RegionInputKind.STATE and "
            "state_inputs/state_outputs are not supported by template "
            "materialization V1"
        )


def _validate_template_unit(unit: TemplateUnit) -> None:
    if unit.representative not in unit.instances:
        raise ValueError(
            f"template {unit.template_id} representative is not an instance"
        )
    representative = unit.representative.interface
    representative_inputs = representative.ordered_inputs
    representative_outputs = representative.ordered_outputs
    for region in unit.instances:
        interface = region.interface
        if len(interface.ordered_inputs) != len(representative_inputs):
            raise ValueError(
                f"template {unit.template_id} input count mismatch"
            )
        for slot, (actual, expected) in enumerate(
            zip(
                interface.ordered_inputs,
                representative_inputs,
                strict=True,
            )
        ):
            if actual.kind is not expected.kind:
                raise ValueError(
                    f"template {unit.template_id} input {slot} kind mismatch"
                )
            actual_meta = graph_value_tensor_meta(actual.value)
            expected_meta = graph_value_tensor_meta(expected.value)
            if tuple(actual_meta.shape) != tuple(expected_meta.shape):
                raise ValueError(
                    f"template {unit.template_id} input {slot} shape mismatch"
                )
            if actual_meta.dtype != expected_meta.dtype:
                raise ValueError(
                    f"template {unit.template_id} input {slot} dtype mismatch"
                )
        if len(interface.ordered_outputs) != len(representative_outputs):
            raise ValueError(
                f"template {unit.template_id} output count mismatch"
            )
        for slot, (actual, expected) in enumerate(
            zip(
                interface.ordered_outputs,
                representative_outputs,
                strict=True,
            )
        ):
            actual_meta = graph_value_tensor_meta(actual)
            expected_meta = graph_value_tensor_meta(expected)
            if tuple(actual_meta.shape) != tuple(expected_meta.shape):
                raise ValueError(
                    f"template {unit.template_id} output {slot} shape mismatch"
                )
            if actual_meta.dtype != expected_meta.dtype:
                raise ValueError(
                    f"template {unit.template_id} output {slot} dtype mismatch"
                )


def build_template_materialization_plan(
    graph,
    structure_index,
    template_index,
    deduplicate_templates: bool = True,
) -> TemplateMaterializationPlan:
    """Assign stable template ids while retaining original Region instances.

    Template materialization V1 treats KV/cache graph inputs as DATA; explicit
    STATE ABI is reserved for a later version.
    """
    regions = list(structure_index.regions)
    for region in regions:
        _validate_v1_region_interface(region)
    region_set = set(regions)
    group_for_region = {}
    for group in template_index.template_groups:
        if group.representative not in group.instances:
            raise ValueError("template representative is not in instances")
        for region in group.instances:
            if region not in region_set:
                raise ValueError(
                    "template instance is not in structure regions"
                )
            if region in group_for_region:
                raise ValueError(
                    "region belongs to more than one template group"
                )
            group_for_region[region] = group
    for region in template_index.non_reusable_regions:
        if region not in region_set:
            raise ValueError("non-reusable Region is not in structure regions")
        if region in group_for_region:
            raise ValueError(
                "non-reusable Region also belongs to a template group"
            )

    templates = []
    region_to_template_id = {}
    emitted_groups = set()
    for region in regions:
        group = group_for_region.get(region) if deduplicate_templates else None
        if group is None:
            instances = (region,)
            representative = region
        else:
            group_key = id(group)
            if group_key in emitted_groups:
                continue
            emitted_groups.add(group_key)
            group_instances = set(group.instances)
            instances = tuple(
                item for item in regions if item in group_instances
            )
            representative = group.representative
        template_id = len(templates)
        unit = TemplateUnit(template_id, representative, instances)
        _validate_template_unit(unit)
        templates.append(unit)
        for instance in instances:
            if instance in region_to_template_id:
                raise ValueError("region maps to more than one template")
            region_to_template_id[instance] = template_id

    if set(region_to_template_id) != region_set:
        raise ValueError("not every Region maps to exactly one Template")

    parameter_indices = {
        param_op: index for index, param_op in enumerate(graph.params)
    }
    return TemplateMaterializationPlan(
        templates, region_to_template_id, parameter_indices
    )


def _detect_regions_and_build_interfaces(
    graph: Graph,
) -> tuple[GraphStructureIndex, TemplateIndex]:
    """Run the existing fused Region/interface/template analysis stages."""
    analysis = graph.analyze_structure(True)
    if analysis.template_index is None:
        raise ValueError("template recognition did not produce a TemplateIndex")
    return analysis.structure_index, analysis.template_index


def _build_partition_sequence(
    structure_index: GraphStructureIndex,
) -> tuple[GraphRegion, ...]:
    """Freeze the existing Region order without reordering any instance."""
    return tuple(structure_index.regions)


def _build_instance_bindings(
    structure_index: GraphStructureIndex,
    materialization_plan: TemplateMaterializationPlan,
) -> list[TemplateInstanceBinding]:
    """Capture each instance ABI in the exact Region interface order."""
    bindings = []
    for region in structure_index.regions:
        ordered_inputs = tuple(region.interface.ordered_inputs)
        parameter_indices = tuple(
            materialization_plan.parameter_indices[input_ref.value.op]
            for input_ref in ordered_inputs
            if input_ref.kind is RegionInputKind.PARAMETER
        )
        bindings.append(
            TemplateInstanceBinding(
                region=region,
                template_id=materialization_plan.region_to_template_id[region],
                ordered_inputs=ordered_inputs,
                ordered_outputs=tuple(region.interface.ordered_outputs),
                parameter_indices=parameter_indices,
                data_inputs=tuple(
                    input_ref.value
                    for input_ref in ordered_inputs
                    if input_ref.kind is RegionInputKind.DATA
                ),
                state_inputs=tuple(
                    input_ref.value
                    for input_ref in ordered_inputs
                    if input_ref.kind is RegionInputKind.STATE
                ),
            )
        )
    return bindings


def _verify_partition_plan(
    graph: Graph,
    plan: TransformerPartitionPlan,
) -> None:
    """Verify coverage, ordering, template ownership, and instance ABIs."""
    sequence = plan.partition_sequence
    if len(sequence) != len(plan.structure_index.regions) or any(
        actual is not expected
        for actual, expected in zip(
            sequence, plan.structure_index.regions, strict=True
        )
    ):
        raise ValueError("partition sequence does not preserve Region order")

    excluded = set(graph.inputs) | set(graph.params)
    eligible = {
        op
        for op in graph.body
        if op not in excluded
        and not isinstance(op, (TensorConstantOp, OutputOp))
    }
    covered = [op for region in sequence for op in region.nodes]
    if len(covered) != len(set(covered)):
        raise ValueError("partition sequence contains a node more than once")
    if set(covered) != eligible:
        raise ValueError(
            "partition sequence does not cover every eligible node"
        )

    body_positions = {op: index for index, op in enumerate(graph.body)}
    body_nodes = set(graph.body)
    previous_region_start = -1
    for region in sequence:
        positions = [body_positions[op] for op in region.nodes]
        if positions != sorted(positions):
            raise ValueError("Region nodes do not preserve graph.body order")
        if positions[0] < previous_region_start:
            raise ValueError(
                "partition sequence does not preserve Region order"
            )
        previous_region_start = positions[0]
        _validate_v1_region_interface(region)
        RegionBuilder._validate_interface(region, body_nodes)

    if len(plan.instance_bindings) != len(sequence):
        raise ValueError("partition plan does not bind every Region instance")
    for region, binding in zip(sequence, plan.instance_bindings, strict=True):
        if binding.region is not region:
            raise ValueError(
                "instance binding order does not match partition order"
            )
        if binding.template_id != plan.region_to_template_id[region]:
            raise ValueError("instance binding has an inconsistent template id")
        if binding.ordered_inputs != tuple(region.interface.ordered_inputs):
            raise ValueError(
                "instance binding does not preserve ordered inputs"
            )
        if binding.ordered_outputs != tuple(region.interface.ordered_outputs):
            raise ValueError(
                "instance binding does not preserve ordered outputs"
            )
        expected_parameter_indices = tuple(
            plan.parameter_indices[input_ref.value.op]
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.PARAMETER
        )
        if binding.parameter_indices != expected_parameter_indices:
            raise ValueError(
                "instance binding does not preserve parameter order"
            )
        expected_data_inputs = tuple(
            input_ref.value
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.DATA
        )
        if binding.data_inputs != expected_data_inputs:
            raise ValueError(
                "instance binding does not preserve data input order"
            )
        expected_state_inputs = tuple(
            input_ref.value
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.STATE
        )
        if binding.state_inputs != expected_state_inputs:
            raise ValueError(
                "instance binding does not preserve state input order"
            )

    region_set = set(sequence)
    grouped_regions = set()
    for group in plan.template_index.template_groups:
        if group.representative is not group.instances[0]:
            raise ValueError(
                "template representative is not its first instance"
            )
        for region in group.instances:
            if region not in region_set or region in grouped_regions:
                raise ValueError(
                    "template instances do not form a valid partition"
                )
            grouped_regions.add(region)
    non_reusable = set(plan.template_index.non_reusable_regions)
    if grouped_regions & non_reusable:
        raise ValueError("a Region is both reusable and non-reusable")
    layer_regions = {
        region for region in sequence if isinstance(region, LayerRegion)
    }
    if grouped_regions | non_reusable != layer_regions:
        raise ValueError(
            "template analysis does not classify every Layer Region"
        )

    if set(plan.region_to_template_id) != region_set:
        raise ValueError("not every Region maps to exactly one Template")
    expected_parameter_indices = {
        parameter: index for index, parameter in enumerate(graph.params)
    }
    if plan.parameter_indices != expected_parameter_indices:
        raise ValueError(
            "partition plan parameter indices do not match Graph order"
        )
    for unit in plan.templates:
        _validate_template_unit(unit)


def build_transformer_partition_plan(
    graph: Graph,
    deduplicate_templates: bool = True,
) -> TransformerPartitionPlan:
    """Build and verify the complete Transformer partition analysis plan."""
    structure_index, template_index = _detect_regions_and_build_interfaces(
        graph
    )
    materialization_plan = build_template_materialization_plan(
        graph,
        structure_index,
        template_index,
        deduplicate_templates=deduplicate_templates,
    )
    partition_sequence = _build_partition_sequence(structure_index)
    instance_bindings = _build_instance_bindings(
        structure_index, materialization_plan
    )
    plan = TransformerPartitionPlan(
        templates=materialization_plan.templates,
        region_to_template_id=materialization_plan.region_to_template_id,
        parameter_indices=materialization_plan.parameter_indices,
        structure_index=structure_index,
        template_index=template_index,
        instance_bindings=instance_bindings,
        partition_sequence=partition_sequence,
    )
    _verify_partition_plan(graph, plan)
    return plan


# Transformer tensor/sequence parallel planning (read-only Stage 1).


class ParallelPlanError(ValueError):
    """A graph property required for parallel planning cannot be proven."""


class ParameterRole(Enum):
    Q_WEIGHT = auto()
    Q_BIAS = auto()
    K_WEIGHT = auto()
    K_BIAS = auto()
    V_WEIGHT = auto()
    V_BIAS = auto()
    O_WEIGHT = auto()
    GATE_WEIGHT = auto()
    UP_WEIGHT = auto()
    DOWN_WEIGHT = auto()
    LM_HEAD_WEIGHT = auto()


class FeatureAxis(Enum):
    INPUT_FEATURE = auto()
    OUTPUT_FEATURE = auto()


class LayoutKind(Enum):
    REPLICATED = auto()
    SHARDED = auto()
    PARTIAL = auto()


class CollectiveKind(Enum):
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()
    ALL_GATHERV = auto()


class RewriteTarget(Enum):
    NEW_SHAPE = auto()
    OPERAND = auto()


@dataclass(frozen=True)
class RankSlice:
    rank: int
    offset: int
    size: int


@dataclass(frozen=True)
class TensorLayout:
    kind: LayoutKind
    shard_axis: int | None = None

    def __post_init__(self) -> None:
        if self.kind is LayoutKind.SHARDED:
            if not isinstance(self.shard_axis, int):
                raise ParallelPlanError(
                    "SHARDED layout requires an integer shard axis"
                )
        elif self.shard_axis is not None:
            raise ParallelPlanError(
                f"{self.kind.name} layout must not specify a shard axis"
            )


@dataclass(frozen=True)
class OperandUseRef:
    consumer: Op
    operand_path: tuple[int | str, ...]


@dataclass(frozen=True)
class ParameterShardSpec:
    parameter: GraphValueRef
    global_parameter_index: int
    template_parameter_slot: int
    role: ParameterRole
    semantic_axis: FeatureAxis
    consumer_use: OperandUseRef
    consumer_shard_axis: int
    storage_shard_axis: int
    storage_to_consumer_permutation: tuple[int, ...]
    global_shape: tuple[int, ...]
    rank_slices: tuple[RankSlice, ...]


@dataclass(frozen=True)
class ValueLayoutSpec:
    value: GraphValueRef
    global_shape: tuple[int, ...]
    local_shapes: tuple[tuple[int, ...], ...]
    layout: TensorLayout


@dataclass(frozen=True)
class OpRewriteSpec:
    op: Op
    target: RewriteTarget
    operand_path: tuple[int | str, ...] | None
    rank_values: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CollectiveBoundary:
    producer: GraphValueRef
    consumers: tuple[OperandUseRef, ...]
    kind: CollectiveKind
    target_layout: TensorLayout
    target_local_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ComputeSegment:
    segment_index: int
    ordered_nodes: tuple[Op, ...]
    ordered_inputs: tuple[GraphValueRef, ...]
    ordered_outputs: tuple[GraphValueRef, ...]


@dataclass(frozen=True)
class TemplateParallelPlan:
    template_id: int
    parameter_shards: tuple[ParameterShardSpec, ...]
    value_layouts: tuple[ValueLayoutSpec, ...]
    op_rewrites: tuple[OpRewriteSpec, ...]
    collectives: tuple[CollectiveBoundary, ...]
    segments: tuple[ComputeSegment, ...]


@dataclass(frozen=True)
class TransformerParallelPlan:
    graph_name: str
    world_size: int
    templates: tuple[TemplateParallelPlan, ...]


@dataclass(frozen=True)
class TransformerParallelConfig:
    tp_size: int = 2
    prefill_sequence_parallel: bool = False
    parallel_lm_head: bool = False


_REPLICATED = TensorLayout(LayoutKind.REPLICATED)
_PARTIAL = TensorLayout(LayoutKind.PARTIAL)


def partition_extent(extent: int, world_size: int) -> tuple[RankSlice, ...]:
    if extent <= 0 or world_size <= 0:
        raise ParallelPlanError(
            "partition extent and world size must be positive"
        )
    base, remainder = divmod(extent, world_size)
    offset = 0
    slices = []
    for rank in range(world_size):
        size = base + (1 if rank < remainder else 0)
        slices.append(RankSlice(rank, offset, size))
        offset += size
    return tuple(slices)


def _global_shape(value: GraphValueRef) -> tuple[int, ...]:
    shape = tuple(graph_value_tensor_meta(value).shape)
    if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ParallelPlanError(
            f"{value.op.name!r} result {value.result_index} has a non-static shape"
        )
    return shape


def _result_shapes(op: Op) -> tuple[tuple[int, ...], ...]:
    meta = op.tensor_meta
    shape = meta.shape if isinstance(meta, TensorMeta) else meta.get("shape")
    if shape is None:
        raise ParallelPlanError(f"operation {op.name!r} has no result shape")
    if (
        isinstance(shape, (list, tuple))
        and shape
        and isinstance(shape[0], (list, tuple))
    ):
        return tuple(tuple(item) for item in shape)
    return (tuple(shape),)


def _rank_shapes(
    shape: tuple[int, ...], axis: int, slices: tuple[RankSlice, ...]
) -> tuple[tuple[int, ...], ...]:
    result = []
    for rank_slice in slices:
        local = list(shape)
        local[axis] = rank_slice.size
        result.append(tuple(local))
    return tuple(result)


def _linear_semantics(op: Op) -> tuple[int, int, int, int | None]:
    """Return activation, weight, input-feature axis and optional bias slot.

    The feature axes describe the actual mathematical weight operand, before
    any parameter-side permutation is mapped back to Graph storage.
    """
    if isinstance(op, operation.AddMMOp):
        return 1, 2, 0, 0
    if isinstance(op, operation.MatmulOp):
        return 0, 1, 0, None
    if isinstance(op, operation.TransposeMatmulFusedOp):
        return 0, 1, 1, None
    raise ParallelPlanError(
        f"operation {op.name!r} is not a supported Linear-like compute op"
    )


def _permutation_for_transform(op: Op, rank: int) -> tuple[int, ...]:
    if isinstance(op, operation.PermuteOp):
        if len(op.args) < 2 or not isinstance(op.args[1], (list, tuple)):
            raise ParallelPlanError(
                f"Permute {op.name!r} has no static permutation"
            )
        permutation = tuple(int(item) for item in op.args[1])
    elif isinstance(op, operation.TransposeOp):
        if len(op.args) < 3:
            raise ParallelPlanError(
                f"Transpose {op.name!r} has no static dimensions"
            )
        dim0, dim1 = int(op.args[1]), int(op.args[2])
        dim0 %= rank
        dim1 %= rank
        values = list(range(rank))
        values[dim0], values[dim1] = values[dim1], values[dim0]
        permutation = tuple(values)
    else:
        raise ParallelPlanError(f"{op.name!r} is not an axis-only transform")
    if sorted(permutation) != list(range(rank)):
        raise ParallelPlanError(
            f"operation {op.name!r} has invalid permutation {permutation}"
        )
    return permutation


def _trace_parameter_operand(
    graph: Graph, consumer: Op, operand_slot: int
) -> tuple[GraphValueRef, tuple[int | str, ...], tuple[int, ...]]:
    uses = [
        (value, path)
        for value, path in iter_op_input_references(
            consumer, graph.node_table, with_paths=True
        )
        if path[:2] == ("args", operand_slot)
    ]
    if len(uses) != 1:
        raise ParallelPlanError(
            f"Linear operand args[{operand_slot}] of {consumer.name!r} does not "
            "contain exactly one tensor reference"
        )
    value, consumer_path = uses[0]
    rank = len(_global_shape(value))
    consumer_to_storage = tuple(range(rank))
    while value.op not in graph.params:
        transform = value.op
        if not isinstance(
            transform, (operation.PermuteOp, operation.TransposeOp)
        ):
            raise ParallelPlanError(
                f"unsupported parameter-side {type(transform).__name__} "
                f"{transform.name!r} before Linear {consumer.name!r}"
            )
        permutation = _permutation_for_transform(transform, rank)
        consumer_to_storage = tuple(
            permutation[axis] for axis in consumer_to_storage
        )
        inputs = list(iter_op_input_references(transform, graph.node_table))
        if len(inputs) != 1:
            raise ParallelPlanError(
                f"parameter transform {transform.name!r} is not unary"
            )
        value = inputs[0]
        if len(_global_shape(value)) != rank:
            raise ParallelPlanError(
                f"parameter transform {transform.name!r} changes rank"
            )
    return value, consumer_path, consumer_to_storage


_PROJECTION_ROLES = {
    "q_proj": (
        ParameterRole.Q_WEIGHT,
        ParameterRole.Q_BIAS,
        FeatureAxis.OUTPUT_FEATURE,
    ),
    "k_proj": (
        ParameterRole.K_WEIGHT,
        ParameterRole.K_BIAS,
        FeatureAxis.OUTPUT_FEATURE,
    ),
    "v_proj": (
        ParameterRole.V_WEIGHT,
        ParameterRole.V_BIAS,
        FeatureAxis.OUTPUT_FEATURE,
    ),
    "o_proj": (
        ParameterRole.O_WEIGHT,
        None,
        FeatureAxis.INPUT_FEATURE,
    ),
    "gate_proj": (
        ParameterRole.GATE_WEIGHT,
        None,
        FeatureAxis.OUTPUT_FEATURE,
    ),
    "up_proj": (
        ParameterRole.UP_WEIGHT,
        None,
        FeatureAxis.OUTPUT_FEATURE,
    ),
    "down_proj": (
        ParameterRole.DOWN_WEIGHT,
        None,
        FeatureAxis.INPUT_FEATURE,
    ),
    "lm_head": (
        ParameterRole.LM_HEAD_WEIGHT,
        None,
        FeatureAxis.OUTPUT_FEATURE,
    ),
}


def _nodes_for_subcomponent(
    unit: TemplateUnit,
    partition_plan: TransformerPartitionPlan,
    subcomponent: str,
) -> tuple[Op, ...]:
    region = unit.representative
    if isinstance(region, LayerRegion):
        return tuple(region.subcomponent_nodes.get(subcomponent, ()))
    result = []
    for op in region.nodes:
        annotation = partition_plan.structure_index.annotations.get(
            op, NodeAnnotation()
        )
        if annotation.subcomponent == subcomponent or (
            subcomponent == "lm_head" and annotation.component == "lm_head"
        ):
            result.append(op)
    return tuple(result)


def _parameter_shards_for_template(
    graph: Graph,
    unit: TemplateUnit,
    partition_plan: TransformerPartitionPlan,
    config: TransformerParallelConfig,
) -> tuple[ParameterShardSpec, ...]:
    parameter_slots = {
        item.value.op: slot
        for slot, item in enumerate(
            ref
            for ref in unit.representative.interface.ordered_inputs
            if ref.kind is RegionInputKind.PARAMETER
        )
    }
    specs = []
    targets = list(_PROJECTION_ROLES)
    if not config.parallel_lm_head:
        targets.remove("lm_head")
    for subcomponent in targets:
        nodes = _nodes_for_subcomponent(unit, partition_plan, subcomponent)
        if not nodes:
            continue
        linears = [
            op
            for op in nodes
            if isinstance(
                op,
                (
                    operation.MatmulOp,
                    operation.AddMMOp,
                    operation.TransposeMatmulFusedOp,
                ),
            )
        ]
        if len(linears) != 1:
            raise ParallelPlanError(
                f"subcomponent {subcomponent!r} in template {unit.template_id} "
                f"contains {len(linears)} supported Linear-like operations"
            )
        linear = linears[0]
        _, weight_slot, input_axis, bias_slot = _linear_semantics(linear)
        weight_role, bias_role, semantic_axis = _PROJECTION_ROLES[subcomponent]
        consumer_axis = (
            input_axis
            if semantic_axis is FeatureAxis.INPUT_FEATURE
            else 1 - input_axis
        )

        def add_spec(
            role, slot, axis, *, linear=linear, semantic_axis=semantic_axis
        ):
            parameter, path, consumer_to_storage = _trace_parameter_operand(
                graph, linear, slot
            )
            storage_shape = _global_shape(parameter)
            operand_values = [
                value
                for value, operand_path in iter_op_input_references(
                    linear, graph.node_table, with_paths=True
                )
                if operand_path == path
            ]
            if len(operand_values) != 1:
                raise ParallelPlanError(
                    f"cannot recover Linear operand {path!r} for {linear.name!r}"
                )
            consumer_shape = _global_shape(operand_values[0])
            expected = tuple(
                storage_shape[index] for index in consumer_to_storage
            )
            if consumer_shape != expected:
                raise ParallelPlanError(
                    f"parameter {parameter.op.name!r} permutation maps shape "
                    f"{storage_shape} to {expected}, not consumer shape "
                    f"{consumer_shape}"
                )
            storage_axis = consumer_to_storage[axis]
            inverse = [0] * len(consumer_to_storage)
            for consumer_index, storage_index in enumerate(consumer_to_storage):
                inverse[storage_index] = consumer_index
            try:
                parameter_slot = parameter_slots[parameter.op]
                parameter_index = partition_plan.parameter_indices[parameter.op]
            except KeyError as error:
                raise ParallelPlanError(
                    f"parameter {parameter.op.name!r} is not a template parameter"
                ) from error
            slices = partition_extent(
                storage_shape[storage_axis], config.tp_size
            )
            specs.append(
                ParameterShardSpec(
                    parameter=parameter,
                    global_parameter_index=parameter_index,
                    template_parameter_slot=parameter_slot,
                    role=role,
                    semantic_axis=semantic_axis,
                    consumer_use=OperandUseRef(linear, path),
                    consumer_shard_axis=axis,
                    storage_shard_axis=storage_axis,
                    storage_to_consumer_permutation=tuple(inverse),
                    global_shape=storage_shape,
                    rank_slices=slices,
                )
            )

        add_spec(weight_role, weight_slot, consumer_axis)
        if bias_role is not None and bias_slot is not None:
            add_spec(bias_role, bias_slot, 0)
    return tuple(specs)


def _prod(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _reshape_shard_axis(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], axis: int
) -> int | None:
    input_suffix = _prod(input_shape[axis:])
    candidates = [
        output_axis
        for output_axis in range(len(output_shape))
        if _prod(output_shape[output_axis:]) == input_suffix
    ]
    if not candidates:
        return None
    first, last = candidates[0], candidates[-1]
    if all(dim == 1 for dim in output_shape[first:last]):
        return last
    return candidates[0] if len(candidates) == 1 else None


def _broadcast_output_axis(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], axis: int
) -> int:
    output_axis = len(output_shape) - len(input_shape) + axis
    if output_axis < 0 or input_shape[axis] != output_shape[output_axis]:
        raise ParallelPlanError(
            f"cannot map sharded broadcast axis {axis} from {input_shape} "
            f"to {output_shape}"
        )
    return output_axis


@dataclass(frozen=True)
class _ResolvedUse:
    value: GraphValueRef
    use: OperandUseRef
    global_shape: tuple[int, ...]
    local_shapes: tuple[tuple[int, ...], ...]
    layout: TensorLayout


@dataclass(frozen=True)
class _PendingBoundary:
    producer: GraphValueRef
    consumer: OperandUseRef
    kind: CollectiveKind
    target_layout: TensorLayout
    target_local_shapes: tuple[tuple[int, ...], ...]


class _TemplateParallelPlanner:
    def __init__(
        self,
        graph: Graph,
        unit: TemplateUnit,
        partition_plan: TransformerPartitionPlan,
        config: TransformerParallelConfig,
    ) -> None:
        self.graph = graph
        self.unit = unit
        self.region = unit.representative
        self.partition_plan = partition_plan
        self.config = config
        self.world_size = config.tp_size
        self.local_shapes: dict[GraphValueRef, tuple[tuple[int, ...], ...]] = {}
        self.layouts: dict[GraphValueRef, TensorLayout] = {}
        self.rewrites: list[OpRewriteSpec] = []
        self.pending: list[_PendingBoundary] = []
        self.positions: dict[Op, int] = {}
        self.op_inputs: dict[Op, tuple[GraphValueRef, ...]] = {}
        self.value_consumers: dict[GraphValueRef, list[Op]] = {}
        self.parameter_shards = _parameter_shards_for_template(
            graph, unit, partition_plan, config
        )
        self._required_replicated: set[Op] = set()

    @property
    def is_prefill_sp(self) -> bool:
        return self.config.prefill_sequence_parallel and (
            "prefill" in self.graph._func_name.lower()
        )

    def _seed(
        self,
        value: GraphValueRef,
        layout: TensorLayout,
        local_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        shape = _global_shape(value)
        self.layouts[value] = layout
        self.local_shapes[value] = local_shapes or (shape,) * self.world_size

    def _prepare_sp_policy(self) -> None:
        if not self.is_prefill_sp:
            return
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            nodes = _nodes_for_subcomponent(
                self.unit, self.partition_plan, name
            )
            if nodes:
                self._required_replicated.add(nodes[0])
        if self.config.parallel_lm_head:
            nodes = _nodes_for_subcomponent(
                self.unit, self.partition_plan, "lm_head"
            )
            if nodes:
                self._required_replicated.add(nodes[0])

    def _seed_inputs(self) -> None:
        sharded_parameters = {
            spec.parameter: spec for spec in self.parameter_shards
        }
        for input_ref in self.region.interface.ordered_inputs:
            value = input_ref.value
            spec = sharded_parameters.get(value)
            if spec is not None:
                self._seed(
                    value,
                    TensorLayout(LayoutKind.SHARDED, spec.storage_shard_axis),
                    _rank_shapes(
                        spec.global_shape,
                        spec.storage_shard_axis,
                        spec.rank_slices,
                    ),
                )
            else:
                self._seed(value, _REPLICATED)

        if not self.is_prefill_sp:
            return
        seed_nodes = []
        for name in ("input_layernorm", "final_norm"):
            seed_nodes.extend(
                _nodes_for_subcomponent(self.unit, self.partition_plan, name)
            )
        internal = set(self.region.nodes)
        for op in seed_nodes:
            for value in iter_op_input_references(op, self.graph.node_table):
                if value.op in internal:
                    continue
                shape = _global_shape(value)
                if len(shape) < 3:
                    continue
                axis = len(shape) - 2
                slices = partition_extent(shape[axis], self.world_size)
                self._seed(
                    value,
                    TensorLayout(LayoutKind.SHARDED, axis),
                    _rank_shapes(shape, axis, slices),
                )

    def _resolve_uses(self, op: Op) -> list[_ResolvedUse]:
        resolved = []
        input_values = []
        for value, path in iter_op_input_references(
            op, self.graph.node_table, with_paths=True
        ):
            input_values.append(value)
            self.value_consumers.setdefault(value, []).append(op)
            if value not in self.local_shapes or value not in self.layouts:
                raise ParallelPlanError(
                    f"input value {value.op.name!r} result {value.result_index} "
                    f"of operation {op.name!r} has no inferred layout/local shape"
                )
            use = OperandUseRef(op, path)
            shape = _global_shape(value)
            local = self.local_shapes[value]
            layout = self.layouts[value]
            if (
                op in self._required_replicated
                and layout.kind is LayoutKind.SHARDED
                and value.op not in self.graph.params
            ):
                target_local = (shape,) * self.world_size
                self.pending.append(
                    _PendingBoundary(
                        value,
                        use,
                        CollectiveKind.ALL_GATHERV,
                        _REPLICATED,
                        target_local,
                    )
                )
                local, layout = target_local, _REPLICATED
            resolved.append(_ResolvedUse(value, use, shape, local, layout))
        self.op_inputs[op] = tuple(input_values)
        return resolved

    @staticmethod
    def _arg_use(uses: list[_ResolvedUse], index: int) -> _ResolvedUse | None:
        matches = [
            use for use in uses if use.use.operand_path[:2] == ("args", index)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ParallelPlanError(
                f"operand args[{index}] contains multiple tensors"
            )
        return matches[0]

    def _record_result(
        self,
        value: GraphValueRef,
        local_shapes: tuple[tuple[int, ...], ...],
        layout: TensorLayout,
    ) -> None:
        if len(local_shapes) != self.world_size:
            raise ParallelPlanError(
                f"operation {value.op.name!r} did not produce one shape per rank"
            )
        global_shape = _global_shape(value)
        for local in local_shapes:
            if len(local) != len(global_shape):
                raise ParallelPlanError(
                    f"local rank mismatch for {value.op.name!r}: "
                    f"global={global_shape}, local={local}"
                )
        self.local_shapes[value] = local_shapes
        self.layouts[value] = layout

    def _materialize_partial(
        self,
        partial: _ResolvedUse,
        target: _ResolvedUse | None,
    ) -> _ResolvedUse:
        if target is None or target.layout.kind is LayoutKind.REPLICATED:
            layout = _REPLICATED
            local_shapes = (partial.global_shape,) * self.world_size
            kind = CollectiveKind.ALL_REDUCE
        elif target.layout.kind is LayoutKind.SHARDED:
            layout = target.layout
            local_shapes = target.local_shapes
            kind = CollectiveKind.REDUCE_SCATTER
        else:
            raise ParallelPlanError(
                f"binary {partial.use.consumer.name!r} has two PARTIAL operands"
            )
        self.pending.append(
            _PendingBoundary(
                partial.value,
                partial.use,
                kind,
                layout,
                local_shapes,
            )
        )
        return _ResolvedUse(
            partial.value,
            partial.use,
            partial.global_shape,
            local_shapes,
            layout,
        )

    def _infer_linear(self, op: Op, uses: list[_ResolvedUse]) -> None:
        activation_slot, weight_slot, input_axis, bias_slot = _linear_semantics(
            op
        )
        activation = self._arg_use(uses, activation_slot)
        weight = self._arg_use(uses, weight_slot)
        if activation is None or weight is None:
            raise ParallelPlanError(
                f"Linear {op.name!r} has missing tensor operands"
            )
        output_value = GraphValueRef(op)
        output_shape = _global_shape(output_value)
        output_axis = 1 - input_axis
        if weight.layout.kind is LayoutKind.SHARDED:
            if weight.layout.shard_axis == output_axis:
                local_shapes = []
                for rank in range(self.world_size):
                    local = list(output_shape)
                    local[-1] = weight.local_shapes[rank][output_axis]
                    local_shapes.append(tuple(local))
                layout = TensorLayout(LayoutKind.SHARDED, len(output_shape) - 1)
                if bias_slot is not None:
                    bias = self._arg_use(uses, bias_slot)
                    if (
                        bias is None
                        or bias.layout.kind is not LayoutKind.SHARDED
                    ):
                        raise ParallelPlanError(
                            f"column-parallel AddMM {op.name!r} requires sharded bias"
                        )
                self._record_result(output_value, tuple(local_shapes), layout)
                return
            if weight.layout.shard_axis == input_axis:
                if (
                    activation.layout.kind is not LayoutKind.SHARDED
                    or activation.layout.shard_axis
                    != len(activation.global_shape) - 1
                ):
                    raise ParallelPlanError(
                        f"row-parallel Linear {op.name!r} requires its activation "
                        "contracting feature to be sharded"
                    )
                for rank in range(self.world_size):
                    if (
                        activation.local_shapes[rank][-1]
                        != (weight.local_shapes[rank][input_axis])
                    ):
                        raise ParallelPlanError(
                            f"row-parallel Linear {op.name!r} has incompatible "
                            "activation and weight contracting dimensions"
                        )
                self._record_result(
                    output_value,
                    (output_shape,) * self.world_size,
                    _PARTIAL,
                )
                return
            raise ParallelPlanError(
                f"Linear {op.name!r} weight is sharded on non-feature axis"
            )
        if (
            weight.layout.kind is LayoutKind.REPLICATED
            and activation.layout.kind is LayoutKind.REPLICATED
        ):
            self._record_result(
                output_value, (output_shape,) * self.world_size, _REPLICATED
            )
            return
        raise ParallelPlanError(
            f"unsupported Linear layouts for {op.name!r}: "
            f"activation={activation.layout}, weight={weight.layout}"
        )

    def _infer_permute(self, op: Op, uses: list[_ResolvedUse]) -> None:
        source = self._arg_use(uses, 0)
        if source is None:
            raise ParallelPlanError(
                f"axis transform {op.name!r} has no tensor input"
            )
        permutation = _permutation_for_transform(op, len(source.global_shape))
        local = tuple(
            tuple(shape[index] for index in permutation)
            for shape in source.local_shapes
        )
        if source.layout.kind is LayoutKind.SHARDED:
            try:
                axis = permutation.index(source.layout.shard_axis)
            except ValueError as error:
                raise ParallelPlanError(
                    f"axis transform {op.name!r} loses sharded axis"
                ) from error
            layout = TensorLayout(LayoutKind.SHARDED, axis)
        else:
            layout = source.layout
        self._record_result(GraphValueRef(op), local, layout)

    def _infer_reshape_like(
        self, op: Op, uses: list[_ResolvedUse], rewrite: bool = False
    ) -> None:
        source = self._arg_use(uses, 0)
        if source is None:
            raise ParallelPlanError(
                f"reshape-like {op.name!r} has no tensor input"
            )
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        if source.layout.kind is LayoutKind.REPLICATED:
            local = (output_shape,) * self.world_size
            layout = _REPLICATED
        elif source.layout.kind is LayoutKind.PARTIAL:
            local = (output_shape,) * self.world_size
            layout = _PARTIAL
        else:
            axis = None
            if source.layout.shard_axis is not None:
                axis = _reshape_shard_axis(
                    source.global_shape,
                    output_shape,
                    source.layout.shard_axis,
                )
            if axis is None:
                raise ParallelPlanError(
                    f"cannot map sharded axis through reshape-like {op.name!r}: "
                    f"{source.global_shape} -> {output_shape}"
                )
            local_values = []
            for rank_shape in source.local_shapes:
                rank_output = list(output_shape)
                global_suffix = _prod(
                    source.global_shape[source.layout.shard_axis :]
                )
                local_suffix = _prod(rank_shape[source.layout.shard_axis :])
                inner = _prod(output_shape[axis + 1 :])
                if local_suffix % inner != 0 or global_suffix % inner != 0:
                    raise ParallelPlanError(
                        f"reshape-like {op.name!r} cannot express a rank-local axis"
                    )
                rank_output[axis] = local_suffix // inner
                rank_output = tuple(rank_output)
                if _prod(rank_output) != _prod(rank_shape):
                    raise ParallelPlanError(
                        f"reshape-like {op.name!r} changes rank-local element count"
                    )
                local_values.append(rank_output)
            local = tuple(local_values)
            layout = TensorLayout(LayoutKind.SHARDED, axis)
        self._record_result(output, local, layout)
        if rewrite:
            self.rewrites.append(
                OpRewriteSpec(op, RewriteTarget.NEW_SHAPE, None, local)
            )

    def _infer_expand(self, op: Op, uses: list[_ResolvedUse]) -> None:
        source = self._arg_use(uses, 0)
        if source is None:
            raise ParallelPlanError(f"Expand {op.name!r} has no tensor input")
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        if source.layout.kind is LayoutKind.PARTIAL:
            raise ParallelPlanError(
                f"Expand {op.name!r} cannot consume PARTIAL"
            )
        if source.layout.kind is LayoutKind.REPLICATED:
            local = (output_shape,) * self.world_size
            layout = _REPLICATED
        else:
            padded_axis = (
                len(output_shape)
                - len(source.global_shape)
                + source.layout.shard_axis
            )
            if padded_axis < 0:
                raise ParallelPlanError(
                    f"Expand {op.name!r} reduces input rank"
                )
            local_values = []
            for rank in range(self.world_size):
                rank_output = list(output_shape)
                original = source.global_shape[source.layout.shard_axis]
                target = output_shape[padded_axis]
                if target not in (original, 1):
                    raise ParallelPlanError(
                        f"Expand {op.name!r} broadcasts a sharded dimension"
                    )
                rank_output[padded_axis] = source.local_shapes[rank][
                    source.layout.shard_axis
                ]
                local_values.append(tuple(rank_output))
            local = tuple(local_values)
            layout = TensorLayout(LayoutKind.SHARDED, padded_axis)
        self._record_result(output, local, layout)
        self.rewrites.append(
            OpRewriteSpec(op, RewriteTarget.OPERAND, ("args", 1), local)
        )

    def _infer_unsqueeze(self, op: Op, uses: list[_ResolvedUse]) -> None:
        source = self._arg_use(uses, 0)
        if (
            source is None
            or len(op.args) < 2
            or not isinstance(op.args[1], int)
        ):
            raise ParallelPlanError(f"Unsqueeze {op.name!r} has no static axis")
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        axis = op.args[1] % len(output_shape)
        expected = list(source.global_shape)
        expected.insert(axis, 1)
        if tuple(expected) != output_shape:
            raise ParallelPlanError(
                f"Unsqueeze {op.name!r} shape is inconsistent with axis {axis}"
            )
        local = []
        for rank_shape in source.local_shapes:
            rank_output = list(rank_shape)
            rank_output.insert(axis, 1)
            local.append(tuple(rank_output))
        if source.layout.kind is LayoutKind.SHARDED:
            shard_axis = source.layout.shard_axis + (
                1 if axis <= source.layout.shard_axis else 0
            )
            layout = TensorLayout(LayoutKind.SHARDED, shard_axis)
        else:
            layout = source.layout
        self._record_result(output, tuple(local), layout)

    def _infer_binary(self, op: Op, uses: list[_ResolvedUse]) -> None:
        tensors = [
            use
            for use in uses
            if use.use.operand_path[:2] in (("args", 0), ("args", 1))
        ]
        if not tensors:
            raise ParallelPlanError(
                f"binary operation {op.name!r} has no tensor input"
            )
        if len(tensors) > 2:
            raise ParallelPlanError(
                f"binary operation {op.name!r} has too many inputs"
            )
        if len(tensors) == 2:
            first, second = tensors
            if first.layout.kind is LayoutKind.PARTIAL:
                first = self._materialize_partial(first, second)
            if second.layout.kind is LayoutKind.PARTIAL:
                second = self._materialize_partial(second, first)
            tensors = [first, second]
        elif tensors[0].layout.kind is LayoutKind.PARTIAL:
            tensors[0] = self._materialize_partial(tensors[0], None)

        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        sharded = [
            use for use in tensors if use.layout.kind is LayoutKind.SHARDED
        ]
        if not sharded:
            layout = _REPLICATED
            local = (output_shape,) * self.world_size
        else:
            axes = {
                _broadcast_output_axis(
                    use.global_shape, output_shape, use.layout.shard_axis
                )
                for use in sharded
            }
            if len(axes) != 1:
                raise ParallelPlanError(
                    f"binary operation {op.name!r} has incompatible sharded axes"
                )
            axis = axes.pop()
            local_values = []
            for rank in range(self.world_size):
                extents = {
                    use.local_shapes[rank][use.layout.shard_axis]
                    for use in sharded
                }
                if len(extents) != 1:
                    raise ParallelPlanError(
                        f"binary operation {op.name!r} has incompatible local shapes"
                    )
                rank_output = list(output_shape)
                rank_output[axis] = extents.pop()
                local_values.append(tuple(rank_output))
            layout = TensorLayout(LayoutKind.SHARDED, axis)
            local = tuple(local_values)
        self._record_result(output, local, layout)

    def _infer_reduce(self, op: Op, uses: list[_ResolvedUse]) -> None:
        source = self._arg_use(uses, 0)
        if source is None:
            raise ParallelPlanError(
                f"reduction {op.name!r} has no tensor input"
            )
        if source.layout.kind is LayoutKind.PARTIAL:
            raise ParallelPlanError(
                f"reduction {op.name!r} cannot consume PARTIAL"
            )
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        if source.layout.kind is LayoutKind.REPLICATED:
            self._record_result(
                output, (output_shape,) * self.world_size, _REPLICATED
            )
            return
        axes_value = op.args[1] if len(op.args) > 1 else None
        if isinstance(axes_value, int):
            axes = (axes_value % len(source.global_shape),)
        elif isinstance(axes_value, (list, tuple)):
            axes = tuple(
                int(axis) % len(source.global_shape) for axis in axes_value
            )
        else:
            raise ParallelPlanError(f"reduction {op.name!r} has no static axes")
        if source.layout.shard_axis in axes:
            raise ParallelPlanError(
                f"reduction {op.name!r} reduces sharded axis {source.layout.shard_axis}"
            )
        keepdim = bool(op.args[2]) if len(op.args) > 2 else False
        output_axis = source.layout.shard_axis
        if not keepdim:
            output_axis -= sum(axis < output_axis for axis in axes)
        local_values = []
        for rank in range(self.world_size):
            rank_output = list(output_shape)
            rank_output[output_axis] = source.local_shapes[rank][
                source.layout.shard_axis
            ]
            local_values.append(tuple(rank_output))
        self._record_result(
            output,
            tuple(local_values),
            TensorLayout(LayoutKind.SHARDED, output_axis),
        )

    def _infer_slice(self, op: Op, uses: list[_ResolvedUse]) -> None:
        source = self._arg_use(uses, 0)
        if source is None:
            raise ParallelPlanError(f"Slice {op.name!r} has no tensor input")
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        if source.layout.kind is not LayoutKind.SHARDED:
            self._record_result(
                output, (output_shape,) * self.world_size, source.layout
            )
            return
        if len(op.args) < 2 or not isinstance(op.args[1], int):
            raise ParallelPlanError(f"Slice {op.name!r} has no static axis")
        slice_axis = op.args[1] % len(source.global_shape)
        if slice_axis == source.layout.shard_axis:
            raise ParallelPlanError(
                f"Slice {op.name!r} slices sharded axis {slice_axis}"
            )
        local_values = []
        for rank in range(self.world_size):
            rank_output = list(output_shape)
            rank_output[source.layout.shard_axis] = source.local_shapes[rank][
                source.layout.shard_axis
            ]
            local_values.append(tuple(rank_output))
        self._record_result(output, tuple(local_values), source.layout)

    def _infer_cat(self, op: Op, uses: list[_ResolvedUse]) -> None:
        tensors = [
            use for use in uses if use.use.operand_path[:2] == ("args", 0)
        ]
        if not tensors:
            raise ParallelPlanError(f"Cat {op.name!r} has no tensor inputs")
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        kinds = {use.layout.kind for use in tensors}
        if kinds == {LayoutKind.REPLICATED}:
            self._record_result(
                output, (output_shape,) * self.world_size, _REPLICATED
            )
            return
        if LayoutKind.PARTIAL in kinds:
            raise ParallelPlanError(f"Cat {op.name!r} cannot consume PARTIAL")
        if kinds != {LayoutKind.SHARDED}:
            raise ParallelPlanError(
                f"Cat {op.name!r} cannot mix REPLICATED and SHARDED inputs"
            )
        axes = {use.layout.shard_axis for use in tensors}
        if len(axes) != 1:
            raise ParallelPlanError(f"Cat {op.name!r} has incompatible layouts")
        shard_axis = axes.pop()
        cat_axis_value = op.args[1] if len(op.args) > 1 else 0
        cat_axis = int(cat_axis_value) % len(output_shape)
        if cat_axis == shard_axis:
            raise ParallelPlanError(
                f"Cat {op.name!r} concatenates sharded axis"
            )
        local_values = []
        for rank in range(self.world_size):
            non_cat_shapes = {
                tuple(
                    dim
                    for axis, dim in enumerate(use.local_shapes[rank])
                    if axis != cat_axis
                )
                for use in tensors
            }
            if len(non_cat_shapes) != 1:
                raise ParallelPlanError(
                    f"Cat {op.name!r} has incompatible local shapes"
                )
            rank_output = list(output_shape)
            rank_output[shard_axis] = tensors[0].local_shapes[rank][shard_axis]
            local_values.append(tuple(rank_output))
        self._record_result(
            output,
            tuple(local_values),
            TensorLayout(LayoutKind.SHARDED, shard_axis),
        )

    def _infer_index_put(self, op: Op, uses: list[_ResolvedUse]) -> None:
        target = self._arg_use(uses, 0)
        update = self._arg_use(uses, 2)
        if target is None or update is None:
            raise ParallelPlanError(
                f"IndexPut {op.name!r} has missing operands"
            )
        output = GraphValueRef(op)
        output_shape = _global_shape(output)
        if update.layout.kind is LayoutKind.PARTIAL:
            raise ParallelPlanError(
                f"IndexPut {op.name!r} cannot store PARTIAL"
            )
        if update.layout.kind is LayoutKind.REPLICATED:
            if target.layout.kind is not LayoutKind.REPLICATED:
                raise ParallelPlanError(
                    f"IndexPut {op.name!r} mixes incompatible layouts"
                )
            self._record_result(
                output, (output_shape,) * self.world_size, _REPLICATED
            )
            return
        axis = update.layout.shard_axis
        if len(update.global_shape) != len(output_shape):
            raise ParallelPlanError(
                f"IndexPut {op.name!r} update rank does not match its cache"
            )
        if update.global_shape[axis] != output_shape[axis]:
            raise ParallelPlanError(
                f"IndexPut {op.name!r} cannot map update shard axis to cache"
            )
        local_values = []
        for rank in range(self.world_size):
            rank_output = list(output_shape)
            rank_output[axis] = update.local_shapes[rank][axis]
            local_values.append(tuple(rank_output))
        local_shapes = tuple(local_values)
        layout = TensorLayout(LayoutKind.SHARDED, axis)
        if target.global_shape != output_shape:
            raise ParallelPlanError(
                f"IndexPut {op.name!r} output shape does not match its cache"
            )
        if target.layout.kind is LayoutKind.REPLICATED:
            region_inputs = {
                input_ref.value
                for input_ref in self.region.interface.ordered_inputs
            }
            if target.value not in region_inputs:
                raise ParallelPlanError(
                    f"IndexPut {op.name!r} cannot shard an internal replicated cache"
                )
            self.local_shapes[target.value] = local_shapes
            self.layouts[target.value] = layout
        elif target.layout != layout or target.local_shapes != local_shapes:
            raise ParallelPlanError(
                f"IndexPut {op.name!r} cache and update layouts do not match"
            )
        self._record_result(
            output,
            local_shapes,
            layout,
        )

    def _infer_passthrough(self, op: Op, uses: list[_ResolvedUse]) -> None:
        affected = [
            use
            for use in uses
            if use.layout.kind is not LayoutKind.REPLICATED
            or any(shape != use.global_shape for shape in use.local_shapes)
        ]
        result_shapes = _result_shapes(op)
        if not affected:
            for result_index, shape in enumerate(result_shapes):
                self._record_result(
                    GraphValueRef(op, result_index),
                    (shape,) * self.world_size,
                    _REPLICATED,
                )
            return
        structural_partial = isinstance(op, operation.CloneOp)
        if (
            any(use.layout.kind is LayoutKind.PARTIAL for use in affected)
            and not structural_partial
        ):
            raise ParallelPlanError(
                f"non-structural {type(op).__name__} {op.name!r} cannot consume PARTIAL"
            )
        if not structural_partial and op._op_type not in (
            operation.OpType.ElementwiseType,
            operation.OpType.GetItemType,
        ):
            raise ParallelPlanError(
                f"unsupported parallel-sensitive {type(op).__name__} {op.name!r}"
            )
        for result_index, shape in enumerate(result_shapes):
            matches = [use for use in affected if use.global_shape == shape]
            if not matches:
                prefix_matches = [
                    use
                    for use in affected
                    if len(use.global_shape) > len(shape)
                    and use.global_shape[: len(shape)] == shape
                    and (
                        use.layout.kind is not LayoutKind.SHARDED
                        or use.layout.shard_axis < len(shape)
                    )
                ]
                prefix_layouts = {use.layout for use in prefix_matches}
                prefix_locals = {
                    tuple(
                        rank_shape[: len(shape)]
                        for rank_shape in use.local_shapes
                    )
                    for use in prefix_matches
                }
                if (
                    prefix_matches
                    and len(prefix_layouts) == 1
                    and len(prefix_locals) == 1
                ):
                    self._record_result(
                        GraphValueRef(op, result_index),
                        prefix_locals.pop(),
                        prefix_layouts.pop(),
                    )
                    continue
                raise ParallelPlanError(
                    f"cannot prove {type(op).__name__} {op.name!r} result "
                    f"{result_index} is shape-preserving"
                )
            layouts = {use.layout for use in matches}
            local_shapes = {use.local_shapes for use in matches}
            if len(layouts) != 1 or len(local_shapes) != 1:
                raise ParallelPlanError(
                    f"shape-preserving {op.name!r} has incompatible parallel inputs"
                )
            self._record_result(
                GraphValueRef(op, result_index),
                local_shapes.pop(),
                layouts.pop(),
            )

    def _infer_op(self, op: Op, uses: list[_ResolvedUse]) -> None:
        if isinstance(
            op,
            (
                operation.MatmulOp,
                operation.AddMMOp,
                operation.TransposeMatmulFusedOp,
            ),
        ):
            self._infer_linear(op, uses)
        elif isinstance(op, (operation.PermuteOp, operation.TransposeOp)):
            self._infer_permute(op, uses)
        elif isinstance(op, (operation.ViewOp, operation.ReshapeOp)):
            self._infer_reshape_like(op, uses, rewrite=True)
        elif isinstance(op, operation.ExpandOp):
            self._infer_expand(op, uses)
        elif isinstance(op, operation.UnsqueezeOp):
            self._infer_unsqueeze(op, uses)
        elif op._op_type is operation.OpType.BroadcastType:
            self._infer_binary(op, uses)
        elif isinstance(op, (operation.MeanOp, operation.SumDimOp)):
            self._infer_reduce(op, uses)
        elif isinstance(op, operation.SliceOp):
            self._infer_slice(op, uses)
        elif isinstance(op, operation.CatOp):
            self._infer_cat(op, uses)
        elif isinstance(op, operation.IndexPutOp):
            self._infer_index_put(op, uses)
        elif op._op_type is operation.OpType.ReshapeType:
            self._infer_reshape_like(op, uses)
        else:
            self._infer_passthrough(op, uses)

    def _group_boundaries(self) -> tuple[CollectiveBoundary, ...]:
        grouped: dict[
            tuple[
                GraphValueRef,
                CollectiveKind,
                TensorLayout,
                tuple[tuple[int, ...], ...],
            ],
            list[OperandUseRef],
        ] = {}
        for boundary in self.pending:
            key = (
                boundary.producer,
                boundary.kind,
                boundary.target_layout,
                boundary.target_local_shapes,
            )
            grouped.setdefault(key, []).append(boundary.consumer)
        return tuple(
            CollectiveBoundary(
                producer=key[0],
                consumers=tuple(consumers),
                kind=key[1],
                target_layout=key[2],
                target_local_shapes=key[3],
            )
            for key, consumers in grouped.items()
        )

    def _segments(
        self, collectives: tuple[CollectiveBoundary, ...]
    ) -> tuple[ComputeSegment, ...]:
        nodes = tuple(self.region.nodes)
        cut_positions = sorted(
            {
                min(
                    self.positions[consumer.consumer]
                    for consumer in boundary.consumers
                )
                for boundary in collectives
            }
        )
        effective_cuts = [
            position for position in cut_positions if position > 0
        ]

        def segment_at(position: int) -> int:
            return sum(cut <= position for cut in effective_cuts)

        op_to_segment = {
            op: segment_at(position) for op, position in self.positions.items()
        }
        region_outputs = set(self.region.interface.ordered_outputs)
        collective_producers = {boundary.producer for boundary in collectives}
        count = len(effective_cuts) + 1
        segment_nodes = [[] for _ in range(count)]
        segment_inputs = [[] for _ in range(count)]
        segment_outputs = [[] for _ in range(count)]
        seen_inputs = [set() for _ in range(count)]
        seen_outputs = [set() for _ in range(count)]

        # Pass 2: one forward traversal; use-def was captured by Pass 1.
        for position, op in enumerate(nodes):
            segment_index = segment_at(position)
            segment_nodes[segment_index].append(op)
            for value in self.op_inputs[op]:
                if (
                    op_to_segment.get(value.op) == segment_index
                    or value in seen_inputs[segment_index]
                ):
                    continue
                seen_inputs[segment_index].add(value)
                segment_inputs[segment_index].append(value)
            for result_index in range(len(_result_shapes(op))):
                value = GraphValueRef(op, result_index)
                has_external_user = any(
                    op_to_segment[consumer] != segment_index
                    for consumer in self.value_consumers.get(value, ())
                )
                if (
                    has_external_user
                    or value in region_outputs
                    or value in collective_producers
                ) and value not in seen_outputs[segment_index]:
                    seen_outputs[segment_index].add(value)
                    segment_outputs[segment_index].append(value)

        segments = tuple(
            ComputeSegment(
                segment_index,
                tuple(segment_nodes[segment_index]),
                tuple(segment_inputs[segment_index]),
                tuple(segment_outputs[segment_index]),
            )
            for segment_index in range(count)
        )

        flattened = tuple(
            op for segment in segments for op in segment.ordered_nodes
        )
        if flattened != nodes or len(flattened) != len(set(flattened)):
            raise ParallelPlanError(
                f"template {self.unit.template_id} segment coverage/order is invalid"
            )
        for boundary in collectives:
            producer_segment = op_to_segment.get(boundary.producer.op)
            consumer_segments = {
                op_to_segment[consumer.consumer]
                for consumer in boundary.consumers
            }
            if len(consumer_segments) != 1:
                raise ParallelPlanError(
                    "grouped collective consumers do not share a downstream segment"
                )
            if (
                producer_segment is not None
                and producer_segment in consumer_segments
            ):
                raise ParallelPlanError(
                    f"collective producer {boundary.producer.op.name!r} remains "
                    "in its consumer segment"
                )
        for segment in segments:
            for value in segment.ordered_inputs + segment.ordered_outputs:
                if value not in self.local_shapes:
                    raise ParallelPlanError(
                        f"segment ABI value {value.op.name!r}:{value.result_index} "
                        "has no local shape"
                    )
                producer_position = self.positions.get(value.op)
                if (
                    value in segment.ordered_inputs
                    and producer_position is not None
                ):
                    if (
                        producer_position
                        >= self.positions[segment.ordered_nodes[0]]
                    ):
                        raise ParallelPlanError(
                            "segment input producer order is invalid"
                        )
        return segments

    def build(self) -> TemplateParallelPlan:
        self._prepare_sp_policy()
        self._seed_inputs()
        for position, op in enumerate(self.region.nodes):
            self.positions[op] = position
            uses = self._resolve_uses(op)
            self._infer_op(op, uses)
        collectives = self._group_boundaries()
        segments = self._segments(collectives)
        value_layouts = tuple(
            ValueLayoutSpec(
                value, _global_shape(value), local, self.layouts[value]
            )
            for value, local in self.local_shapes.items()
        )
        return TemplateParallelPlan(
            template_id=self.unit.template_id,
            parameter_shards=self.parameter_shards,
            value_layouts=value_layouts,
            op_rewrites=tuple(self.rewrites),
            collectives=collectives,
            segments=segments,
        )


def build_transformer_parallel_plan(
    graph: Graph,
    template_plan: TransformerPartitionPlan,
    config: TransformerParallelConfig,
) -> TransformerParallelPlan:
    """Build a read-only TP/SP plan over each unique representative template."""
    if config.tp_size != 2:
        raise ParallelPlanError(
            f"unsupported tp_size={config.tp_size}; Stage 1 supports only tp_size=2"
        )
    if template_plan.parameter_indices != {
        parameter: index for index, parameter in enumerate(graph.params)
    }:
        raise ParallelPlanError("partition plan does not belong to this Graph")
    templates = tuple(
        _TemplateParallelPlanner(graph, unit, template_plan, config).build()
        for unit in template_plan.templates
    )
    if tuple(plan.template_id for plan in templates) != tuple(
        unit.template_id for unit in template_plan.templates
    ):
        raise ParallelPlanError("parallel plan template order is invalid")
    return TransformerParallelPlan(graph._func_name, config.tp_size, templates)
