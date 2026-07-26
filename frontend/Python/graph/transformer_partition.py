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

from .operation import Op, OutputOp, TensorConstantOp
from .structure_analysis import ModuleStructureAnalyzer, NodeAnnotation
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
    return sorted(value.items(), key=lambda item: _stable_operand_dict_key(item[0]))


def _resolve_operand_node_reference(value, node_table: dict[str, Op]) -> Op | None:
    if isinstance(value, str):
        return node_table.get(value)
    return None


def _iter_operand_value_references(
    value, node_table: dict[str, Op], result_index: int = 0
):
    referenced = _resolve_operand_node_reference(value, node_table)
    if referenced is not None:
        yield GraphValueRef(referenced, result_index)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_operand_value_references(
                item, node_table, result_index
            )
    elif isinstance(value, dict):
        for key, item in _operand_dict_items(value):
            yield from _iter_operand_value_references(
                key, node_table, result_index
            )
            yield from _iter_operand_value_references(
                item, node_table, result_index
            )


def iter_op_input_references(op: Op, node_table: dict[str, Op]):
    for index, value in enumerate(op.args):
        result_index = op._args_index[index] if index < len(op._args_index) else 0
        yield from _iter_operand_value_references(
            value, node_table, result_index
        )
    for key, value in _operand_dict_items(op.kwargs):
        yield from _iter_operand_value_references(key, node_table)
        yield from _iter_operand_value_references(value, node_table)


class RegionBuilder:
    """Build a deterministic structural index without mutating its graph."""

    def __init__(self, graph: "Graph") -> None:
        self._graph = graph

    def build(
        self, template_recognizer: "TemplateRecognizer | None" = None
    ) -> GraphStructureIndex:
        graph = self._graph
        analyzer = ModuleStructureAnalyzer()
        annotations = analyzer.analyze(graph).node_annotations
        body_positions: dict[Op, int] = {}
        body_nodes: set[Op] = set()
        eligible: list[Op] = []
        layer_nodes: dict[int, list[Op]] = {}
        prelude_nodes: list[Op] = []
        epilogue_nodes: list[Op] = []
        unassigned: list[tuple[int, Op]] = []
        params = set(graph.params)
        excluded = set(graph.inputs) | params

        # Classifications are already complete, including topology refinements.
        # Retain enough ordered buckets to construct regions in one traversal.
        for position, op in enumerate(graph.body):
            body_positions[op] = position
            body_nodes.add(op)
            annotation = annotations.get(op, NodeAnnotation())
            if op in excluded or isinstance(op, (TensorConstantOp, OutputOp)):
                continue
            eligible.append(op)
            if annotation.layer_index is not None:
                layer_nodes.setdefault(annotation.layer_index, []).append(op)
            elif annotation.component == "embedding":
                prelude_nodes.append(op)
            elif (
                annotation.component == "lm_head"
                or annotation.subcomponent == "final_norm"
            ):
                epilogue_nodes.append(op)
            else:
                unassigned.append((position, op))

        eligible_set = set(eligible)

        regions: list[GraphRegion] = []
        node_to_region: dict[Op, GraphRegion] = {}
        for layer_index, nodes in layer_nodes.items():
            component_nodes: dict[str, list[Op]] = {}
            subcomponent_nodes: dict[str, list[Op]] = {}
            for op in nodes:
                annotation = annotations[op]
                if annotation.component is not None:
                    component_nodes.setdefault(annotation.component, []).append(op)
                if annotation.subcomponent is not None:
                    subcomponent_nodes.setdefault(annotation.subcomponent, []).append(
                        op
                    )
            region = LayerRegion(
                kind=RegionKind.LAYER,
                nodes=nodes,
                layer_index=layer_index,
                component_nodes=component_nodes,
                subcomponent_nodes=subcomponent_nodes,
            )
            self._assign(region, node_to_region)
            regions.append(region)

        layer_positions = [
            body_positions[op] for nodes in layer_nodes.values() for op in nodes
        ]
        first_layer = min(layer_positions) if layer_positions else None
        last_layer = max(layer_positions) if layer_positions else None

        if first_layer is not None and last_layer is not None:
            remaining = []
            for position, op in unassigned:
                if position < first_layer:
                    prelude_nodes.append(op)
                elif position > last_layer:
                    epilogue_nodes.append(op)
                else:
                    remaining.append((position, op))
            unassigned = remaining

        if prelude_nodes:
            prelude_nodes.sort(key=body_positions.__getitem__)
            region = GraphRegion(RegionKind.PRELUDE, prelude_nodes)
            self._assign(region, node_to_region)
            regions.append(region)
        if epilogue_nodes:
            epilogue_nodes.sort(key=body_positions.__getitem__)
            region = GraphRegion(RegionKind.EPILOGUE, epilogue_nodes)
            self._assign(region, node_to_region)
            regions.append(region)

        unknown_run: list[Op] = []
        previous_position: int | None = None

        def finish_unknown_run() -> None:
            if not unknown_run:
                return
            region = GraphRegion(RegionKind.UNKNOWN, list(unknown_run))
            self._assign(region, node_to_region)
            regions.append(region)
            unknown_run.clear()

        for position, op in unassigned:
            if previous_position is not None and position != previous_position + 1:
                finish_unknown_run()
            unknown_run.append(op)
            previous_position = position
        finish_unknown_run()

        regions.sort(key=lambda region: body_positions[region.nodes[0]])
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
            if template_recognizer is not None and isinstance(region, LayerRegion):
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
    def _assign(region: GraphRegion, node_to_region: dict[Op, GraphRegion]) -> None:
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
                and layer_resolution.layer_index == self._region.layer_index
            ):
                path = layer_resolution.canonical_module_path
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
                "nodes": self._resolve_node_refs(
                    self._nodes, external_slots
                ),
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
    graph, structure_index, template_index
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
                raise ValueError("template instance is not in structure regions")
            if region in group_for_region:
                raise ValueError("region belongs to more than one template group")
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
        group = group_for_region.get(region)
        if group is None:
            instances = (region,)
            representative = region
        else:
            group_key = id(group)
            if group_key in emitted_groups:
                continue
            emitted_groups.add(group_key)
            instances = tuple(item for item in regions if item in group.instances)
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
        if op not in excluded and not isinstance(op, (TensorConstantOp, OutputOp))
    }
    covered = [op for region in sequence for op in region.nodes]
    if len(covered) != len(set(covered)):
        raise ValueError("partition sequence contains a node more than once")
    if set(covered) != eligible:
        raise ValueError("partition sequence does not cover every eligible node")

    body_positions = {op: index for index, op in enumerate(graph.body)}
    previous_region_start = -1
    for region in sequence:
        positions = [body_positions[op] for op in region.nodes]
        if positions != sorted(positions):
            raise ValueError("Region nodes do not preserve graph.body order")
        if positions[0] < previous_region_start:
            raise ValueError("partition sequence does not preserve Region order")
        previous_region_start = positions[0]
        _validate_v1_region_interface(region)
        RegionBuilder._validate_interface(region, set(graph.body))

    if len(plan.instance_bindings) != len(sequence):
        raise ValueError("partition plan does not bind every Region instance")
    for region, binding in zip(sequence, plan.instance_bindings, strict=True):
        if binding.region is not region:
            raise ValueError("instance binding order does not match partition order")
        if binding.template_id != plan.region_to_template_id[region]:
            raise ValueError("instance binding has an inconsistent template id")
        if binding.ordered_inputs != tuple(region.interface.ordered_inputs):
            raise ValueError("instance binding does not preserve ordered inputs")
        if binding.ordered_outputs != tuple(region.interface.ordered_outputs):
            raise ValueError("instance binding does not preserve ordered outputs")
        expected_parameter_indices = tuple(
            plan.parameter_indices[input_ref.value.op]
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.PARAMETER
        )
        if binding.parameter_indices != expected_parameter_indices:
            raise ValueError("instance binding does not preserve parameter order")
        expected_data_inputs = tuple(
            input_ref.value
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.DATA
        )
        if binding.data_inputs != expected_data_inputs:
            raise ValueError("instance binding does not preserve data input order")
        expected_state_inputs = tuple(
            input_ref.value
            for input_ref in region.interface.ordered_inputs
            if input_ref.kind is RegionInputKind.STATE
        )
        if binding.state_inputs != expected_state_inputs:
            raise ValueError("instance binding does not preserve state input order")

    region_set = set(sequence)
    grouped_regions = set()
    for group in plan.template_index.template_groups:
        if group.representative is not group.instances[0]:
            raise ValueError("template representative is not its first instance")
        for region in group.instances:
            if region not in region_set or region in grouped_regions:
                raise ValueError("template instances do not form a valid partition")
            grouped_regions.add(region)
    non_reusable = set(plan.template_index.non_reusable_regions)
    if grouped_regions & non_reusable:
        raise ValueError("a Region is both reusable and non-reusable")
    layer_regions = {
        region for region in sequence if isinstance(region, LayerRegion)
    }
    if grouped_regions | non_reusable != layer_regions:
        raise ValueError("template analysis does not classify every Layer Region")

    if set(plan.region_to_template_id) != region_set:
        raise ValueError("not every Region maps to exactly one Template")
    expected_parameter_indices = {
        parameter: index for index, parameter in enumerate(graph.params)
    }
    if plan.parameter_indices != expected_parameter_indices:
        raise ValueError("partition plan parameter indices do not match Graph order")
    for unit in plan.templates:
        _validate_template_unit(unit)


def build_transformer_partition_plan(
    graph: Graph,
) -> TransformerPartitionPlan:
    """Build and verify the complete Transformer partition analysis plan."""
    structure_index, template_index = _detect_regions_and_build_interfaces(graph)
    materialization_plan = build_template_materialization_plan(
        graph, structure_index, template_index
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
