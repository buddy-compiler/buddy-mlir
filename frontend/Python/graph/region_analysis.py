# ===- region_analysis.py -----------------------------------------------------
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

"""Structural regions over an existing Buddy graph.

Regions are descriptions only: every node reference is an operation already in
the source graph, and building the index does not rewrite the graph or create
subgraphs.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from .operation import Op, OutputOp, TensorConstantOp
from .structure_analysis import ModuleStructureAnalyzer, NodeAnnotation

if TYPE_CHECKING:
    from .graph import Graph
    from .template_analysis import TemplateRecognizer


class RegionKind(Enum):
    PRELUDE = auto()
    LAYER = auto()
    EPILOGUE = auto()
    UNKNOWN = auto()


@dataclass
class RegionInterface:
    data_inputs: list[Op] = field(default_factory=list)
    data_outputs: list[Op] = field(default_factory=list)
    parameters: list[Op] = field(default_factory=list)
    constants: list[Op] = field(default_factory=list)
    state_inputs: list[Op] = field(default_factory=list)
    state_outputs: list[Op] = field(default_factory=list)


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


def _iter_operand_node_references(value, node_table: dict[str, Op]):
    referenced = _resolve_operand_node_reference(value, node_table)
    if referenced is not None:
        yield referenced
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_operand_node_references(item, node_table)
    elif isinstance(value, dict):
        for key, item in _operand_dict_items(value):
            yield from _iter_operand_node_references(key, node_table)
            yield from _iter_operand_node_references(item, node_table)


class RegionBuilder:
    """Build a deterministic structural index without mutating its graph."""

    def __init__(self, graph: "Graph") -> None:
        self._graph = graph

    def build(
        self, template_recognizer: "TemplateRecognizer | None" = None
    ) -> GraphStructureIndex:
        graph = self._graph
        analyzer = ModuleStructureAnalyzer()
        annotations: dict[Op, NodeAnnotation] = {}
        body_positions: dict[Op, int] = {}
        body_nodes: set[Op] = set()
        eligible: list[Op] = []
        layer_nodes: dict[int, list[Op]] = {}
        prelude_nodes: list[Op] = []
        epilogue_nodes: list[Op] = []
        unassigned: list[tuple[int, Op]] = []
        params = set(graph.params)
        excluded = set(graph.inputs) | params

        # Pass 1: classify every graph node exactly once and retain enough
        # ordered buckets to construct regions without another body traversal.
        for position, op in enumerate(graph.body):
            body_positions[op] = position
            body_nodes.add(op)
            annotation = analyzer.analyze_node(op)
            if (
                annotation.layer_index is not None
                or annotation.component is not None
                or annotation.subcomponent is not None
            ):
                annotations[op] = annotation
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
        # Pass 2: each Region node is visited once for strict edge validation,
        # interface construction, and optional canonical fingerprint tokens.
        for region in regions:
            fingerprint_builder = None
            if template_recognizer is not None and isinstance(region, LayerRegion):
                fingerprint_builder = template_recognizer.make_builder(
                    graph, region, annotations, node_to_region, params
                )
            region.interface = self._build_interface(
                region,
                node_to_region,
                body_positions,
                params,
                fingerprint_builder,
            )
            self._validate_interface(region, body_nodes)
            if fingerprint_builder is not None:
                canonical, summary = fingerprint_builder.finish(region.interface)
                template_recognizer.add(region, canonical, summary)

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
        fingerprint_builder=None,
    ) -> RegionInterface:
        interface = RegionInterface()
        seen_inputs: set[Op] = set()
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
            for dependency in _iter_operand_node_references(
                op.args, self._graph.node_table
            ):
                if (
                    node_to_region.get(dependency) is region
                    or dependency in seen_inputs
                ):
                    continue
                seen_inputs.add(dependency)
                if dependency in params:
                    interface.parameters.append(dependency)
                elif isinstance(dependency, TensorConstantOp):
                    interface.constants.append(dependency)
                else:
                    interface.data_inputs.append(dependency)
            for dependency in _iter_operand_node_references(
                op.kwargs, self._graph.node_table
            ):
                if (
                    node_to_region.get(dependency) is region
                    or dependency in seen_inputs
                ):
                    continue
                seen_inputs.add(dependency)
                if dependency in params:
                    interface.parameters.append(dependency)
                elif isinstance(dependency, TensorConstantOp):
                    interface.constants.append(dependency)
                else:
                    interface.data_inputs.append(dependency)
            has_external_child = False
            for child_name in op._children:
                child = self._resolve_edge(op, child_name, "child")
                if node_to_region.get(child) is not region:
                    has_external_child = True
            if has_external_child:
                interface.data_outputs.append(op)
            if fingerprint_builder is not None:
                fingerprint_builder.consume(op)

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
            seen: set[Op] = set()
            for op in nodes:
                if op not in body_nodes:
                    raise ValueError(
                        f"region {region.kind.name} {category} references node "
                        f"{op.name!r} outside the original graph"
                    )
                if op in seen:
                    raise ValueError(
                        f"region {region.kind.name} {category} contains "
                        f"duplicate node {op.name!r}"
                    )
                seen.add(op)

        input_categories = (
            set(interface.data_inputs),
            set(interface.parameters),
            set(interface.constants),
        )
        if any(
            left & right
            for position, left in enumerate(input_categories)
            for right in input_categories[position + 1 :]
        ):
            raise ValueError(
                f"region {region.kind.name} classifies an external dependency "
                "in more than one input category"
            )
        for op in interface.data_outputs:
            if op not in region.nodes:
                raise ValueError(
                    f"region {region.kind.name} data output {op.name!r} is not "
                    "an internal producer"
                )
        if interface.state_inputs or interface.state_outputs:
            raise ValueError(
                f"region {region.kind.name} inferred state in a version-one "
                "structure index"
            )
