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


class RegionBuilder:
    """Build a deterministic structural index without mutating its graph."""

    def __init__(self, graph: "Graph") -> None:
        self._graph = graph

    def build(self) -> GraphStructureIndex:
        graph = self._graph
        snapshot = self._snapshot_graph()
        annotations = ModuleStructureAnalyzer().analyze(graph).node_annotations
        body_positions = {op: index for index, op in enumerate(graph.body)}
        eligible = self._eligible_nodes()
        eligible_set = set(eligible)

        layer_nodes: dict[int, list[Op]] = {}
        for op in eligible:
            annotation = annotations.get(op)
            if annotation is not None and annotation.layer_index is not None:
                layer_nodes.setdefault(annotation.layer_index, []).append(op)

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

        prelude_nodes: list[Op] = []
        epilogue_nodes: list[Op] = []
        for op in eligible:
            if op in node_to_region:
                continue
            annotation = annotations.get(op)
            if annotation is not None and annotation.component == "embedding":
                prelude_nodes.append(op)
            elif annotation is not None and (
                annotation.component == "lm_head"
                or annotation.subcomponent == "final_norm"
            ):
                epilogue_nodes.append(op)

        explicitly_assigned = set(prelude_nodes) | set(epilogue_nodes)
        if first_layer is not None and last_layer is not None:
            for op in eligible:
                if op in node_to_region or op in explicitly_assigned:
                    continue
                position = body_positions[op]
                if position < first_layer:
                    prelude_nodes.append(op)
                elif position > last_layer:
                    epilogue_nodes.append(op)

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

        def finish_unknown_run() -> None:
            if not unknown_run:
                return
            region = GraphRegion(RegionKind.UNKNOWN, list(unknown_run))
            self._assign(region, node_to_region)
            regions.append(region)
            unknown_run.clear()

        for op in graph.body:
            if op in eligible_set and op not in node_to_region:
                unknown_run.append(op)
            else:
                finish_unknown_run()
        finish_unknown_run()

        regions.sort(key=lambda region: body_positions[region.nodes[0]])
        for region in regions:
            region.interface = self._build_interface(region)

        index = GraphStructureIndex(annotations, regions, node_to_region)
        self._validate(index, eligible_set)
        self._validate_graph_unchanged(snapshot)
        return index

    def _eligible_nodes(self) -> list[Op]:
        excluded = set(self._graph.inputs) | set(self._graph.params)
        return [
            op
            for op in self._graph.body
            if op not in excluded and not isinstance(op, (TensorConstantOp, OutputOp))
        ]

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
    ) -> RegionInterface:
        interface = RegionInterface()
        region_nodes = set(region.nodes)
        params = set(self._graph.params)
        seen_inputs: set[Op] = set()

        for op in region.nodes:
            for parent_name in op.parents:
                parent = self._resolve_edge(op, parent_name, "parent")
                if parent in region_nodes or parent in seen_inputs:
                    continue
                seen_inputs.add(parent)
                if parent in params:
                    interface.parameters.append(parent)
                elif isinstance(parent, TensorConstantOp):
                    interface.constants.append(parent)
                else:
                    interface.data_inputs.append(parent)

        for op in region.nodes:
            has_external_child = False
            for child_name in op._children:
                child = self._resolve_edge(op, child_name, "child")
                if child not in region_nodes:
                    has_external_child = True
            if has_external_child:
                interface.data_outputs.append(op)

        return interface

    def _validate(self, index: GraphStructureIndex, eligible: set[Op]) -> None:
        body_set = set(self._graph.body)
        body_positions = {op: position for position, op in enumerate(self._graph.body)}
        interface_nodes = (
            set(self._graph.inputs)
            | set(self._graph.params)
            | {
                op
                for op in self._graph.body
                if isinstance(op, (TensorConstantOp, OutputOp))
            }
        )
        counts: dict[Op, int] = {}
        for region in index.regions:
            local_seen: set[Op] = set()
            for op in region.nodes:
                if op not in body_set:
                    raise ValueError(
                        f"region {region.kind.name} references node "
                        f"{op.name!r} that is not an original graph node"
                    )
                if op in local_seen:
                    raise ValueError(
                        f"region {region.kind.name} contains duplicate node "
                        f"{op.name!r}"
                    )
                local_seen.add(op)
                counts[op] = counts.get(op, 0) + 1
                if index.node_to_region.get(op) is not region:
                    raise ValueError(
                        f"node_to_region is inconsistent for node {op.name!r}"
                    )
            positions = [body_positions[op] for op in region.nodes]
            if positions != sorted(positions):
                raise ValueError(
                    f"region {region.kind.name} nodes do not preserve "
                    "graph.body order"
                )
            self._validate_interface(region, body_set)

            if isinstance(region, LayerRegion):
                for grouping_name, groups in (
                    ("component_nodes", region.component_nodes),
                    ("subcomponent_nodes", region.subcomponent_nodes),
                ):
                    for key, nodes in groups.items():
                        if any(node not in local_seen for node in nodes):
                            offending = next(
                                node for node in nodes if node not in local_seen
                            )
                            raise ValueError(
                                f"{grouping_name}[{key!r}] references node "
                                f"{offending.name!r} outside its LayerRegion"
                            )
                        group_positions = [body_positions[node] for node in nodes]
                        if group_positions != sorted(group_positions):
                            raise ValueError(
                                f"{grouping_name}[{key!r}] does not preserve "
                                "graph.body order"
                            )

        for op in eligible:
            count = counts.get(op, 0)
            if count != 1:
                raise ValueError(
                    f"region-eligible node {op.name!r} belongs to {count} "
                    "regions; expected exactly one"
                )
        for op in interface_nodes:
            if op in index.node_to_region:
                raise ValueError(
                    f"graph interface node {op.name!r} must not belong to a " "region"
                )
        for op, region in index.node_to_region.items():
            if op not in eligible:
                raise ValueError(
                    f"non-eligible node {op.name!r} appears in node_to_region"
                )
            if op not in region.nodes:
                raise ValueError(
                    f"node_to_region points node {op.name!r} at a region that "
                    "does not contain it"
                )

        region_positions = [body_positions[region.nodes[0]] for region in index.regions]
        if region_positions != sorted(region_positions):
            raise ValueError(
                "GraphStructureIndex.regions is not ordered by each region's "
                "first graph.body node"
            )

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

    def _snapshot_graph(self):
        graph = self._graph
        return (
            id(graph.body),
            tuple(id(op) for op in graph.body),
            id(graph.node_table),
            tuple((name, id(op)) for name, op in graph.node_table.items()),
            {
                op: (
                    id(op.parents),
                    tuple(op.parents),
                    id(op._children),
                    tuple(op._children),
                    id(op.args),
                    tuple(id(arg) for arg in op.args),
                )
                for op in graph.body
            },
        )

    def _validate_graph_unchanged(self, snapshot) -> None:
        if self._snapshot_graph() != snapshot:
            raise RuntimeError(
                "RegionBuilder modified graph.body, graph.node_table, or an "
                "operation's parents, children, or args"
            )
