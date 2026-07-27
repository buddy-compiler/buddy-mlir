import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .graph import Graph
from .operation import (
    AddOp,
    MeanOp,
    MulOp,
    Op,
    PowOp,
    RsqrtOp,
)

if TYPE_CHECKING:
    from .transformer_partition import GraphStructureIndex, TemplateIndex


@dataclass(frozen=True)
class NodeAnnotation:
    layer_index: int | None = None
    component: str | None = None
    subcomponent: str | None = None
    layer_resolutions: tuple["LayerPathResolution | None", ...] = field(
        default=(), compare=False, repr=False
    )


@dataclass(frozen=True)
class IndexedPathOccurrence:
    index: int
    index_position: int
    canonical_module_path: str


@dataclass(frozen=True)
class LayerPathResolution:
    layer_index: int
    index_position: int
    canonical_module_path: str


@dataclass
class StructureAnalysisResult:
    node_annotations: dict[Op, NodeAnnotation]


@dataclass(frozen=True)
class GraphStructureAnalysisResult:
    structure_index: "GraphStructureIndex"
    template_index: "TemplateIndex | None"


def _consistent(values):
    values = {value for value in values if value is not None}
    return next(iter(values)) if len(values) == 1 else None


def _class_component(module_class: str | None) -> str | None:
    if not module_class:
        return None
    name = module_class.rsplit(".", 1)[-1]
    if name in ("RMSNorm", "LayerNorm") or name.endswith(("RMSNorm", "LayerNorm")):
        return "norm"
    if name == "Attention" or name.endswith("Attention"):
        return "attention"
    if name == "MLP" or name.endswith("MLP"):
        return "mlp"
    if name == "Embedding" or name.endswith("Embedding"):
        return "embedding"
    return None


def parse_canonical_integer(segment: str) -> int | None:
    """Parse one canonical non-negative decimal segment."""
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", segment):
        return None
    return int(segment)


def parse_indexed_path_occurrences(
    path: str | None,
) -> tuple[IndexedPathOccurrence, ...]:
    """Enumerate canonical non-negative integer segments in a module path."""
    if not path:
        return ()
    tokens = path.split(".")
    occurrences = []
    for index_position, token in enumerate(tokens):
        integer_value = parse_canonical_integer(token)
        if integer_value is None:
            continue
        canonical_tokens = list(tokens)
        canonical_tokens[index_position] = "{L}"
        occurrences.append(
            IndexedPathOccurrence(
                index=integer_value,
                index_position=index_position,
                canonical_module_path=".".join(canonical_tokens),
            )
        )
    return tuple(occurrences)


def resolve_transformer_layer_path(
    path: str | None,
) -> LayerPathResolution | None:
    """Select the unique Phase 1 Transformer-layer occurrence, if any."""
    if not path:
        return None
    tokens = path.split(".")
    matches = []
    for occurrence in parse_indexed_path_occurrences(path):
        position = occurrence.index_position
        known_nested_path = position >= 2 and tokens[position - 2 : position] in (
            ["model", "layers"],
            ["encoder", "layer"],
        )
        known_root_path = (
            position == 1 and tokens[0] in ("layers", "blocks")
        )
        if known_nested_path or known_root_path:
            matches.append(
                LayerPathResolution(
                    layer_index=occurrence.index,
                    index_position=occurrence.index_position,
                    canonical_module_path=occurrence.canonical_module_path,
                )
            )
    return matches[0] if len(matches) == 1 else None


def _classify_path(path: str | None):
    if not path:
        return None, None, None
    tokens = path.split(".")
    layer_resolution = resolve_transformer_layer_path(path)

    component = subcomponent = None
    for token in tokens:
        if token in ("input_layernorm", "post_attention_layernorm"):
            component, subcomponent = "norm", token
        elif token in ("q_proj", "k_proj", "v_proj", "o_proj"):
            component, subcomponent = "attention", token
        elif token in ("gate_proj", "up_proj", "down_proj"):
            component, subcomponent = "mlp", token
        elif token in ("self_attn", "attention", "attn") and component is None:
            component = "attention"
        elif token == "mlp" and component is None:
            component = "mlp"
        elif token in ("embed_tokens", "embedding", "embeddings"):
            component = "embedding"
        elif token == "lm_head":
            component = "lm_head"
    if tokens == ["model", "norm"]:
        component, subcomponent = "norm", "final_norm"
    return layer_resolution, component, subcomponent


def _direct_nodes(graph: Graph, names) -> set[Op]:
    return {
        node
        for name in names
        if (node := graph.node_table.get(name)) is not None
    }

def _single_direct_node(graph: Graph, names, op_type=None):
    names = tuple(names)
    if len(names) != 1:
        return None

    node = graph.node_table.get(names[0])
    if node is None:
        return None
    if op_type is not None and not isinstance(node, op_type):
        return None
    return node


def _match_rmsnorm_chain(graph: Graph, power: PowOp):
    """Match one functional RMSNorm from its power operation."""
    root = _single_direct_node(graph, power.parents, Op)
    mean = _single_direct_node(graph, power._children, MeanOp)
    if root is None or mean is None:
        return None
    epsilon_add = _single_direct_node(graph, mean._children, AddOp)
    if epsilon_add is None:
        return None
    rsqrt = _single_direct_node(graph, epsilon_add._children, RsqrtOp)
    if rsqrt is None:
        return None
    inner_mul = _single_direct_node(graph, rsqrt._children, MulOp)
    if (
        inner_mul is None
        or _direct_nodes(graph, inner_mul.parents) != {root, rsqrt}
    ):
        return None
    outer_mul = _single_direct_node(graph, inner_mul._children, MulOp)
    if (
        outer_mul is None
        or len(_direct_nodes(graph, outer_mul.parents) - {inner_mul}) != 1
    ):
        return None
    return (
        (power, mean, epsilon_add, rsqrt, inner_mul, outer_mul),
        root,
        outer_mul,
    )


class ModuleStructureAnalyzer:
    def analyze_node(self, op: Op) -> NodeAnnotation:
        """Classify one node using the same rules as whole-graph analysis."""
        layers = []
        components = []
        subcomponents = []
        layer_resolutions = []
        for source in op._source_meta:
            layer_resolution, component, subcomponent = _classify_path(
                source.module_path
            )
            if component is None:
                component = _class_component(source.module_class)
            layers.append(
                layer_resolution.layer_index
                if layer_resolution is not None
                else None
            )
            components.append(component)
            subcomponents.append(subcomponent)
            layer_resolutions.append(layer_resolution)
        layer = _consistent(layers)
        component = _consistent(components)
        subcomponent = _consistent(subcomponents)
        if component is None:
            subcomponent = None
        return NodeAnnotation(
            layer,
            component,
            subcomponent,
            tuple(layer_resolutions),
        )

    def analyze(self, graph: Graph) -> StructureAnalysisResult:
        annotations = {}
        for op in graph.body:
            annotation = self.analyze_node(op)
            if (
                annotation.layer_index is not None
                or annotation.component is not None
                or annotation.subcomponent is not None
            ):
                annotations[op] = annotation
        self._complete_layer_annotations(graph, annotations)
        self._refine_layer_annotations(graph, annotations)
        return StructureAnalysisResult(annotations)

    @staticmethod
    def _complete_layer_annotations(
        graph: Graph, annotations: dict[Op, NodeAnnotation]
    ) -> None:
        """Fill unowned runs bounded by matching layer annotations."""
        run_start = None
        left_annotation = None
        for position, op in enumerate(graph.body):
            annotation = annotations.get(op, NodeAnnotation())
            if annotation.layer_index is None:
                if run_start is None:
                    run_start = position
                continue

            if (
                run_start is not None
                and left_annotation is not None
                and left_annotation.layer_index == annotation.layer_index
            ):
                for unowned in graph.body[run_start:position]:
                    current = annotations.get(unowned, NodeAnnotation())
                    annotations[unowned] = replace(
                        current, layer_index=annotation.layer_index
                    )
            run_start = None
            left_annotation = annotation

    @staticmethod
    def _refine_layer_annotations(
        graph: Graph,
        annotations: dict[Op, NodeAnnotation],
    ) -> None:
        """Refine residual and boundary ownership from direct topology."""
        known_layers = {
            annotation.layer_index
            for annotation in annotations.values()
            if annotation.layer_index is not None
        }
        if not known_layers:
            return

        first_layer = min(known_layers)
        graph_inputs = set(graph.inputs)

        for op in graph.body:
            if isinstance(op, AddOp):
                ModuleStructureAnalyzer._refine_add_annotation(
                    graph,
                    op,
                    annotations,
                )
            elif isinstance(op, PowOp):
                ModuleStructureAnalyzer._refine_rmsnorm_annotation(
                    graph,
                    op,
                    annotations,
                    first_layer,
                    graph_inputs,
                )
                
    @staticmethod
    def _neighbor_annotations(
        graph: Graph,
        annotations: dict[Op, NodeAnnotation],
        names,
        layer_index: int,
    ) -> list[NodeAnnotation]:
        result = []
        for name in names:
            neighbor = graph.node_table.get(name)
            annotation = annotations.get(neighbor)
            if (
                annotation is not None
                and annotation.layer_index == layer_index
            ):
                result.append(annotation)
        return result

    @staticmethod
    def _refine_add_annotation(
        graph: Graph,
        op: AddOp,
        annotations: dict[Op, NodeAnnotation],
    ) -> None:
        annotation = annotations.get(op, NodeAnnotation())
        if annotation.component is not None:
            return

        if annotation.layer_index is None:
            parent_annotations = [
                annotations[parent]
                for parent in _direct_nodes(graph, op.parents)
                if parent in annotations
            ]
            mlp_layers = {
                parent.layer_index
                for parent in parent_annotations
                if parent.layer_index is not None
                and parent.component == "mlp"
            }
            if len(mlp_layers) == 1:
                layer_index = next(iter(mlp_layers))
                same_layer_parents = sum(
                    parent.layer_index == layer_index
                    for parent in parent_annotations
                )
                if same_layer_parents >= 2:
                    annotation = replace(
                        annotation,
                        layer_index=layer_index,
                    )

        if annotation.layer_index is None:
            return

        parents = ModuleStructureAnalyzer._neighbor_annotations(
            graph,
            annotations,
            op.parents,
            annotation.layer_index,
        )
        children = ModuleStructureAnalyzer._neighbor_annotations(
            graph,
            annotations,
            op._children,
            annotation.layer_index,
        )

        if any(parent.component == "mlp" for parent in parents):
            annotations[op] = NodeAnnotation(
                annotation.layer_index,
                "residual",
                "post_mlp_residual",
                annotation.layer_resolutions,
            )
        elif (
            any(parent.component == "attention" for parent in parents)
            and any(
                child.component == "norm"
                and child.subcomponent == "post_attention_layernorm"
                for child in children
            )
        ):
            annotations[op] = NodeAnnotation(
                annotation.layer_index,
                "residual",
                "post_attention_residual",
                annotation.layer_resolutions,
            )

    @staticmethod
    def _refine_rmsnorm_annotation(
        graph: Graph,
        power: PowOp,
        annotations: dict[Op, NodeAnnotation],
        first_layer: int,
        graph_inputs: set[Op],
    ) -> None:
        rmsnorm = _match_rmsnorm_chain(graph, power)
        if rmsnorm is None:
            return

        norm_nodes, root, norm_output = rmsnorm
        if any(
            annotations.get(node, NodeAnnotation()).layer_index is not None
            for node in norm_nodes
        ):
            return

        child_layers = {
            child_annotation.layer_index
            for child in _direct_nodes(graph, norm_output._children)
            if (
                child_annotation := annotations.get(child)
            ) is not None
            and child_annotation.layer_index is not None
        }
        if len(child_layers) != 1:
            return

        child_layer = next(iter(child_layers))
        root_annotation = annotations.get(root, NodeAnnotation())
        target_layer = None

        # A functional RMSNorm from layer N feeding only layer N+1 is
        # treated as layer N+1's input normalization.
        if (
            root_annotation.layer_index is not None
            and child_layer == root_annotation.layer_index + 1
        ):
            target_layer = child_layer
        elif root in graph_inputs and child_layer == first_layer:
            target_layer = child_layer
        elif (
            isinstance(root, AddOp)
            and root_annotation.layer_index is None
        ):
            root_parents = _direct_nodes(graph, root.parents)
            residual_parents = [
                parent
                for parent in root_parents
                if (
                    parent_annotation := annotations.get(parent)
                ) is not None
                and parent_annotation.layer_index == child_layer - 1
                and parent_annotation.component == "residual"
                and parent_annotation.subcomponent
                == "post_mlp_residual"
            ]
            if (
                len(root_parents) == 2
                and len(residual_parents) == 1
                and next(
                    iter(root_parents - {residual_parents[0]})
                )
                in graph_inputs
            ):
                annotations[root] = replace(
                    root_annotation,
                    layer_index=child_layer - 1,
                )
                target_layer = child_layer

        if target_layer is None:
            return

        for node in norm_nodes:
            current = annotations.get(node, NodeAnnotation())
            annotations[node] = replace(
                current,
                layer_index=target_layer,
                component="norm",
                subcomponent="input_layernorm",
            )                                