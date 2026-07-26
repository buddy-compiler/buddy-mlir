import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .graph import Graph
from .operation import AddOp, Op

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
        self._classify_residuals(graph, annotations)
        self._complete_layer_annotations(graph, annotations)
        self._classify_residuals(graph, annotations, unowned_only=True)
        return StructureAnalysisResult(annotations)

    @staticmethod
    def _complete_layer_annotations(
        graph: Graph, annotations: dict[Op, NodeAnnotation]
    ) -> None:
        """Fill unowned runs bounded by matching layer/component annotations."""
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
    def _classify_residuals(
        graph: Graph,
        annotations: dict[Op, NodeAnnotation],
        unowned_only: bool = False,
    ) -> None:
        """Refine layer-local residual adds from direct topology evidence."""

        def neighbor_annotations(names, layer_index):
            result = []
            for name in names:
                neighbor = graph.node_table.get(name)
                annotation = annotations.get(neighbor)
                if annotation is not None and annotation.layer_index == layer_index:
                    result.append(annotation)
            return result

        for op in graph.body:
            annotation = annotations.get(op, NodeAnnotation())
            if (
                not isinstance(op, AddOp)
                or annotation.component is not None
            ):
                continue

            if annotation.layer_index is None:
                if not unowned_only:
                    continue
                parent_annotations = [
                    annotations[parent]
                    for parent in {
                        graph.node_table.get(name) for name in op.parents
                    }
                    if parent in annotations
                ]
                mlp_layers = {
                    parent.layer_index
                    for parent in parent_annotations
                    if parent.layer_index is not None
                    and parent.component == "mlp"
                }
                if len(mlp_layers) != 1:
                    continue
                layer_index = next(iter(mlp_layers))
                if (
                    sum(
                        parent.layer_index == layer_index
                        for parent in parent_annotations
                    )
                    < 2
                ):
                    continue
                annotation = replace(annotation, layer_index=layer_index)
            elif unowned_only:
                continue

            parents = neighbor_annotations(op.parents, annotation.layer_index)
            children = neighbor_annotations(op._children, annotation.layer_index)

            mlp_input = any(parent.component == "mlp" for parent in parents)
            if mlp_input:
                annotations[op] = NodeAnnotation(
                    annotation.layer_index,
                    "residual",
                    "post_mlp_residual",
                    annotation.layer_resolutions,
                )
                continue

            attention_input = any(
                parent.component == "attention" for parent in parents
            )
            post_attention_norm_output = any(
                child.component == "norm"
                and child.subcomponent == "post_attention_layernorm"
                for child in children
            )
            if attention_input and post_attention_norm_output:
                annotations[op] = NodeAnnotation(
                    annotation.layer_index,
                    "residual",
                    "post_attention_residual",
                    annotation.layer_resolutions,
                )
