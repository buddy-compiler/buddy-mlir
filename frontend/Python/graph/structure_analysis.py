import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .graph import Graph
from .operation import AddOp, Op

if TYPE_CHECKING:
    from .region_analysis import GraphStructureIndex
    from .template_analysis import TemplateIndex


@dataclass(frozen=True)
class NodeAnnotation:
    layer_index: int | None = None
    component: str | None = None
    subcomponent: str | None = None


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


def _classify_path(path: str | None):
    if not path:
        return None, None, None
    tokens = path.split(".")
    layer = None
    for index in range(len(tokens) - 2):
        if tokens[index : index + 2] in (
            ["model", "layers"],
            ["encoder", "layer"],
        ) and re.fullmatch(r"(?:0|[1-9][0-9]*)", tokens[index + 2]):
            layer = int(tokens[index + 2])
            break

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
    return layer, component, subcomponent


class ModuleStructureAnalyzer:
    def analyze_node(self, op: Op) -> NodeAnnotation:
        """Classify one node using the same rules as whole-graph analysis."""
        layers = []
        components = []
        subcomponents = []
        for source in op._source_meta:
            layer, component, subcomponent = _classify_path(source.module_path)
            if component is None:
                component = _class_component(source.module_class)
            layers.append(layer)
            components.append(component)
            subcomponents.append(subcomponent)
        layer = _consistent(layers)
        component = _consistent(components)
        subcomponent = _consistent(subcomponents)
        if component is None:
            subcomponent = None
        return NodeAnnotation(layer, component, subcomponent)

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
        return StructureAnalysisResult(annotations)

    @staticmethod
    def _classify_residuals(
        graph: Graph, annotations: dict[Op, NodeAnnotation]
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

        for op, annotation in list(annotations.items()):
            if (
                not isinstance(op, AddOp)
                or annotation.layer_index is None
                or annotation.component is not None
            ):
                continue

            parents = neighbor_annotations(op.parents, annotation.layer_index)
            children = neighbor_annotations(op._children, annotation.layer_index)

            mlp_input = any(parent.component == "mlp" for parent in parents)
            if mlp_input:
                annotations[op] = NodeAnnotation(
                    annotation.layer_index,
                    "residual",
                    "post_mlp_residual",
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
                )
