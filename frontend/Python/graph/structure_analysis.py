import re
from dataclasses import dataclass

from .graph import Graph
from .operation import Op


@dataclass(frozen=True)
class NodeAnnotation:
    layer_index: int | None = None
    component: str | None = None
    subcomponent: str | None = None


@dataclass
class StructureAnalysisResult:
    node_annotations: dict[Op, NodeAnnotation]


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
        if tokens[index : index + 2] == ["model", "layers"] and re.fullmatch(
            r"(?:0|[1-9][0-9]*)", tokens[index + 2]
        ):
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
    def analyze(self, graph: Graph) -> StructureAnalysisResult:
        annotations = {}
        for op in graph.body:
            layers = []
            components = []
            subcomponents = []
            for source in op._source_meta:
                layer, component, subcomponent = _classify_path(
                    source.module_path
                )
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
            annotation = NodeAnnotation(layer, component, subcomponent)
            if (
                annotation.layer_index is not None
                or annotation.component is not None
                or annotation.subcomponent is not None
            ):
                annotations[op] = annotation
        return StructureAnalysisResult(annotations)
