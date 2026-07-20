# RUN: %PYTHON %s

from buddy.compiler.graph.operation import Op
from buddy.compiler.graph.source_meta import SourceMeta, extract_source_meta
from buddy.compiler.graph.structure_analysis import ModuleStructureAnalyzer


class Qwen2Attention:
    pass


class BadString:
    def __str__(self):
        raise RuntimeError("bad string")


class BadMetaNode:
    @property
    def meta(self):
        raise RuntimeError("bad meta")


class BadPath:
    def __getattribute__(self, name):
        raise RuntimeError("bad path")


class BadClass:
    def __getattribute__(self, name):
        raise RuntimeError("bad class")


class SomeAttention:
    pass


class SomeMLP:
    pass


class SomeRMSNorm:
    pass


class Node:
    def __init__(self, meta=None):
        self.meta = meta


source = extract_source_meta(
    Node(
        {
            "nn_module_stack": {
                "ignored-object-id": (
                    "L['self'].model.layers.27.self_attn",
                    Qwen2Attention,
                ),
                "also-ignored": (
                    "L['self'].model.layers.27.self_attn.q_proj",
                    Qwen2Attention,
                ),
            },
            "original_aten": "aten.mm.default",
        }
    )
)
assert source == (
    SourceMeta(
        "model.layers.27.self_attn.q_proj",
        f"{__name__}.Qwen2Attention",
        "aten.mm.default",
    ),
)
assert extract_source_meta(Node({})) == ()
assert extract_source_meta(Node(None)) == ()
bad_aten_source = extract_source_meta(
    Node(
        {
            "nn_module_stack": {"id": ("L['self'].model.layers.0.mlp", int)},
            "original_aten": BadString(),
        }
    )
)[0]
assert bad_aten_source.module_path == "model.layers.0.mlp"
assert bad_aten_source.module_class == "builtins.int"
assert bad_aten_source.original_aten is None
assert extract_source_meta(BadMetaNode()) == ()

for aten_name in ("aten.mean.dim", "aten.index_copy.default", "aten.add.Tensor"):
    assert extract_source_meta(Node({"original_aten": aten_name})) == (
        SourceMeta(original_aten=aten_name),
    )
for unstable in (
    "<OpOverload(op='aten.mean', overload='dim')>",
    "<Foo object at 0x1234>",
    "Foo object at 0x1234",
    "not_aten.mean.dim",
    "",
):
    assert extract_source_meta(Node({"original_aten": unstable})) == ()

# Malformed stack entries isolate path and class extraction from one another.
class_only = extract_source_meta(
    Node({"nn_module_stack": {"id": (BadPath(), SomeAttention)}})
)
assert class_only == (SourceMeta(module_class=f"{__name__}.SomeAttention"),)
path_only = extract_source_meta(
    Node({"nn_module_stack": {"id": ("model.layers.3.mlp", BadClass())}})
)
assert path_only == (SourceMeta(module_path="model.layers.3.mlp"),)
assert extract_source_meta(Node({"nn_module_stack": {"id": object()}})) == ()

# Equal-depth paths keep the original mapping traversal order.
tie = extract_source_meta(
    Node(
        {
            "nn_module_stack": {
                "first": ("model.layers.1.block", SomeAttention),
                "second": ("model.layers.2.block", SomeMLP),
            }
        }
    )
)
assert tie[0].module_path == "model.layers.1.block"
assert tie[0].module_class == f"{__name__}.SomeAttention"


def annotation(*sources):
    op = Op()
    op._source_meta = sources
    graph = type("GraphStub", (), {"body": [op]})()
    return ModuleStructureAnalyzer().analyze(graph).node_annotations.get(op)


cases = {
    "model.layers.0.input_layernorm": (0, "norm", "input_layernorm"),
    "model.layers.27.post_attention_layernorm": (
        27,
        "norm",
        "post_attention_layernorm",
    ),
    "model.norm": (None, "norm", "final_norm"),
    "model.layers.2.self_attn": (2, "attention", None),
    "model.layers.2.self_attn.q_proj": (2, "attention", "q_proj"),
    "model.layers.2.self_attn.k_proj": (2, "attention", "k_proj"),
    "model.layers.2.self_attn.v_proj": (2, "attention", "v_proj"),
    "model.layers.2.self_attn.o_proj": (2, "attention", "o_proj"),
    "model.layers.2.mlp": (2, "mlp", None),
    "model.layers.2.mlp.gate_proj": (2, "mlp", "gate_proj"),
    "model.layers.2.mlp.up_proj": (2, "mlp", "up_proj"),
    "model.layers.2.mlp.down_proj": (2, "mlp", "down_proj"),
    "model.embed_tokens": (None, "embedding", None),
    "lm_head": (None, "lm_head", None),
}
for path, expected in cases.items():
    value = annotation(SourceMeta(module_path=path))
    assert (value.layer_index, value.component, value.subcomponent) == expected

# BERT-family encoder paths contribute a layer index without adding new
# component or subcomponent semantics.
encoder_layer_cases = {
    "bert.encoder.layer.0.attention.self": 0,
    "encoder.encoder.layer.11.output": 11,
    "model.encoder.layer.3.intermediate": 3,
}
for path, expected_layer in encoder_layer_cases.items():
    assert annotation(SourceMeta(module_path=path)).layer_index == expected_layer
for path in (
    "bert.encoder.layers.0.attention",
    "bert.encoder.layer_norm.0",
    "bert.encoder.layer.foo.attention",
    "bert.someencoder.layer.0",
    "bert.encoder.block.layer.0",
    "bert.encoder.layer.-1",
    "bert.encoder.layer.+1",
    "bert.encoder.layer.1a",
):
    value = annotation(SourceMeta(module_path=path))
    assert value is None or value.layer_index is None

plain_encoder_value = annotation(
    SourceMeta(module_path="bert.encoder.layer.0.output.dense")
)
assert (plain_encoder_value.component, plain_encoder_value.subcomponent) == (
    None,
    None,
)

# Multiple candidates keep the simple left-to-right, first-match behavior.
assert (
    annotation(
        SourceMeta(module_path="model.layers.7.encoder.layer.3.output")
    ).layer_index
    == 7
)
assert (
    annotation(
        SourceMeta(module_path="encoder.layer.3.model.layers.7.mlp")
    ).layer_index
    == 3
)

# Structure analysis does not mutate the graph or SourceMeta containers.
purity_op = Op()
purity_sources = (
    SourceMeta(module_path="bert.encoder.layer.0.output.dense"),
)
purity_op._source_meta = purity_sources
purity_body = [purity_op]
purity_graph = type("GraphStub", (), {"body": purity_body})()
ModuleStructureAnalyzer().analyze(purity_graph)
assert purity_graph.body is purity_body and purity_graph.body == [purity_op]
assert purity_op._source_meta is purity_sources

same = SourceMeta(module_path="model.layers.0.self_attn.q_proj")
assert annotation(same, same) == annotation(same)
layer_conflict = annotation(
    SourceMeta(module_path="model.layers.0.self_attn.q_proj"),
    SourceMeta(module_path="model.layers.27.self_attn.q_proj"),
)
assert layer_conflict.layer_index is None
assert layer_conflict.component == "attention"
assert layer_conflict.subcomponent == "q_proj"
component_conflict = annotation(
    SourceMeta(module_path="model.layers.0.self_attn.q_proj"),
    SourceMeta(module_path="model.layers.0.mlp.up_proj"),
)
assert component_conflict.layer_index == 0
assert component_conflict.component is None
assert component_conflict.subcomponent is None
conflict_op = Op()
conflict_op._source_meta = (
    SourceMeta(module_path="model.layers.0.self_attn.q_proj"),
    SourceMeta(module_path="model.layers.27.mlp.up_proj"),
)
conflict_graph = type("GraphStub", (), {"body": [conflict_op]})()
result = ModuleStructureAnalyzer().analyze(conflict_graph)
assert conflict_op not in result.node_annotations
subcomponent_conflict = annotation(
    SourceMeta(module_path="model.layers.0.self_attn.q_proj"),
    SourceMeta(module_path="model.layers.0.self_attn.k_proj"),
)
assert subcomponent_conflict.layer_index == 0
assert subcomponent_conflict.component == "attention"
assert subcomponent_conflict.subcomponent is None
assert annotation(SourceMeta()) is None
assert annotation() is None

# A path can supply the layer while its class supplies only the component.
fallback_cases = (
    (SomeAttention, "attention"),
    (SomeMLP, "mlp"),
    (SomeRMSNorm, "norm"),
)
for cls, component in fallback_cases:
    value = annotation(
        SourceMeta(
            module_path="model.layers.0.block",
            module_class=f"{__name__}.{cls.__name__}",
        )
    )
    assert (value.layer_index, value.component, value.subcomponent) == (
        0,
        component,
        None,
    )

# A classified path wins, and class fallback never guesses a subcomponent.
value = annotation(
    SourceMeta(
        module_path="model.layers.0.mlp",
        module_class=f"{__name__}.SomeAttention",
    )
)
assert (value.layer_index, value.component, value.subcomponent) == (0, "mlp", None)
