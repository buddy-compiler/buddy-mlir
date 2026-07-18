# RUN: %PYTHON %s

import copy
import gc
import json

import buddy.compiler.graph.operation as operation_module
import torch
from buddy.compiler.graph.graph import Graph, NodeType
from buddy.compiler.graph.operation import (
    AddOp,
    CallOp,
    MatmulOp,
    Op,
    OpType,
    OutputOp,
    PlaceholderOp,
)
from buddy.compiler.graph.region_analysis import LayerRegion, RegionKind
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.graph.structure_analysis import (
    GraphStructureAnalysisResult,
    ModuleStructureAnalyzer,
    StructureAnalysisResult,
)


def node(cls, name, path=None, aten=None, shape=(2, 4), dtype=torch.float32):
    result = cls()
    result.name = name
    result.tensor_meta = {"shape": shape, "dtype": dtype}
    if path is not None or aten is not None:
        result._source_meta = (
            SourceMeta(
                module_path=path,
                module_class="example.DecoderLayer",
                original_aten=aten,
            ),
        )
    return result


def add(graph, op, kind=NodeType.OtherNode):
    graph.add_node(op, kind)
    return op


def bind(parent, child, *, argument=True):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    if argument:
        child.add_argument(parent.name)


def three_identical_layers():
    graph = Graph({}, "three_templates")
    graph_input = add(graph, node(PlaceholderOp, "tokens"), NodeType.InputNode)
    parameters = []
    for layer in range(3):
        parameter = node(PlaceholderOp, f"model.layers.{layer}.weight")
        parameter._arguments = [[layer, layer + 1, layer + 2]]
        parameters.append(add(graph, parameter, NodeType.FakeNode))

    layer_nodes = []
    for layer in range(3):
        prefix = f"renamed_node_{layer}"
        matmul = add(
            graph,
            node(
                MatmulOp,
                f"{prefix}_matmul",
                f"model.layers.{layer}.self_attn.q_proj",
                "aten.mm.default",
            ),
        )
        residual = add(
            graph,
            node(
                AddOp,
                f"{prefix}_residual",
                f"model.layers.{layer}.mlp.down_proj",
                "aten.add.Tensor",
            ),
        )
        output = add(graph, node(OutputOp, f"output_{layer}"))
        bind(graph_input, matmul)
        bind(parameters[layer], matmul)
        bind(matmul, residual)
        residual.add_argument(1.0)
        bind(residual, output)
        layer_nodes.append((matmul, residual))
    return graph, layer_nodes


# Identical layers ignore layer number, Buddy name, parameter name, and value.
graph, layer_nodes = three_identical_layers()
result = graph.analyze_structure(True)
assert isinstance(result, GraphStructureAnalysisResult)
assert isinstance(ModuleStructureAnalyzer().analyze(graph), StructureAnalysisResult)
template_index = result.template_index
layers = sorted(
    (
        region
        for region in result.structure_index.regions
        if isinstance(region, LayerRegion)
    ),
    key=lambda region: region.layer_index,
)
assert len(layers) == 3
assert len(template_index.template_groups) == 1
assert template_index.non_reusable_regions == []
group = template_index.template_groups[0]
assert group.representative is layers[0]
assert group.instances == layers
assert len(group.instances) == 3
fingerprints = [template_index.region_fingerprints[layer] for layer in layers]
assert len({fingerprint.digest for fingerprint in fingerprints}) == 1
assert all(fingerprint is fingerprints[0] for fingerprint in fingerprints)
assert group.fingerprint is fingerprints[0]
assert isinstance(group.canonical_form, bytes) and group.canonical_form
assert sum(1 for item in template_index.template_groups if item.canonical_form) == 1
group_canonical = json.loads(group.canonical_form)
assert all(
    "argument_result_indices" not in item["attributes"]
    for item in group_canonical["nodes"]
)


# A kwargs-only use-def dependency is a Region input and fingerprint slot.
kwargs_graph = Graph({}, "kwargs_only_input")
kwargs_input = add(
    kwargs_graph, node(PlaceholderOp, "kwargs_input"), NodeType.InputNode
)
for layer_index in range(2):
    kwargs_node = add(
        kwargs_graph,
        node(
            AddOp,
            f"kwargs_node_{layer_index}",
            f"model.layers.{layer_index}.mlp.down_proj",
            "aten.add.Tensor",
        ),
    )
    kwargs_output = add(
        kwargs_graph, node(OutputOp, f"kwargs_output_{layer_index}")
    )
    bind(kwargs_input, kwargs_node, argument=False)
    kwargs_node.kwargs["other"] = kwargs_input.name
    bind(kwargs_node, kwargs_output)
kwargs_result = kwargs_graph.analyze_structure(True)
kwargs_layers = sorted(
    (
        region
        for region in kwargs_result.structure_index.regions
        if isinstance(region, LayerRegion)
    ),
    key=lambda region: region.layer_index,
)
assert [region.interface.data_inputs for region in kwargs_layers] == [
    [kwargs_input],
    [kwargs_input],
]
kwargs_group = kwargs_result.template_index.template_groups[0]
kwargs_canonical = json.loads(kwargs_group.canonical_form)
assert len(kwargs_canonical["external_slots"]["input"]) == 1
assert kwargs_group.fingerprint.summary.data_input_count == 1


def single_layer(variant=None):
    graph = Graph({}, f"single_{variant}")
    graph_input = add(graph, node(PlaceholderOp, "input"), NodeType.InputNode)
    parameter0 = add(
        graph, node(PlaceholderOp, "weight0"), NodeType.FakeNode
    )
    parameter1 = None
    if variant == "parameter_count":
        parameter1 = add(
            graph, node(PlaceholderOp, "weight1"), NodeType.FakeNode
        )

    second_cls = MatmulOp if variant == "op_type" else AddOp
    first = add(
        graph,
        node(
            MatmulOp,
            "first",
            "model.layers.7.self_attn.q_proj",
            "aten.mm.default",
            shape=(2, 5) if variant == "shape" else (2, 4),
            dtype=torch.float16 if variant == "dtype" else torch.float32,
        ),
    )
    second = add(
        graph,
        node(
            second_cls,
            "second",
            "model.layers.7.mlp.down_proj",
            "aten.add.Tensor",
        ),
    )
    output = add(graph, node(OutputOp, "output"))
    extra_output = (
        add(graph, node(OutputOp, "extra_output"))
        if variant == "output_count"
        else None
    )

    if variant == "args_order":
        bind(parameter0, first)
        bind(graph_input, first)
    else:
        bind(graph_input, first)
        bind(parameter0, first)
    if parameter1 is not None:
        bind(parameter1, first)
    if variant == "topology":
        bind(graph_input, second)
    else:
        bind(first, second)
    second.add_argument(2.0)
    second.kwargs["alpha"] = 2 if variant == "kwargs" else 1
    if variant == "stable_attribute":
        first._layout = "NHWC"
    elif variant == "newshape":
        first._newshape = [99, 101]
    elif variant == "op_type_classification":
        original_op_type = first._op_type
        first._op_type = (
            OpType.Unfusable
            if original_op_type != OpType.Unfusable
            else OpType.ElementwiseType
        )
        assert first._op_type != original_op_type
    bind(second, output)
    if extra_output is not None:
        bind(first, extra_output)
    return graph


def fingerprint(graph):
    analyzed = graph.analyze_structure(True)
    assert len(analyzed.template_index.region_fingerprints) == 1
    return next(iter(analyzed.template_index.region_fingerprints.values()))


# Representative semantic and interface changes must never be grouped.
base_digest = fingerprint(single_layer()).digest
for difference in (
    "op_type",
    "args_order",
    "kwargs",
    "topology",
    "stable_attribute",
    "shape",
    "dtype",
    "parameter_count",
    "output_count",
):
    assert fingerprint(single_layer(difference)).digest != base_digest, difference

for ignored_field in ("newshape", "op_type_classification"):
    assert fingerprint(single_layer(ignored_field)).digest == base_digest, ignored_field


def call_graph(callee, argument_result_index):
    graph = Graph({}, "call_template")
    graph_input = add(graph, node(PlaceholderOp, "call_input"), NodeType.InputNode)
    call = add(
        graph,
        node(CallOp, "call", "model.layers.0.mlp", "aten.add.Tensor"),
    )
    output = add(graph, node(OutputOp, "call_output"))
    bind(graph_input, call)
    call.call_func_name = callee
    call._args_index = [argument_result_index]
    bind(call, output)
    return graph


call_digest = fingerprint(call_graph("callee_a", 0)).digest
assert fingerprint(call_graph("callee_b", 0)).digest != call_digest
assert fingerprint(call_graph("callee_a", 1)).digest != call_digest


# A singleton retains only its small fingerprint, never a canonical byte form.
unique_graph = single_layer()
unique_result = unique_graph.analyze_structure(True)
unique_template = unique_result.template_index
unique_layer = next(iter(unique_template.region_fingerprints))
assert unique_template.template_groups == []
assert unique_template.non_reusable_regions == [unique_layer]
assert not hasattr(unique_template.region_fingerprints[unique_layer], "canonical_form")


# Prelude, epilogue, and unknown regions are structurally indexed but excluded.
excluded_graph = Graph({}, "excluded_regions")
excluded_input = add(
    excluded_graph, node(PlaceholderOp, "excluded_input"), NodeType.InputNode
)
embedding = add(
    excluded_graph, node(AddOp, "embedding", "model.embed_tokens")
)
layer_node = add(
    excluded_graph,
    node(MatmulOp, "layer", "model.layers.0.self_attn.q_proj"),
)
unknown = add(excluded_graph, node(AddOp, "unknown"))
layer_node_1 = add(
    excluded_graph,
    node(MatmulOp, "layer_1", "model.layers.1.self_attn.q_proj"),
)
final_norm = add(excluded_graph, node(AddOp, "norm", "model.norm"))
head = add(excluded_graph, node(MatmulOp, "head", "lm_head"))
excluded_output = add(excluded_graph, node(OutputOp, "excluded_output"))
for parent, child in zip(
    [excluded_input, embedding, layer_node, unknown, layer_node_1, final_norm, head],
    [embedding, layer_node, unknown, layer_node_1, final_norm, head, excluded_output],
    strict=True,
):
    bind(parent, child)
excluded_result = excluded_graph.analyze_structure(True)
assert {region.kind for region in excluded_result.structure_index.regions} == {
    RegionKind.PRELUDE,
    RegionKind.LAYER,
    RegionKind.EPILOGUE,
    RegionKind.UNKNOWN,
}
assert all(
    isinstance(region, LayerRegion)
    for region in excluded_result.template_index.region_fingerprints
)
assert len(excluded_result.template_index.region_fingerprints) == 2


# An opaque literal makes only its owning Layer non-reusable; supported Layers
# in the same graph still fingerprint and group normally.
class OpaqueLiteral:
    pass


def layer_regions(result):
    return sorted(
        (r for r in result.structure_index.regions if isinstance(r, LayerRegion)),
        key=lambda r: r.layer_index,
    )


mixed_graph = Graph({}, "opaque_and_reusable")
mixed_input = add(mixed_graph, node(PlaceholderOp, "mixed_input"), NodeType.InputNode)
for layer_index, literal in enumerate((OpaqueLiteral(), 1.0, 1.0)):
    mixed_node = add(
        mixed_graph,
        node(AddOp, f"mixed_{layer_index}", f"model.layers.{layer_index}.mlp"),
    )
    bind(mixed_input, mixed_node)
    mixed_node.add_argument(literal)
    bind(mixed_node, add(mixed_graph, node(OutputOp, f"mixed_out_{layer_index}")))
mixed_result = mixed_graph.analyze_structure(True)
mixed_layers = layer_regions(mixed_result)
mixed_template = mixed_result.template_index
assert mixed_template.non_reusable_regions == mixed_layers[:1]
assert mixed_layers[0] not in mixed_template.region_fingerprints
assert len(mixed_template.template_groups) == 1
assert mixed_template.template_groups[0].instances == mixed_layers[1:]


# Dict items are ordered before operand slot assignment, and list pairs allow
# internal _NodeRef values to resolve into JSON-native ["node", local_id].
dict_graph = Graph({}, "dict_internal_reference")
dict_input = add(dict_graph, node(PlaceholderOp, "dict_input"), NodeType.InputNode)
for layer_index in range(2):
    producer = add(
        dict_graph,
        node(
            MatmulOp,
            f"dict_p_{layer_index}",
            f"model.layers.{layer_index}.self_attn.q_proj",
            "aten.mm.default",
        ),
    )
    consumer = add(
        dict_graph,
        node(
            AddOp,
            f"dict_c_{layer_index}",
            f"model.layers.{layer_index}.mlp.down_proj",
            "aten.add.Tensor",
        ),
    )
    bind(dict_input, producer)
    bind(producer, consumer, argument=False)
    pairs = [("external", dict_input.name), ("internal", producer.name)]
    consumer.kwargs["operands"] = dict(reversed(pairs) if layer_index else pairs)
    bind(consumer, add(dict_graph, node(OutputOp, f"dict_out_{layer_index}")))
dict_result = dict_graph.analyze_structure(True)
dict_layers = layer_regions(dict_result)
dict_template = dict_result.template_index
assert (
    dict_template.region_fingerprints[dict_layers[0]].digest
    == dict_template.region_fingerprints[dict_layers[1]].digest
)
assert len(dict_template.template_groups) == 1
assert dict_template.template_groups[0].instances == dict_layers
assert len(dict_template.template_groups[0].instances) == 2
dict_canonical = json.loads(dict_template.template_groups[0].canonical_form)
dict_items = dict_canonical["nodes"][1]["kwargs"][1][0][1][1]
assert [["literal", "internal"], ["node", 0]] in dict_items


# Repeated analysis is deterministic and returns exactly the cached objects.
cached_graph, _ = three_identical_layers()
first_result = cached_graph.analyze_structure(True)
second_result = cached_graph.analyze_structure(True)
assert first_result.structure_index is second_result.structure_index
assert first_result.template_index is second_result.template_index
assert cached_graph.structure_index is first_result.structure_index
assert cached_graph.template_index is first_result.template_index
assert cached_graph.analyze_structure(False).template_index is first_result.template_index
assert [
    group.fingerprint.digest
    for group in first_result.template_index.template_groups
] == [
    group.fingerprint.digest
    for group in second_result.template_index.template_groups
]
for first_group, second_group in zip(
    first_result.template_index.template_groups,
    second_result.template_index.template_groups,
    strict=True,
):
    assert first_group is second_group
    assert first_group.representative is second_group.representative
    assert first_group.instances is second_group.instances


# Compatibility path supplements the original index and Region objects only.
compat_graph, _ = three_identical_layers()
structure_only = compat_graph.analyze_structure(False)
assert structure_only.template_index is None
assert compat_graph.template_index is None
compat_index = structure_only.structure_index
assert compat_graph.build_structure_index() is compat_index
compat_regions = list(compat_index.regions)
compat_result = compat_graph.analyze_structure(True)
assert compat_result.structure_index is compat_index
assert all(
    old is new
    for old, new in zip(
        compat_regions, compat_result.structure_index.regions, strict=True
    )
)
assert compat_result.template_index is compat_graph.template_index


# Analysis is descriptive: it creates no Graph or Buddy Op and mutates no use-def.
pure_graph, _ = three_identical_layers()
body_ref = pure_graph.body
body_snapshot = list(body_ref)
table_ref = pure_graph.node_table
table_snapshot = list(table_ref.items())
node_snapshot = {
    op: (
        op.parents,
        list(op.parents),
        op._children,
        list(op._children),
        op.args,
        copy.deepcopy(op.args),
        op.kwargs,
        copy.deepcopy(op.kwargs),
    )
    for op in pure_graph.body
}
op_classes = {
    value
    for value in vars(operation_module).values()
    if isinstance(value, type) and issubclass(value, Op)
}
graph_ids = {id(value) for value in gc.get_objects() if type(value) is Graph}
op_ids = {id(value) for value in gc.get_objects() if type(value) in op_classes}
pure_graph.analyze_structure(True)
assert {id(value) for value in gc.get_objects() if type(value) is Graph} == graph_ids
assert {id(value) for value in gc.get_objects() if type(value) in op_classes} == op_ids
assert pure_graph.body is body_ref and pure_graph.body == body_snapshot
assert all(
    actual is expected
    for actual, expected in zip(pure_graph.body, body_snapshot)
)
assert pure_graph.node_table is table_ref
assert list(pure_graph.node_table.items()) == table_snapshot
for op, snapshot in node_snapshot.items():
    (
        parents_ref,
        parents,
        children_ref,
        children,
        args_ref,
        args,
        kwargs_ref,
        kwargs,
    ) = snapshot
    assert op.parents is parents_ref and op.parents == parents
    assert op._children is children_ref and op._children == children
    assert op.args is args_ref and op.args == args
    assert op.kwargs is kwargs_ref and op.kwargs == kwargs
