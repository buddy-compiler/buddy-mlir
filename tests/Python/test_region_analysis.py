# RUN: %PYTHON %s

import gc

import buddy.compiler.graph.operation as operation_module
from buddy.compiler.graph.graph import Graph, NodeType
from buddy.compiler.graph.operation import (
    AddOp,
    CallOp,
    MatmulOp,
    Op,
    OutputOp,
    PlaceholderOp,
    TensorConstantOp,
)
from buddy.compiler.graph.region_analysis import (
    GraphValueRef,
    GraphRegion,
    LayerRegion,
    RegionInputKind,
    RegionInputRef,
    RegionKind,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.graph.structure_analysis import NodeAnnotation


def make_node(node_class, name, module_path=None):
    node = node_class()
    node.name = name
    if module_path is not None:
        node._source_meta = (SourceMeta(module_path=module_path),)
    return node


def add(graph, node, node_type=NodeType.OtherNode):
    graph.add_node(node, node_type)
    return node


def connect(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    child.add_argument(parent.name)


def standard_graph():
    graph = Graph({}, "region_test")
    graph_input = add(graph, make_node(PlaceholderOp, "input"), NodeType.InputNode)
    parameter = add(graph, make_node(PlaceholderOp, "weight"), NodeType.FakeNode)
    constant = add(graph, make_node(TensorConstantOp, "constant"))
    embedding = add(
        graph,
        make_node(AddOp, "embedding", "model.embed_tokens"),
    )
    layer0_q = add(
        graph,
        make_node(MatmulOp, "layer0_q", "model.layers.0.self_attn.q_proj"),
    )
    layer0_down = add(
        graph,
        make_node(AddOp, "layer0_down", "model.layers.0.mlp.down_proj"),
    )
    layer1_q = add(
        graph,
        make_node(MatmulOp, "layer1_q", "model.layers.1.self_attn.q_proj"),
    )
    layer1_down = add(
        graph,
        make_node(AddOp, "layer1_down", "model.layers.1.mlp.down_proj"),
    )
    final_norm = add(graph, make_node(AddOp, "final_norm", "model.norm"))
    lm_head = add(graph, make_node(MatmulOp, "lm_head", "lm_head"))
    output = add(graph, make_node(OutputOp, "output"))

    # Repeated edges exercise stable interface deduplication.
    connect(graph_input, embedding)
    connect(graph_input, embedding)
    connect(parameter, embedding)
    connect(constant, embedding)
    connect(embedding, layer0_q)
    connect(layer0_q, layer0_down)
    connect(layer0_down, layer1_q)
    connect(layer1_q, layer1_down)
    connect(layer1_down, final_norm)
    connect(final_norm, lm_head)
    connect(lm_head, output)
    return graph, {node.name: node for node in graph.body}


# A: complete standard structure and real SourceMeta classification.
graph, nodes = standard_graph()
assert graph.structure_index is None
index = graph.build_structure_index()
assert graph.structure_index is index
assert [region.kind for region in index.regions] == [
    RegionKind.PRELUDE,
    RegionKind.LAYER,
    RegionKind.LAYER,
    RegionKind.EPILOGUE,
]
prelude, layer0, layer1, epilogue = index.regions
assert type(prelude) is GraphRegion
assert isinstance(layer0, LayerRegion)
assert isinstance(layer1, LayerRegion)
assert type(epilogue) is GraphRegion
assert prelude.nodes == [nodes["embedding"]]
assert layer0.layer_index == 0
assert layer1.layer_index == 1
assert layer0.nodes == [nodes["layer0_q"], nodes["layer0_down"]]
assert layer1.nodes == [nodes["layer1_q"], nodes["layer1_down"]]
assert layer0.component_nodes == {
    "attention": [nodes["layer0_q"]],
    "mlp": [nodes["layer0_down"]],
}
assert layer0.subcomponent_nodes == {
    "q_proj": [nodes["layer0_q"]],
    "down_proj": [nodes["layer0_down"]],
}
assert epilogue.nodes == [nodes["final_norm"], nodes["lm_head"]]
assert len(index.node_to_region) == 7
for excluded in ("input", "weight", "constant", "output"):
    assert nodes[excluded] not in index.node_to_region
assert len({id(node) for region in index.regions for node in region.nodes}) == 7


# B: macro interfaces, ordered deduplication, and intentionally empty state.
assert prelude.interface.data_inputs == [nodes["input"]]
assert prelude.interface.parameters == [nodes["weight"]]
assert prelude.interface.constants == [nodes["constant"]]
assert prelude.interface.data_outputs == [nodes["embedding"]]
assert prelude.interface.ordered_inputs == [
    RegionInputRef(RegionInputKind.DATA, GraphValueRef(nodes["input"])),
    RegionInputRef(RegionInputKind.PARAMETER, GraphValueRef(nodes["weight"])),
    RegionInputRef(RegionInputKind.CONSTANT, GraphValueRef(nodes["constant"])),
]
assert layer0.interface.data_inputs == [nodes["embedding"]]
assert layer0.interface.data_outputs == [nodes["layer0_down"]]
assert layer1.interface.data_inputs == [nodes["layer0_down"]]
assert layer1.interface.data_outputs == [nodes["layer1_down"]]
assert epilogue.interface.data_inputs == [nodes["layer1_down"]]
assert epilogue.interface.data_outputs == [nodes["lm_head"]]
for region in index.regions:
    interface = region.interface
    assert interface.state_inputs == []
    assert interface.state_outputs == []
    all_inputs = interface.data_inputs + interface.parameters + interface.constants
    assert len(all_inputs) == len({id(node) for node in all_inputs})
    assert len(interface.data_outputs) == len(
        {id(node) for node in interface.data_outputs}
    )


# Result indices remain part of Region boundaries.
multi_graph = Graph({}, "multi_result_interface")
multi_input = add(
    multi_graph, make_node(PlaceholderOp, "multi_input"), NodeType.InputNode
)
multi_producer = add(
    multi_graph,
    make_node(CallOp, "multi_producer", "model.layers.0.self_attn.q_proj"),
)
multi_producer.tensor_meta = {
    "shape": [(2, 4), (2, 5)],
    "dtype": ["f32", "f16"],
}
multi_consumer = add(
    multi_graph,
    make_node(AddOp, "multi_consumer", "model.layers.1.mlp.down_proj"),
)
multi_output = add(multi_graph, make_node(OutputOp, "multi_output"))
connect(multi_input, multi_producer)
multi_producer.add_children(multi_consumer.name)
multi_consumer.add_parent(multi_producer.name)
multi_consumer.add_argument(multi_producer.name, 1)
connect(multi_consumer, multi_output)
multi_index = multi_graph.build_structure_index()
multi_layer0, multi_layer1 = multi_index.regions
assert multi_layer0.interface.ordered_outputs == [
    GraphValueRef(multi_producer, 1)
]
assert multi_layer1.interface.ordered_inputs == [
    RegionInputRef(
        RegionInputKind.DATA, GraphValueRef(multi_producer, 1)
    )
]


# C: unknown nodes remain separate from layers; assigned nodes split runs.
unknown_graph = Graph({}, "unknown_test")
unknown_input = add(
    unknown_graph,
    make_node(PlaceholderOp, "unknown_input"),
    NodeType.InputNode,
)
unknown_sequence = [
    add(
        unknown_graph,
        make_node(MatmulOp, "l0_first", "model.layers.0.self_attn.q_proj"),
    ),
    add(unknown_graph, make_node(AddOp, "unknown_a")),
    add(unknown_graph, make_node(AddOp, "unknown_b")),
    add(
        unknown_graph,
        make_node(AddOp, "l0_second", "model.layers.0.mlp.down_proj"),
    ),
    add(unknown_graph, make_node(AddOp, "unknown_c")),
    add(unknown_graph, make_node(AddOp, "unknown_d")),
    add(
        unknown_graph,
        make_node(MatmulOp, "l1", "model.layers.1.self_attn.q_proj"),
    ),
]
unknown_output = add(unknown_graph, make_node(OutputOp, "unknown_output"))
previous = unknown_input
for node in unknown_sequence:
    connect(previous, node)
    previous = node
connect(previous, unknown_output)
unknown_index = unknown_graph.build_structure_index()
assert [region.kind for region in unknown_index.regions] == [
    RegionKind.LAYER,
    RegionKind.UNKNOWN,
    RegionKind.UNKNOWN,
    RegionKind.LAYER,
]
unknown_regions = [
    region for region in unknown_index.regions if region.kind is RegionKind.UNKNOWN
]
assert [[node.name for node in region.nodes] for region in unknown_regions] == [
    ["unknown_a", "unknown_b"],
    ["unknown_c", "unknown_d"],
]
for region in unknown_regions:
    for node in region.nodes:
        assert unknown_index.annotations.get(node) is None
        assert not isinstance(unknown_index.node_to_region[node], LayerRegion)


# D: cache identity and purity of all graph-side containers and Buddy nodes.
pure_graph, pure_nodes = standard_graph()
body_ref = pure_graph.body
body_snapshot = list(pure_graph.body)
table_ref = pure_graph.node_table
table_snapshot = list(pure_graph.node_table.items())
edge_snapshot = {
    node: (
        node.parents,
        list(node.parents),
        node._children,
        list(node._children),
        node.args,
        list(node.args),
    )
    for node in pure_graph.body
}
op_classes = {
    value
    for value in vars(operation_module).values()
    if isinstance(value, type) and issubclass(value, Op)
}
graph_ids_before = {id(value) for value in gc.get_objects() if type(value) is Graph}
op_ids_before = {id(value) for value in gc.get_objects() if type(value) in op_classes}
first_index = pure_graph.build_structure_index()
second_index = pure_graph.build_structure_index()
assert {
    id(value) for value in gc.get_objects() if type(value) is Graph
} == graph_ids_before
assert {
    id(value) for value in gc.get_objects() if type(value) in op_classes
} == op_ids_before
assert first_index is second_index
assert pure_graph.body is body_ref
assert all(
    actual is expected
    for actual, expected in zip(pure_graph.body, body_snapshot, strict=True)
)
assert pure_graph.node_table is table_ref
assert list(pure_graph.node_table.items()) == table_snapshot
for node, snapshot in edge_snapshot.items():
    parents_ref, parents, children_ref, children, args_ref, args = snapshot
    assert node.parents is parents_ref and node.parents == parents
    assert node._children is children_ref and node._children == children
    assert node.args is args_ref and node.args == args
assert set(first_index.node_to_region) == {
    pure_nodes[name]
    for name in (
        "embedding",
        "layer0_q",
        "layer0_down",
        "layer1_q",
        "layer1_down",
        "final_norm",
        "lm_head",
    )
}
assert all(
    node in body_snapshot for region in first_index.regions for node in region.nodes
)


# E: no-layer graphs use semantic endpoints and unknown contiguous runs.
no_layer_graph = Graph({}, "no_layer_test")
no_layer_input = add(
    no_layer_graph,
    make_node(PlaceholderOp, "no_layer_input"),
    NodeType.InputNode,
)
no_layer_embedding = add(
    no_layer_graph,
    make_node(AddOp, "no_layer_embedding", "model.embed_tokens"),
)
no_layer_unknown = add(no_layer_graph, make_node(AddOp, "no_layer_unknown"))
no_layer_final = add(no_layer_graph, make_node(AddOp, "no_layer_final", "model.norm"))
no_layer_head = add(no_layer_graph, make_node(MatmulOp, "no_layer_head", "lm_head"))
no_layer_output = add(no_layer_graph, make_node(OutputOp, "no_layer_output"))
no_layer_sequence = [
    no_layer_input,
    no_layer_embedding,
    no_layer_unknown,
    no_layer_final,
    no_layer_head,
    no_layer_output,
]
for parent, child in zip(no_layer_sequence[:-1], no_layer_sequence[1:], strict=True):
    connect(parent, child)
no_layer_index = no_layer_graph.build_structure_index()
assert [region.kind for region in no_layer_index.regions] == [
    RegionKind.PRELUDE,
    RegionKind.UNKNOWN,
    RegionKind.EPILOGUE,
]
assert no_layer_index.regions[0].nodes == [no_layer_embedding]
assert no_layer_index.regions[1].nodes == [no_layer_unknown]
assert no_layer_index.regions[2].nodes == [
    no_layer_final,
    no_layer_head,
]


def assert_dangling_edge_rejected(graph, owner_name, edge_kind, missing_name):
    try:
        graph.build_structure_index()
    except RuntimeError as error:
        message = str(error)
        assert owner_name in message
        assert edge_kind in message
        assert missing_name in message
    else:
        raise AssertionError(
            f"expected dangling {edge_kind} {missing_name!r} to be rejected"
        )


# F: a removed consumer must not be interpreted as an external Region user.
dangling_child_graph = Graph({}, "dangling_child_test")
dangling_producer = add(
    dangling_child_graph,
    make_node(AddOp, "dangling_producer"),
)
dangling_output = add(
    dangling_child_graph,
    make_node(OutputOp, "dangling_output"),
)
connect(dangling_producer, dangling_output)
dangling_producer.add_children("removed_consumer")
assert_dangling_edge_rejected(
    dangling_child_graph,
    "dangling_producer",
    "child",
    "removed_consumer",
)


# G: Region inputs must resolve to existing graph operations.
dangling_parent_graph = Graph({}, "dangling_parent_test")
dangling_consumer = add(
    dangling_parent_graph,
    make_node(AddOp, "dangling_consumer"),
)
dangling_consumer.add_parent("missing_producer")
dangling_consumer.add_argument("missing_producer")
assert_dangling_edge_rejected(
    dangling_parent_graph,
    "dangling_consumer",
    "parent",
    "missing_producer",
)


# H: layer-local residual adds use topology, not names, order, or layer number.
residual_graph = Graph({}, "residual_semantics_test")
residual_input = add(
    residual_graph,
    make_node(PlaceholderOp, "residual_input"),
    NodeType.InputNode,
)
attention_output = add(
    residual_graph,
    make_node(
        MatmulOp,
        "projected_branch",
        "model.layers.13.self_attn.o_proj",
    ),
)
attention_residual = add(
    residual_graph,
    make_node(AddOp, "merge_alpha", "model.layers.13"),
)
ordinary_add = add(
    residual_graph,
    make_node(AddOp, "mystery_sum", "model.layers.13"),
)
post_attention_norm = add(
    residual_graph,
    make_node(
        AddOp,
        "normalize_after_branch",
        "model.layers.13.post_attention_layernorm",
    ),
)
mlp_output = add(
    residual_graph,
    make_node(MatmulOp, "contracted_branch", "model.layers.13.mlp.down_proj"),
)
mlp_residual = add(
    residual_graph,
    make_node(AddOp, "merge_omega", "model.layers.13"),
)
residual_output = add(
    residual_graph,
    make_node(OutputOp, "residual_output"),
)
connect(residual_input, attention_output)
connect(residual_input, attention_residual)
connect(attention_output, attention_residual)
connect(residual_input, ordinary_add)
connect(ordinary_add, residual_output)
connect(attention_residual, post_attention_norm)
connect(post_attention_norm, mlp_output)
connect(attention_residual, mlp_residual)
connect(mlp_output, mlp_residual)
connect(mlp_residual, residual_output)

residual_body_ref = residual_graph.body
residual_body_snapshot = list(residual_graph.body)
residual_table_ref = residual_graph.node_table
residual_table_snapshot = list(residual_graph.node_table.items())
residual_node_snapshot = {
    node: (
        node.parents,
        list(node.parents),
        node._children,
        list(node._children),
        node.args,
        list(node.args),
        node.kwargs,
        dict(node.kwargs),
    )
    for node in residual_graph.body
}

residual_first = residual_graph.analyze_structure()
residual_second = residual_graph.analyze_structure()
residual_index = residual_first.structure_index
assert residual_second.structure_index is residual_index
assert residual_index.annotations[attention_residual] == NodeAnnotation(
    layer_index=13,
    component="residual",
    subcomponent="post_attention_residual",
)
assert residual_index.annotations[mlp_residual] == NodeAnnotation(
    layer_index=13,
    component="residual",
    subcomponent="post_mlp_residual",
)
assert residual_index.annotations[ordinary_add] == NodeAnnotation(layer_index=13)
residual_region = next(
    region
    for region in residual_index.regions
    if isinstance(region, LayerRegion) and region.layer_index == 13
)
assert residual_region.component_nodes["residual"] == [
    attention_residual,
    mlp_residual,
]
assert residual_region.subcomponent_nodes["post_attention_residual"] == [
    attention_residual
]
assert residual_region.subcomponent_nodes["post_mlp_residual"] == [mlp_residual]

assert residual_graph.body is residual_body_ref
assert residual_graph.body == residual_body_snapshot
assert residual_graph.node_table is residual_table_ref
assert list(residual_graph.node_table.items()) == residual_table_snapshot
for node, snapshot in residual_node_snapshot.items():
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
    assert node.parents is parents_ref and node.parents == parents
    assert node._children is children_ref and node._children == children
    assert node.args is args_ref and node.args == args
    assert node.kwargs is kwargs_ref and node.kwargs == kwargs
