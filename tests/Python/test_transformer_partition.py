# RUN: %PYTHON %s

# Region recognition and interface analysis tests.

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
    MeanOp,
    MulOp,
    Op,
    OpType,
    OutputOp,
    PlaceholderOp,
    PowOp,
    RsqrtOp,
    TensorConstantOp,
    UnsqueezeOp,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.graph.structure_analysis import (
    GraphStructureAnalysisResult,
    ModuleStructureAnalyzer,
    NodeAnnotation,
    StructureAnalysisResult,
    parse_canonical_integer,
    parse_indexed_path_occurrences,
    resolve_transformer_layer_path,
)
from buddy.compiler.graph.transformer_partition import (
    GraphRegion,
    GraphValueRef,
    LayerRegion,
    RegionInputKind,
    RegionInputRef,
    RegionKind,
    build_transformer_partition_plan,
)


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


# Path parsing is syntax-only, enumerates every canonical integer segment, and
# canonicalizes only the selected occurrence.
for token, expected in (("0", 0), ("1", 1), ("12", 12)):
    assert parse_canonical_integer(token) == expected
    occurrences = parse_indexed_path_occurrences(f"layers.{token}.attn")
    assert [occurrence.index for occurrence in occurrences] == [expected]
for token in ("01", "+1", "-1", "1.0"):
    assert parse_canonical_integer(token) is None
for token in ("01", "+1", "-1"):
    assert parse_indexed_path_occurrences(f"layers.{token}.attn") == ()
assert parse_indexed_path_occurrences("layers.attn") == ()
multiple_occurrences = parse_indexed_path_occurrences(
    "foo.layers.1.blocks.2.attn"
)
assert [
    (occurrence.index, occurrence.index_position)
    for occurrence in multiple_occurrences
] == [(1, 2), (2, 4)]
assert [
    occurrence.canonical_module_path for occurrence in multiple_occurrences
] == [
    "foo.layers.{L}.blocks.2.attn",
    "foo.layers.1.blocks.{L}.attn",
]


def classify_path(path):
    return ModuleStructureAnalyzer().analyze_node(
        make_node(AddOp, f"classify_{path}", path)
    )


# Phase 1 accepts layers/blocks containers under any prefix and the established
# encoder.layer grammar.
for path, expected_container, expected_index in (
    ("model.layers.0", "model.layers", 0),
    ("foo.model.layers.1", "foo.model.layers", 1),
    ("encoder.layer.0", "encoder.layer", 0),
    ("foo.encoder.layer.1", "foo.encoder.layer", 1),
    ("layers.0.self_attn.q_proj", "layers", 0),
    ("layers.27.mlp.down_proj", "layers", 27),
    ("blocks.0.attn.qkv", "blocks", 0),
    ("blocks.23.norm2", "blocks", 23),
    ("model.encoder.layers.0", "model.encoder.layers", 0),
    ("model.decoder.layers.0", "model.decoder.layers", 0),
    ("vision_model.encoder.layers.0", "vision_model.encoder.layers", 0),
    ("foo.layers.0.self_attn", "foo.layers", 0),
    ("foo.blocks.0.attn", "foo.blocks", 0),
):
    annotation = classify_path(path)
    assert annotation.layer_index == expected_index, path
    assert annotation.layer_container == expected_container, path
    resolution = resolve_transformer_layer_path(path)
    assert resolution is not None
    assert resolution.layer_index == expected_index
    assert resolution.layer_container == expected_container
    assert (
        annotation.layer_resolutions[0].canonical_module_path
        == resolution.canonical_module_path
    )

for path in (
    "deepstack_merger_list.0.norm",
    "experts.0",
    "stages.0",
    "heads.0",
    "adapters.0",
    "branches.0",
    "layer.0.attn",
):
    assert classify_path(path).layer_index is None, path
    assert resolve_transformer_layer_path(path) is None

# Two accepted occurrences are ambiguous and must not use the first match.
ambiguous_path = "foo.model.layers.1.encoder.layer.2.attention"
ambiguous_annotation = classify_path(ambiguous_path)
assert ambiguous_annotation.layer_index is None
assert ambiguous_annotation.layer_resolutions == (None,)
assert resolve_transformer_layer_path(ambiguous_path) is None

multi_stack_source = make_node(AddOp, "multi_stack_source")
multi_stack_source._source_meta = (
    SourceMeta(module_path="model.encoder.layers.0.self_attn.q_proj"),
    SourceMeta(module_path="model.decoder.layers.0.self_attn.q_proj"),
)
multi_stack_annotation = ModuleStructureAnalyzer().analyze_node(
    multi_stack_source
)
assert multi_stack_annotation.layer_index is None
assert multi_stack_annotation.layer_container is None


def root_layer_graph(container):
    graph = Graph({}, f"{container}_region_test")
    graph_input = add(
        graph,
        make_node(PlaceholderOp, f"{container}_input"),
        NodeType.InputNode,
    )
    layer0 = add(
        graph,
        make_node(AddOp, f"{container}_0", f"{container}.0.attn.qkv"),
    )
    layer1 = add(
        graph,
        make_node(AddOp, f"{container}_1", f"{container}.1.attn.qkv"),
    )
    output = add(graph, make_node(OutputOp, f"{container}_output"))
    connect(graph_input, layer0)
    connect(layer0, layer1)
    connect(layer1, output)
    return graph


for root_container in ("blocks", "layers"):
    root_regions = (
        root_layer_graph(root_container).build_structure_index().regions
    )
    assert all(isinstance(region, LayerRegion) for region in root_regions)
    assert [region.layer_index for region in root_regions] == [0, 1]
    assert [region.layer_container for region in root_regions] == [
        root_container,
        root_container,
    ]


def standard_graph():
    graph = Graph({}, "region_test")
    graph_input = add(
        graph, make_node(PlaceholderOp, "input"), NodeType.InputNode
    )
    parameter = add(
        graph, make_node(PlaceholderOp, "weight"), NodeType.FakeNode
    )
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
assert layer0.layer_container == "model.layers"
assert layer1.layer_container == "model.layers"
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
    all_inputs = (
        interface.data_inputs + interface.parameters + interface.constants
    )
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
    RegionInputRef(RegionInputKind.DATA, GraphValueRef(multi_producer, 1))
]


# C: matching layer/component anchors complete only the enclosed unowned run.
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
        make_node(AddOp, "l0_second", "model.layers.0.self_attn.o_proj"),
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
    RegionKind.LAYER,
]
layer0_region = next(
    region
    for region in unknown_index.regions
    if isinstance(region, LayerRegion) and region.layer_index == 0
)
assert [node.name for node in layer0_region.nodes] == [
    "l0_first",
    "unknown_a",
    "unknown_b",
    "l0_second",
]
unknown_regions = [
    region
    for region in unknown_index.regions
    if region.kind is RegionKind.UNKNOWN
]
assert [[node.name for node in region.nodes] for region in unknown_regions] == [
    ["unknown_c", "unknown_d"],
]
for region in unknown_regions:
    for node in region.nodes:
        assert unknown_index.annotations.get(node) is None
        assert not isinstance(unknown_index.node_to_region[node], LayerRegion)


# Region construction is streaming: finalized boundary/layer identities never
# reopen, while an UNKNOWN run joins only matching annotations on both sides.
continuity_graph = Graph({}, "streaming_region_continuity")
continuity_input = add(
    continuity_graph,
    make_node(PlaceholderOp, "continuity_input"),
    NodeType.InputNode,
)
encoder_prepare = add(continuity_graph, make_node(CallOp, "encoder_prepare"))
encoder_prelude = add(
    continuity_graph,
    make_node(AddOp, "encoder_prelude", "model.encoder.embed_tokens"),
)
encoder_prepare_after_anchor = add(
    continuity_graph, make_node(CallOp, "encoder_prepare_after_anchor")
)
encoder_l0_first = add(
    continuity_graph,
    make_node(
        MatmulOp,
        "encoder_l0_first",
        "model.encoder.layers.0.self_attn.q_proj",
    ),
)
same_layer_unknown = add(
    continuity_graph, make_node(CallOp, "same_layer_unknown")
)
encoder_l0_second = add(
    continuity_graph,
    make_node(
        MatmulOp,
        "encoder_l0_second",
        "model.encoder.layers.0.self_attn.o_proj",
    ),
)
different_layer_unknown = add(
    continuity_graph, make_node(CallOp, "different_layer_unknown")
)
encoder_l1 = add(
    continuity_graph,
    make_node(
        MatmulOp,
        "encoder_l1",
        "model.encoder.layers.1.self_attn.q_proj",
    ),
)
decoder_l0 = add(
    continuity_graph,
    make_node(
        MatmulOp,
        "decoder_l0",
        "model.decoder.layers.0.self_attn.q_proj",
    ),
)
decoder_prelude_first = add(
    continuity_graph,
    make_node(AddOp, "decoder_prelude_first", "model.decoder.embed_tokens"),
)
decoder_unknown_first = add(
    continuity_graph, make_node(CallOp, "decoder_unknown_first")
)
decoder_unknown_second = add(
    continuity_graph, make_node(CallOp, "decoder_unknown_second")
)
decoder_prelude_second = add(
    continuity_graph,
    make_node(AddOp, "decoder_prelude_second", "model.decoder.embedding"),
)
continuity_output = add(
    continuity_graph, make_node(OutputOp, "continuity_output")
)
continuity_sequence = [
    continuity_input,
    encoder_prepare,
    encoder_prelude,
    encoder_prepare_after_anchor,
    encoder_l0_first,
    same_layer_unknown,
    encoder_l0_second,
    different_layer_unknown,
    encoder_l1,
    decoder_l0,
    decoder_prelude_first,
    decoder_unknown_first,
    decoder_unknown_second,
    decoder_prelude_second,
    continuity_output,
]
for parent, child in zip(
    continuity_sequence[:-1], continuity_sequence[1:], strict=True
):
    connect(parent, child)

continuity_index = continuity_graph.build_structure_index()
assert [region.kind for region in continuity_index.regions] == [
    RegionKind.PRELUDE,
    RegionKind.LAYER,
    RegionKind.UNKNOWN,
    RegionKind.LAYER,
    RegionKind.LAYER,
    RegionKind.PRELUDE,
]
encoder_prelude_region, encoder_l0_region, transition_region = (
    continuity_index.regions[:3]
)
encoder_l1_region, decoder_l0_region, decoder_prelude_region = (
    continuity_index.regions[3:]
)
assert encoder_prelude_region.nodes == [
    encoder_prepare,
    encoder_prelude,
    encoder_prepare_after_anchor,
]
assert decoder_prelude_region.nodes == [
    decoder_prelude_first,
    decoder_unknown_first,
    decoder_unknown_second,
    decoder_prelude_second,
]
assert encoder_prelude_region is not decoder_prelude_region
assert encoder_l0_region.nodes == [
    encoder_l0_first,
    same_layer_unknown,
    encoder_l0_second,
]
assert transition_region.nodes == [different_layer_unknown]
assert encoder_l1_region.nodes == [encoder_l1]
assert decoder_l0_region.nodes == [decoder_l0]
assert encoder_l0_region.layer_container == "model.encoder.layers"
assert decoder_l0_region.layer_container == "model.decoder.layers"
assert encoder_l0_region is not decoder_l0_region

continuity_positions = {
    node: position for position, node in enumerate(continuity_graph.body)
}
covered_nodes = []
previous_end = None
for region in continuity_index.regions:
    positions = [continuity_positions[node] for node in region.nodes]
    assert positions == list(range(positions[0], positions[-1] + 1))
    if previous_end is not None:
        assert previous_end < positions[0]
    previous_end = positions[-1]
    covered_nodes.extend(region.nodes)
assert covered_nodes == continuity_sequence[1:-1]
assert len(covered_nodes) == len(set(covered_nodes))
assert all(
    continuity_index.node_to_region[node] is region
    for region in continuity_index.regions
    for node in region.nodes
)

# Completion preserves existing annotation fields, does not overwrite ownership,
# and leaves graph-head/tail runs with only one anchor unowned.
completion_graph = Graph({}, "layer_completion_test")
completion_nodes = [
    add(completion_graph, make_node(AddOp, "head_unknown")),
    add(
        completion_graph,
        make_node(AddOp, "l0_left", "model.layers.0.mlp.gate_proj"),
    ),
    add(
        completion_graph,
        make_node(AddOp, "annotated_unknown", "mlp.down_proj"),
    ),
    add(
        completion_graph,
        make_node(AddOp, "l0_right", "model.layers.0.mlp.up_proj"),
    ),
    add(
        completion_graph,
        make_node(AddOp, "owned_l1", "model.layers.1.mlp.down_proj"),
    ),
    add(completion_graph, make_node(AddOp, "tail_unknown")),
]
completion_annotations = (
    ModuleStructureAnalyzer().analyze(completion_graph).node_annotations
)
annotated_unknown = completion_annotations[completion_nodes[2]]
assert (
    annotated_unknown.layer_index,
    annotated_unknown.layer_container,
    annotated_unknown.component,
    annotated_unknown.subcomponent,
) == (0, "model.layers", "mlp", "down_proj")
assert completion_annotations[completion_nodes[4]].layer_index == 1
assert (
    completion_annotations[completion_nodes[4]].layer_container
    == "model.layers"
)
assert completion_nodes[0] not in completion_annotations
assert completion_nodes[5] not in completion_annotations


# Multi-stack layer identity keeps local indices scoped to their full container.
multi_stack_graph = Graph({}, "multi_stack_layer_identity_test")
multi_stack_input = add(
    multi_stack_graph,
    make_node(PlaceholderOp, "multi_stack_input"),
    NodeType.InputNode,
)
multi_stack_nodes = [
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "encoder_0_left",
            "model.encoder.layers.0.self_attn.q_proj",
        ),
    ),
    add(multi_stack_graph, make_node(UnsqueezeOp, "encoder_0_completed")),
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "encoder_0_right",
            "model.encoder.layers.0.mlp.down_proj",
        ),
    ),
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "encoder_1",
            "model.encoder.layers.1.self_attn.q_proj",
        ),
    ),
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "decoder_0_left",
            "model.decoder.layers.0.self_attn.q_proj",
        ),
    ),
    add(multi_stack_graph, make_node(UnsqueezeOp, "decoder_0_completed")),
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "decoder_0_right",
            "model.decoder.layers.0.mlp.down_proj",
        ),
    ),
    add(
        multi_stack_graph,
        make_node(
            MatmulOp,
            "decoder_1",
            "model.decoder.layers.1.self_attn.q_proj",
        ),
    ),
]
multi_stack_output = add(
    multi_stack_graph,
    make_node(OutputOp, "multi_stack_output"),
)
previous = multi_stack_input
for node in multi_stack_nodes:
    connect(previous, node)
    previous = node
connect(previous, multi_stack_output)

multi_stack_index = multi_stack_graph.build_structure_index()
multi_stack_regions = [
    region
    for region in multi_stack_index.regions
    if isinstance(region, LayerRegion)
]
assert (
    multi_stack_index.annotations[multi_stack_nodes[0]].layer_container,
    multi_stack_index.annotations[multi_stack_nodes[0]].layer_index,
) == ("model.encoder.layers", 0)
assert (
    multi_stack_index.annotations[multi_stack_nodes[4]].layer_container,
    multi_stack_index.annotations[multi_stack_nodes[4]].layer_index,
) == ("model.decoder.layers", 0)
assert [
    (region.layer_container, region.layer_index)
    for region in multi_stack_regions
] == [
    ("model.encoder.layers", 0),
    ("model.encoder.layers", 1),
    ("model.decoder.layers", 0),
    ("model.decoder.layers", 1),
]
assert [
    (region.layer_container, region.layer_index)
    for region in multi_stack_index.regions
] == [
    ("model.encoder.layers", 0),
    ("model.encoder.layers", 1),
    ("model.decoder.layers", 0),
    ("model.decoder.layers", 1),
]
assert (
    multi_stack_index.annotations[multi_stack_nodes[1]].layer_container,
    multi_stack_index.annotations[multi_stack_nodes[1]].layer_index,
) == ("model.encoder.layers", 0)
assert (
    multi_stack_index.annotations[multi_stack_nodes[5]].layer_container,
    multi_stack_index.annotations[multi_stack_nodes[5]].layer_index,
) == ("model.decoder.layers", 0)

# Matching local indices in different containers do not bound completion.
cross_stack_completion_graph = Graph({}, "cross_stack_completion_test")
cross_stack_left = add(
    cross_stack_completion_graph,
    make_node(
        MatmulOp,
        "cross_stack_left",
        "model.encoder.layers.0.self_attn.q_proj",
    ),
)
cross_stack_unknown = add(
    cross_stack_completion_graph,
    make_node(UnsqueezeOp, "cross_stack_unknown"),
)
cross_stack_right = add(
    cross_stack_completion_graph,
    make_node(
        MatmulOp,
        "cross_stack_right",
        "model.decoder.layers.0.self_attn.q_proj",
    ),
)
cross_stack_completion_index = (
    cross_stack_completion_graph.build_structure_index()
)
assert cross_stack_unknown not in cross_stack_completion_index.annotations
assert not isinstance(
    cross_stack_completion_index.node_to_region[cross_stack_unknown],
    LayerRegion,
)

# Residual refinement counts same-layer parents by full layer key.
cross_stack_residual_graph = Graph({}, "cross_stack_residual_test")
cross_stack_mlp = add(
    cross_stack_residual_graph,
    make_node(
        MatmulOp,
        "cross_stack_mlp",
        "model.encoder.layers.0.mlp.down_proj",
    ),
)
cross_stack_attention = add(
    cross_stack_residual_graph,
    make_node(
        MatmulOp,
        "cross_stack_attention",
        "model.decoder.layers.0.self_attn.q_proj",
    ),
)
cross_stack_add = add(
    cross_stack_residual_graph,
    make_node(AddOp, "cross_stack_add"),
)
connect(cross_stack_mlp, cross_stack_add)
connect(cross_stack_attention, cross_stack_add)
cross_stack_residual_index = cross_stack_residual_graph.build_structure_index()
assert cross_stack_add not in cross_stack_residual_index.annotations
assert not isinstance(
    cross_stack_residual_index.node_to_region[cross_stack_add],
    LayerRegion,
)


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
graph_ids_before = {
    id(value) for value in gc.get_objects() if type(value) is Graph
}
op_ids_before = {
    id(value) for value in gc.get_objects() if type(value) in op_classes
}
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
    node in body_snapshot
    for region in first_index.regions
    for node in region.nodes
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
no_layer_final = add(
    no_layer_graph, make_node(AddOp, "no_layer_final", "model.norm")
)
no_layer_head = add(
    no_layer_graph, make_node(MatmulOp, "no_layer_head", "lm_head")
)
no_layer_output = add(no_layer_graph, make_node(OutputOp, "no_layer_output"))
no_layer_sequence = [
    no_layer_input,
    no_layer_embedding,
    no_layer_unknown,
    no_layer_final,
    no_layer_head,
    no_layer_output,
]
for parent, child in zip(
    no_layer_sequence[:-1], no_layer_sequence[1:], strict=True
):
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
    layer_container="model.layers",
)
assert residual_index.annotations[mlp_residual] == NodeAnnotation(
    layer_index=13,
    component="residual",
    subcomponent="post_mlp_residual",
    layer_container="model.layers",
)
assert residual_index.annotations[ordinary_add] == NodeAnnotation(
    layer_index=13,
    layer_container="model.layers",
)
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


# Boundary post-MLP residuals close the producing layer without absorbing
# side-module fan-out or the following epilogue.
boundary_graph = Graph({}, "boundary_residual_ownership_test")
boundary_input = add(
    boundary_graph,
    make_node(PlaceholderOp, "boundary_input"),
    NodeType.InputNode,
)
layer0_hidden = add(
    boundary_graph,
    make_node(
        AddOp,
        "layer0_hidden",
        "blocks.0.attn.proj",
    ),
)
completed_skip = add(
    boundary_graph,
    make_node(AddOp, "completed_skip"),
)
layer0_mlp = add(
    boundary_graph,
    make_node(MatmulOp, "layer0_mlp", "blocks.0.mlp.down_proj"),
)
internal_residual = add(
    boundary_graph,
    make_node(AddOp, "internal_boundary_sum"),
)
side_module = add(
    boundary_graph,
    make_node(AddOp, "side_module", "deepstack_merger_list.0.norm"),
)
layer1_hidden = add(
    boundary_graph,
    make_node(
        AddOp,
        "layer1_hidden",
        "blocks.1.attn.proj",
    ),
)
layer1_mlp = add(
    boundary_graph,
    make_node(MatmulOp, "layer1_mlp", "blocks.1.mlp.down_proj"),
)
final_residual = add(
    boundary_graph,
    make_node(AddOp, "tail_boundary_sum"),
)
main_merger = add(
    boundary_graph,
    make_node(AddOp, "main_merger", "merger.norm"),
)
boundary_output = add(
    boundary_graph,
    make_node(OutputOp, "boundary_output"),
)
connect(boundary_input, layer0_hidden)
connect(layer0_hidden, completed_skip)
connect(completed_skip, internal_residual)
connect(layer0_mlp, internal_residual)
connect(internal_residual, side_module)
connect(internal_residual, layer1_hidden)
connect(layer1_hidden, layer1_mlp)
connect(layer1_hidden, final_residual)
connect(layer1_mlp, final_residual)
connect(final_residual, main_merger)
connect(side_module, boundary_output)
connect(main_merger, boundary_output)

boundary_index = boundary_graph.build_structure_index()
assert boundary_index.annotations[completed_skip] == NodeAnnotation(
    layer_index=0,
    layer_container="blocks",
)
assert boundary_index.annotations[internal_residual] == NodeAnnotation(
    layer_index=0,
    component="residual",
    subcomponent="post_mlp_residual",
    layer_container="blocks",
)
assert boundary_index.node_to_region[internal_residual] is next(
    region
    for region in boundary_index.regions
    if isinstance(region, LayerRegion) and region.layer_index == 0
)
assert (
    boundary_index.annotations.get(side_module, NodeAnnotation()).layer_index
    is None
)
assert not isinstance(boundary_index.node_to_region[side_module], LayerRegion)
assert boundary_index.annotations[final_residual] == NodeAnnotation(
    layer_index=1,
    component="residual",
    subcomponent="post_mlp_residual",
    layer_container="blocks",
)
assert boundary_index.node_to_region[final_residual] is next(
    region
    for region in boundary_index.regions
    if isinstance(region, LayerRegion) and region.layer_index == 1
)
assert (
    boundary_index.annotations.get(main_merger, NodeAnnotation()).layer_index
    is None
)
assert not isinstance(boundary_index.node_to_region[main_merger], LayerRegion)


def append_functional_rmsnorm(graph, prefix, hidden, weight):
    power = add(graph, make_node(PowOp, f"{prefix}_power"))
    mean = add(graph, make_node(MeanOp, f"{prefix}_mean"))
    epsilon_add = add(graph, make_node(AddOp, f"{prefix}_epsilon"))
    rsqrt = add(graph, make_node(RsqrtOp, f"{prefix}_rsqrt"))
    normalized = add(graph, make_node(MulOp, f"{prefix}_normalized"))
    weighted = add(graph, make_node(MulOp, f"{prefix}_weighted"))
    connect(hidden, power)
    connect(power, mean)
    connect(mean, epsilon_add)
    connect(epsilon_add, rsqrt)
    connect(hidden, normalized)
    connect(rsqrt, normalized)
    connect(normalized, weighted)
    connect(weight, weighted)
    return (power, mean, epsilon_add, rsqrt, normalized, weighted)


def append_layer_exit(graph, prefix, layer_index, hidden):
    attention = add(
        graph,
        make_node(
            MatmulOp,
            f"{prefix}_attention",
            f"layers.{layer_index}.self_attn.q_proj",
        ),
    )
    mlp = add(
        graph,
        make_node(
            MatmulOp,
            f"{prefix}_mlp",
            f"layers.{layer_index}.mlp.down_proj",
        ),
    )
    residual = add(graph, make_node(AddOp, f"{prefix}_residual"))
    connect(hidden, attention)
    connect(attention, mlp)
    connect(attention, residual)
    connect(mlp, residual)
    return attention, residual


# Functional RMSNorm adjacency and deepstack mixing do not cross containers.
cross_stack_norm_graph = Graph({}, "cross_stack_rmsnorm_boundary_test")
cross_stack_norm_input = add(
    cross_stack_norm_graph,
    make_node(PlaceholderOp, "cross_stack_norm_input"),
    NodeType.InputNode,
)
cross_stack_norm_weight = add(
    cross_stack_norm_graph,
    make_node(PlaceholderOp, "cross_stack_norm_weight"),
    NodeType.FakeNode,
)
cross_stack_encoder_attention = add(
    cross_stack_norm_graph,
    make_node(
        MatmulOp,
        "cross_stack_encoder_attention",
        "model.encoder.layers.0.self_attn.q_proj",
    ),
)
cross_stack_encoder_mlp = add(
    cross_stack_norm_graph,
    make_node(
        MatmulOp,
        "cross_stack_encoder_mlp",
        "model.encoder.layers.0.mlp.down_proj",
    ),
)
cross_stack_encoder_residual = add(
    cross_stack_norm_graph,
    make_node(AddOp, "cross_stack_encoder_residual"),
)
connect(cross_stack_encoder_attention, cross_stack_encoder_residual)
connect(cross_stack_encoder_mlp, cross_stack_encoder_residual)
cross_stack_mix = add(
    cross_stack_norm_graph,
    make_node(AddOp, "cross_stack_mix"),
)
connect(cross_stack_encoder_residual, cross_stack_mix)
connect(cross_stack_norm_input, cross_stack_mix)
cross_stack_norm = append_functional_rmsnorm(
    cross_stack_norm_graph,
    "cross_stack_norm",
    cross_stack_mix,
    cross_stack_norm_weight,
)
cross_stack_decoder_consumer = add(
    cross_stack_norm_graph,
    make_node(
        MatmulOp,
        "cross_stack_decoder_consumer",
        "model.decoder.layers.1.self_attn.q_proj",
    ),
)
connect(cross_stack_norm[-1], cross_stack_decoder_consumer)
cross_stack_norm_index = cross_stack_norm_graph.build_structure_index()
assert cross_stack_norm_index.annotations[
    cross_stack_encoder_residual
] == NodeAnnotation(
    layer_index=0,
    component="residual",
    subcomponent="post_mlp_residual",
    layer_container="model.encoder.layers",
)
assert cross_stack_mix not in cross_stack_norm_index.annotations
assert all(
    node not in cross_stack_norm_index.annotations for node in cross_stack_norm
)

# First-layer ownership is computed independently for each container.
per_stack_first_graph = Graph({}, "per_stack_first_layer_test")
per_stack_first_input = add(
    per_stack_first_graph,
    make_node(PlaceholderOp, "per_stack_first_input"),
    NodeType.InputNode,
)
per_stack_first_weight = add(
    per_stack_first_graph,
    make_node(PlaceholderOp, "per_stack_first_weight"),
    NodeType.FakeNode,
)
add(
    per_stack_first_graph,
    make_node(
        MatmulOp,
        "per_stack_encoder_0",
        "model.encoder.layers.0.self_attn.q_proj",
    ),
)
per_stack_first_norm = append_functional_rmsnorm(
    per_stack_first_graph,
    "per_stack_first_norm",
    per_stack_first_input,
    per_stack_first_weight,
)
per_stack_decoder_3 = add(
    per_stack_first_graph,
    make_node(
        MatmulOp,
        "per_stack_decoder_3",
        "model.decoder.layers.3.self_attn.q_proj",
    ),
)
connect(per_stack_first_norm[-1], per_stack_decoder_3)
per_stack_first_index = per_stack_first_graph.build_structure_index()
assert all(
    (
        per_stack_first_index.annotations[node].layer_container,
        per_stack_first_index.annotations[node].layer_index,
    )
    == ("model.decoder.layers", 3)
    for node in per_stack_first_norm
)


# Functional boundary RMSNorm chains split ownership at adjacent layer anchors.
# The same topology closes the first and last layer without absorbing runtime
# preparation, final norm, or the output head.
decoder_boundary_graph = Graph({}, "functional_boundary_ownership_test")
decoder_inputs = [
    add(
        decoder_boundary_graph,
        make_node(PlaceholderOp, name),
        NodeType.InputNode,
    )
    for name in ("decoder_hidden", "runtime_trig", "side_input")
]
decoder_weights = [
    add(
        decoder_boundary_graph,
        make_node(PlaceholderOp, name),
        NodeType.FakeNode,
    )
    for name in (
        "layer0_weight",
        "layer1_weight",
        "final_weight",
        "head_weight",
    )
]

(
    layer0_weight,
    layer1_weight,
    final_weight,
    head_weight,
) = decoder_weights

decoder_hidden, runtime_trig, side_input = decoder_inputs

runtime_prepare0 = add(
    decoder_boundary_graph,
    make_node(UnsqueezeOp, "trig0"),
)
runtime_prepare1 = add(
    decoder_boundary_graph,
    make_node(UnsqueezeOp, "trig1"),
)
connect(runtime_trig, runtime_prepare0)
connect(runtime_prepare0, runtime_prepare1)

layer0_input_norm = append_functional_rmsnorm(
    decoder_boundary_graph,
    "entry_norm",
    decoder_hidden,
    layer0_weight,
)


layer0_hidden, layer0_residual = append_layer_exit(
    decoder_boundary_graph, "first", 0, layer0_input_norm[-1]
)
deepstack_mix = add(decoder_boundary_graph, make_node(AddOp, "side_input_mix"))
connect(layer0_residual, deepstack_mix)
connect(side_input, deepstack_mix)

layer1_input_norm = append_functional_rmsnorm(
    decoder_boundary_graph, "internal_norm", deepstack_mix, layer1_weight
)
layer1_hidden, last_residual = append_layer_exit(
    decoder_boundary_graph, "last", 1, layer1_input_norm[-1]
)

final_norm = append_functional_rmsnorm(
    decoder_boundary_graph, "output_norm", last_residual, final_weight
)
head_matmul = add(
    decoder_boundary_graph, make_node(MatmulOp, "head_mm", "lm_head")
)
decoder_boundary_output = add(
    decoder_boundary_graph, make_node(OutputOp, "decoder_output")
)
connect(final_norm[-1], head_matmul)
connect(head_weight, head_matmul)
connect(head_matmul, decoder_boundary_output)

decoder_boundary_index = decoder_boundary_graph.build_structure_index()
assert all(
    decoder_boundary_index.annotations[node]
    == NodeAnnotation(
        0,
        "norm",
        "input_layernorm",
        layer_container="layers",
    )
    for node in layer0_input_norm
)
assert all(
    decoder_boundary_index.node_to_region[node]
    is decoder_boundary_index.node_to_region[layer0_hidden]
    for node in layer0_input_norm
)
assert decoder_boundary_index.annotations[layer0_residual] == NodeAnnotation(
    0,
    "residual",
    "post_mlp_residual",
    layer_container="layers",
)
assert decoder_boundary_index.annotations[deepstack_mix] == NodeAnnotation(
    layer_index=0,
    layer_container="layers",
)

assert all(
    node not in decoder_boundary_index.annotations for node in final_norm
)

assert decoder_boundary_index.annotations[head_matmul] == NodeAnnotation(
    component="lm_head"
)

assert all(
    not isinstance(
        decoder_boundary_index.node_to_region[node],
        LayerRegion,
    )
    for node in final_norm + (head_matmul,)
)

assert (
    decoder_boundary_index.node_to_region[final_norm[0]].kind
    is RegionKind.UNKNOWN
)
assert (
    decoder_boundary_index.node_to_region[head_matmul].kind
    is RegionKind.EPILOGUE
)

# assert decoder_boundary_index.annotations[last_residual] == NodeAnnotation(
#     1, "residual", "post_mlp_residual"
# )
# assert all(
#     decoder_boundary_index.annotations[node]
#     == NodeAnnotation(None, "norm", "final_norm")
#     for node in final_norm
# )


# assert all(
#     decoder_boundary_index.annotations[node]
#     == NodeAnnotation(component="lm_head")
#     for node in head_nodes
# )
# assert all(
#     not isinstance(decoder_boundary_index.node_to_region[node], LayerRegion)
#     for node in final_norm + head_nodes
# )
assert decoder_boundary_index.regions[0].kind is RegionKind.UNKNOWN
assert decoder_boundary_index.regions[0].nodes == [
    runtime_prepare0,
    runtime_prepare1,
]


# Multiple directly consuming layer anchors do not provide a unique boundary.
ambiguous_boundary_graph = Graph({}, "ambiguous_functional_boundary_test")
ambiguous_hidden, ambiguous_side = [
    add(
        ambiguous_boundary_graph,
        make_node(PlaceholderOp, name),
        NodeType.InputNode,
    )
    for name in ("ambiguous_hidden", "ambiguous_side")
]
ambiguous_weight = add(
    ambiguous_boundary_graph,
    make_node(PlaceholderOp, "ambiguous_weight"),
    NodeType.FakeNode,
)
_, ambiguous_residual = append_layer_exit(
    ambiguous_boundary_graph, "ambiguous_left", 4, ambiguous_hidden
)
ambiguous_mix = add(ambiguous_boundary_graph, make_node(AddOp, "ambiguous_mix"))
connect(ambiguous_residual, ambiguous_mix)
connect(ambiguous_side, ambiguous_mix)
ambiguous_norm = append_functional_rmsnorm(
    ambiguous_boundary_graph, "ambiguous_norm", ambiguous_mix, ambiguous_weight
)
ambiguous_layer5 = add(
    ambiguous_boundary_graph,
    make_node(MatmulOp, "ambiguous_layer5", "layers.5.self_attn.q_proj"),
)
ambiguous_layer6 = add(
    ambiguous_boundary_graph,
    make_node(MatmulOp, "ambiguous_layer6", "layers.6.self_attn.q_proj"),
)
ambiguous_output = add(
    ambiguous_boundary_graph, make_node(OutputOp, "ambiguous_output")
)
connect(ambiguous_norm[-1], ambiguous_layer5)
connect(ambiguous_norm[-1], ambiguous_layer6)
connect(ambiguous_layer5, ambiguous_output)
connect(ambiguous_layer6, ambiguous_output)

ambiguous_boundary_index = ambiguous_boundary_graph.build_structure_index()
assert ambiguous_boundary_index.annotations[
    ambiguous_residual
] == NodeAnnotation(
    4,
    "residual",
    "post_mlp_residual",
    layer_container="layers",
)
assert ambiguous_mix not in ambiguous_boundary_index.annotations
assert all(
    node not in ambiguous_boundary_index.annotations for node in ambiguous_norm
)
assert not isinstance(
    ambiguous_boundary_index.node_to_region[ambiguous_mix], LayerRegion
)


# An Add with only an MLP input is not sufficient residual topology.
non_residual_graph = Graph({}, "non_residual_boundary_add_test")
non_residual_input = add(
    non_residual_graph,
    make_node(PlaceholderOp, "non_residual_input"),
    NodeType.InputNode,
)
non_residual_mlp = add(
    non_residual_graph,
    make_node(MatmulOp, "non_residual_mlp", "blocks.7.mlp.down_proj"),
)
non_residual_add = add(
    non_residual_graph,
    make_node(AddOp, "non_residual_add"),
)
non_residual_output = add(
    non_residual_graph,
    make_node(OutputOp, "non_residual_output"),
)
connect(non_residual_input, non_residual_mlp)
connect(non_residual_input, non_residual_add)
connect(non_residual_mlp, non_residual_add)
connect(non_residual_add, non_residual_output)
non_residual_index = non_residual_graph.build_structure_index()
assert non_residual_add not in non_residual_index.annotations
assert not isinstance(
    non_residual_index.node_to_region[non_residual_add], LayerRegion
)


# Template fingerprint and grouping analysis tests.


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
assert isinstance(
    ModuleStructureAnalyzer().analyze(graph), StructureAnalysisResult
)
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
assert (
    sum(1 for item in template_index.template_groups if item.canonical_form)
    == 1
)
group_canonical = json.loads(group.canonical_form)
assert all(
    "argument_result_indices" not in item["attributes"]
    for item in group_canonical["nodes"]
)


def deepseek_like_28_layer_graph():
    graph = Graph({}, "deepseek_like_partition")
    graph_input = add(graph, node(PlaceholderOp, "tokens"), NodeType.InputNode)
    embedding = add(
        graph,
        node(
            AddOp, "embedding", "model.embed_tokens", "aten.embedding.default"
        ),
    )
    bind(graph_input, embedding)
    previous = embedding
    for layer_index in range(28):
        parameter = add(
            graph,
            node(PlaceholderOp, f"model.layers.{layer_index}.weight"),
            NodeType.FakeNode,
        )
        attention = add(
            graph,
            node(
                MatmulOp,
                f"layer_{layer_index}_attention",
                f"model.layers.{layer_index}.self_attn.q_proj",
                "aten.mm.default",
            ),
        )
        mlp = add(
            graph,
            node(
                AddOp,
                f"layer_{layer_index}_mlp",
                f"model.layers.{layer_index}.mlp.down_proj",
                "aten.add.Tensor",
            ),
        )
        bind(previous, attention)
        bind(parameter, attention)
        bind(attention, mlp)
        mlp.add_argument(1.0)
        previous = mlp
    final_norm = add(graph, node(AddOp, "final_norm", "model.norm"))
    lm_head = add(graph, node(MatmulOp, "lm_head", "lm_head"))
    output = add(graph, node(OutputOp, "output"))
    bind(previous, final_norm)
    bind(final_norm, lm_head)
    bind(lm_head, output)
    return graph


# Lightweight DeepSeek-shaped coverage: preserve all 28 Layer instances,
# their order, one shared representative, and singleton prelude/epilogue units.
deepseek_graph = deepseek_like_28_layer_graph()
deepseek_body = list(deepseek_graph.body)
deepseek_plan = build_transformer_partition_plan(deepseek_graph)
deepseek_plan_again = build_transformer_partition_plan(deepseek_graph)
deepseek_layers = [
    region
    for region in deepseek_plan.partition_sequence
    if isinstance(region, LayerRegion)
]
assert [region.kind for region in deepseek_plan.partition_sequence] == [
    RegionKind.PRELUDE,
    *([RegionKind.LAYER] * 28),
    RegionKind.EPILOGUE,
]
assert [region.layer_index for region in deepseek_layers] == list(range(28))
assert len(deepseek_plan.template_index.template_groups) == 1
deepseek_group = deepseek_plan.template_index.template_groups[0]
assert deepseek_group.representative is deepseek_layers[0]
assert deepseek_group.instances == deepseek_layers
assert deepseek_plan.template_index.non_reusable_regions == []
assert [
    deepseek_plan.region_to_template_id[region] for region in deepseek_layers
] == [1] * 28
assert [
    tuple(input_ref.kind for input_ref in binding.ordered_inputs)
    for binding in deepseek_plan.instance_bindings[1:-1]
] == [(RegionInputKind.DATA, RegionInputKind.PARAMETER)] * 28
assert deepseek_plan_again == deepseek_plan
assert deepseek_plan_again.structure_index is deepseek_plan.structure_index
assert deepseek_plan_again.template_index is deepseek_plan.template_index
assert deepseek_graph.body == deepseek_body


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
    parameter0 = add(graph, node(PlaceholderOp, "weight0"), NodeType.FakeNode)
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
    assert fingerprint(single_layer(difference)).digest != base_digest, (
        difference
    )

for ignored_field in ("newshape", "op_type_classification"):
    assert fingerprint(single_layer(ignored_field)).digest == base_digest, (
        ignored_field
    )


def call_graph(callee, argument_result_index):
    graph = Graph({}, "call_template")
    graph_input = add(
        graph, node(PlaceholderOp, "call_input"), NodeType.InputNode
    )
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
assert not hasattr(
    unique_template.region_fingerprints[unique_layer], "canonical_form"
)


# Prelude, epilogue, and unknown regions are structurally indexed but excluded.
excluded_graph = Graph({}, "excluded_regions")
excluded_input = add(
    excluded_graph, node(PlaceholderOp, "excluded_input"), NodeType.InputNode
)
embedding = add(excluded_graph, node(AddOp, "embedding", "model.embed_tokens"))
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
    [
        excluded_input,
        embedding,
        layer_node,
        unknown,
        layer_node_1,
        final_norm,
        head,
    ],
    [
        embedding,
        layer_node,
        unknown,
        layer_node_1,
        final_norm,
        head,
        excluded_output,
    ],
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
        (
            r
            for r in result.structure_index.regions
            if isinstance(r, LayerRegion)
        ),
        key=lambda r: r.layer_index,
    )


def two_source_normalization_layers(path_format, extra_path_format=None):
    graph = Graph({}, "source_normalization")
    graph_input = add(
        graph, node(PlaceholderOp, "source_input"), NodeType.InputNode
    )
    for layer_index in range(2):
        layer_node = node(
            AddOp,
            f"source_{layer_index}",
            path_format.format(layer_index),
        )
        if extra_path_format is not None:
            layer_node._source_meta = (
                *layer_node._source_meta,
                SourceMeta(
                    module_path=extra_path_format.format(layer_index),
                    module_class="example.DecoderLayer",
                ),
            )
        layer_node = add(graph, layer_node)
        bind(graph_input, layer_node)
        bind(
            layer_node,
            add(graph, node(OutputOp, f"source_out_{layer_index}")),
        )
    return graph.analyze_structure(True)


# All accepted grammar families normalize their selected layer occurrence.
for normalized_path_format in (
    "model.layers.{}.mlp.down_proj",
    "encoder.layer.{}.attention",
    "layers.{}.self_attn.q_proj",
    "blocks.{}.attn.qkv",
):
    normalized_result = two_source_normalization_layers(normalized_path_format)
    normalized_layers = layer_regions(normalized_result)
    assert [region.layer_index for region in normalized_layers] == [0, 1]
    assert (
        len(
            {
                normalized_result.template_index.region_fingerprints[
                    region
                ].digest
                for region in normalized_layers
            }
        )
        == 1
    )

# An unaccepted indexed source is preserved even when another SourceMeta puts
# the Op in a LayerRegion; its differing number remains fingerprint-visible.
deepstack_result = two_source_normalization_layers(
    "blocks.{}.attn.qkv",
    "deepstack_merger_list.{}.norm",
)
deepstack_layers = layer_regions(deepstack_result)
assert (
    len(
        {
            deepstack_result.template_index.region_fingerprints[region].digest
            for region in deepstack_layers
        }
    )
    == 2
)
assert deepstack_result.template_index.template_groups == []


# Encoder layer numbers are normalized out of SourceMeta before
# fingerprinting, just like decoder model.layers.<N> paths.
encoder_graph = Graph({}, "encoder_layer_source_normalization")
encoder_input = add(
    encoder_graph,
    node(PlaceholderOp, "encoder_input"),
    NodeType.InputNode,
)

for layer_index in range(2):
    encoder_node = add(
        encoder_graph,
        node(
            AddOp,
            f"encoder_{layer_index}",
            f"bert.encoder.layer.{layer_index}.output.dense",
        ),
    )
    bind(encoder_input, encoder_node)
    bind(
        encoder_node,
        add(
            encoder_graph,
            node(OutputOp, f"encoder_out_{layer_index}"),
        ),
    )

encoder_result = encoder_graph.analyze_structure(True)
encoder_layers = layer_regions(encoder_result)
encoder_template = encoder_result.template_index

assert [region.layer_index for region in encoder_layers] == [0, 1]
assert len(encoder_template.region_fingerprints) == 2
assert (
    len(
        {
            fingerprint.digest
            for fingerprint in encoder_template.region_fingerprints.values()
        }
    )
    == 1
)
assert encoder_template.non_reusable_regions == []
assert len(encoder_template.template_groups) == 1
assert encoder_template.template_groups[0].instances == encoder_layers


mixed_graph = Graph({}, "opaque_and_reusable")
mixed_input = add(
    mixed_graph, node(PlaceholderOp, "mixed_input"), NodeType.InputNode
)
for layer_index, literal in enumerate((OpaqueLiteral(), 1.0, 1.0)):
    mixed_node = add(
        mixed_graph,
        node(AddOp, f"mixed_{layer_index}", f"model.layers.{layer_index}.mlp"),
    )
    bind(mixed_input, mixed_node)
    mixed_node.add_argument(literal)
    bind(
        mixed_node, add(mixed_graph, node(OutputOp, f"mixed_out_{layer_index}"))
    )
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
dict_input = add(
    dict_graph, node(PlaceholderOp, "dict_input"), NodeType.InputNode
)
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
    consumer.kwargs["operands"] = dict(
        reversed(pairs) if layer_index else pairs
    )
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
assert (
    cached_graph.analyze_structure(False).template_index
    is first_result.template_index
)
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
assert {
    id(value) for value in gc.get_objects() if type(value) is Graph
} == graph_ids
assert {
    id(value) for value in gc.get_objects() if type(value) in op_classes
} == op_ids
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
