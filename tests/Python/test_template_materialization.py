# RUN: %PYTHON %s

import importlib.util
import json
import os
import tempfile

from buddy.compiler.graph import (
    Graph,
    NodeType,
    TemplatePartitionedGraphDriver,
    TensorDType,
    build_template_materialization_plan,
)
from buddy.compiler.graph.operation import (
    AddOp,
    CallOp,
    DivOp,
    GetItemOp,
    MulOp,
    OutputOp,
    PlaceholderOp,
    TensorConstantOp,
)
from buddy.compiler.graph.region_analysis import (
    LayerRegion,
    RegionInputKind,
    RegionInputRef,
)
from buddy.compiler.graph.source_meta import SourceMeta
from buddy.compiler.ops import func, tosa


def node(cls, name, shape, path=None):
    op = cls()
    op.name = name
    op.tensor_meta = {"shape": shape, "dtype": TensorDType.Float32}
    if path is not None:
        op._source_meta = (
            SourceMeta(
                module_path=path,
                module_class="example.DecoderLayer",
            ),
        )
    return op


def add(graph, op, kind=NodeType.OtherNode):
    graph.add_node(op, kind)
    return op


def bind(parent, child):
    parent.add_children(child.name)
    child.add_parent(parent.name)
    child.add_argument(parent.name)


def expect_value_error(message, callback):
    try:
        callback()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def materialization_analysis(graph):
    analysis = graph.analyze_structure(True)
    return analysis, build_template_materialization_plan(
        graph, analysis.structure_index, analysis.template_index
    )


def single_layer_graph_with_constant():
    graph = Graph({**tosa.ops_registry, **func.ops_registry}, "constant_decode")
    graph_input = add(
        graph,
        node(PlaceholderOp, "constant_tokens", (2, 4)),
        NodeType.InputNode,
    )
    constant = add(graph, node(TensorConstantOp, "external_constant", (2, 4)))
    layer = add(
        graph,
        node(
            AddOp,
            "constant_layer",
            (2, 4),
            "model.layers.0.mlp.down_proj",
        ),
    )
    bind(graph_input, layer)
    bind(constant, layer)
    output = add(graph, node(OutputOp, "constant_output", ()))
    bind(layer, output)
    return graph


def local_multi_result_graph(nested):
    graph = Graph({**tosa.ops_registry, **func.ops_registry}, "forward_decode")
    graph_input = add(
        graph,
        node(PlaceholderOp, "multi_tokens", (2, 4)),
        NodeType.InputNode,
    )
    producer = add(
        graph,
        node(
            CallOp,
            "multi_producer",
            [(2, 4), (2, 4)],
            "model.layers.0.self_attn.q_proj",
        ),
    )
    producer.tensor_meta = {
        "shape": [(2, 4), (2, 4)],
        "dtype": [TensorDType.Float32, TensorDType.Float32],
    }
    producer.call_func_name = "multi_source"
    bind(graph_input, producer)
    consumer = add(
        graph,
        node(
            AddOp,
            "multi_consumer",
            (2, 4),
            "model.layers.0.mlp.down_proj",
        ),
    )
    producer.add_children(consumer.name)
    consumer.add_parent(producer.name)
    consumer.add_argument(
        [producer.name] if nested else producer.name,
        1,
    )
    output = add(graph, node(OutputOp, "multi_output", ()))
    bind(consumer, output)
    return graph


def five_region_graph(func_name="forward_decode", output_indices=(4,)):
    ops_registry = {**tosa.ops_registry, **func.ops_registry}
    graph = Graph(ops_registry, func_name)
    graph_input = add(
        graph,
        node(PlaceholderOp, "tokens", (2, 4)),
        NodeType.InputNode,
    )
    parameters = []
    for index, shape in enumerate(
        ((2, 4), (2, 4), (2, 4), (2, 4), (2, 4))
    ):
        parameters.append(
            add(
                graph,
                node(PlaceholderOp, f"weight{index}", shape),
                NodeType.FakeNode,
            )
        )

    classes = (MulOp, AddOp, AddOp, AddOp, DivOp)
    regions = []
    previous = graph_input
    for index, cls in enumerate(classes):
        op = add(
            graph,
            node(
                cls,
                f"layer{index}",
                (2, 4),
                f"model.layers.{index}.self_attn.q_proj",
            ),
        )
        bind(previous, op)
        bind(parameters[index], op)
        regions.append(op)
        previous = op
    output = add(graph, node(OutputOp, "output", ()))
    for index in output_indices:
        bind(regions[index], output)
    return graph, parameters, regions


# External TensorConstant inputs fail while building the plan, before a Driver
# can construct a combined forward graph.
constant_graph = single_layer_graph_with_constant()
constant_analysis = constant_graph.analyze_structure(True)
expect_value_error(
    "external TensorConstant inputs are not supported by template "
    "materialization V1",
    lambda: build_template_materialization_plan(
        constant_graph,
        constant_analysis.structure_index,
        constant_analysis.template_index,
    ),
)

# A nested top-level operand cannot safely share one non-zero result index.
nested_graph = local_multi_result_graph(True)
nested_analysis, nested_plan = materialization_analysis(nested_graph)
nested_driver = TemplatePartitionedGraphDriver(
    nested_graph, nested_analysis.structure_index, nested_plan
)
expect_value_error(
    "nested operands with non-zero result indices are not supported by "
    "template materialization V1",
    nested_driver.build_template_subgraphs,
)

# The same non-zero result index remains supported for a flat operand.
flat_graph = local_multi_result_graph(False)
flat_analysis, flat_plan = materialization_analysis(flat_graph)
flat_driver = TemplatePartitionedGraphDriver(
    flat_graph, flat_analysis.structure_index, flat_plan
)
flat_subgraph = flat_driver.build_template_subgraphs()[0]
flat_consumer = next(
    op for op in flat_subgraph.body if op.name == "multi_consumer"
)
assert flat_consumer.args == ["multi_producer"]
assert flat_consumer._args_index == [1]

# Explicit STATE is reserved even when a caller manually adds it to an
# otherwise valid Region interface. KV/cache graph inputs remain DATA in V1.
state_graph, _, _ = five_region_graph()
state_analysis = state_graph.analyze_structure(True)
state_region = state_analysis.structure_index.regions[0]
state_input = state_region.interface.ordered_inputs[0]
state_region.interface.ordered_inputs[0] = RegionInputRef(
    RegionInputKind.STATE, state_input.value
)
state_region.interface.data_inputs.remove(state_input.value.op)
state_region.interface.state_inputs.append(state_input.value.op)
expect_value_error(
    "explicit RegionInputKind.STATE and state_inputs/state_outputs are not "
    "supported by template materialization V1",
    lambda: build_template_materialization_plan(
        state_graph,
        state_analysis.structure_index,
        state_analysis.template_index,
    ),
)


graph, parameters, region_nodes = five_region_graph()
analysis = graph.analyze_structure(True)
structure_index = analysis.structure_index
template_index = analysis.template_index
regions = structure_index.regions

# Plan: singleton, three repeated instances, singleton.
assert len(regions) == 5
assert all(isinstance(region, LayerRegion) for region in regions)
plan = build_template_materialization_plan(
    graph, structure_index, template_index
)
assert len(plan.templates) == 3
assert [unit.template_id for unit in plan.templates] == [0, 1, 2]
assert [plan.region_to_template_id[region] for region in regions] == [
    0,
    1,
    1,
    1,
    2,
]
assert plan.templates[1].representative is regions[1]
assert plan.templates[1].instances == tuple(regions[1:4])
assert plan.parameter_indices == {
    parameter: index for index, parameter in enumerate(parameters)
}
for region in regions:
    assert [item.kind for item in region.interface.ordered_inputs] == [
        RegionInputKind.DATA,
        RegionInputKind.PARAMETER,
    ]

# Three repeated Region instances materialize one subgraph.
driver = TemplatePartitionedGraphDriver(graph, structure_index, plan)
subgraphs = driver.build_template_subgraphs()
assert len(subgraphs) == 3
assert [subgraph._func_name for subgraph in subgraphs] == [
    "subgraph0_decode0",
    "subgraph0_decode1",
    "subgraph0_decode2",
]
assert [op.name for op in subgraphs[1].body if isinstance(op, AddOp)] == [
    region_nodes[1].name
]
for subgraph in subgraphs:
    subgraph.lower_to_top_level_ir()

# Combined forward is Region-instance ordered and reuses the template symbol.
combined = driver.construct_template_combined_main_graph(True)
calls = [op for op in driver.combined_graph.body if isinstance(op, CallOp)]
assert [call.call_func_name for call in calls] == [
    "subgraph0_decode0",
    "subgraph0_decode1",
    "subgraph0_decode1",
    "subgraph0_decode1",
    "subgraph0_decode2",
]
for index, call in enumerate(calls):
    assert parameters[index].name in call.args
assert calls[2].args[0] == calls[1].name
assert calls[2]._args_index[0] == 0
assert calls[3].args[0] == calls[2].name
assert calls[3]._args_index[0] == 0
combined_output = next(
    op for op in driver.combined_graph.body if isinstance(op, OutputOp)
)
assert combined_output.args == [calls[-1].name]
combined_text = str(combined)
assert combined_text.count("call @subgraph0_decode1") == 3


# The graph function name is the only phase identity stored by the Driver.
prefill_graph, prefill_parameters, prefill_region_nodes = five_region_graph(
    "forward_prefill"
)
prefill_analysis, prefill_plan = materialization_analysis(prefill_graph)
prefill_regions = prefill_analysis.structure_index.regions
assert len(prefill_regions) == 5
assert len(prefill_plan.templates) == 3
prefill_driver = TemplatePartitionedGraphDriver(
    prefill_graph, prefill_analysis.structure_index, prefill_plan
)
assert not hasattr(prefill_driver, "_phase")
prefill_subgraphs = prefill_driver.build_template_subgraphs()
assert len(prefill_subgraphs) == 3
assert [subgraph._func_name for subgraph in prefill_subgraphs] == [
    "subgraph0_prefill0",
    "subgraph0_prefill1",
    "subgraph0_prefill2",
]
for subgraph in prefill_subgraphs:
    subgraph.lower_to_top_level_ir()
prefill_combined = prefill_driver.construct_template_combined_main_graph(True)
prefill_calls = [
    op for op in prefill_driver.combined_graph.body if isinstance(op, CallOp)
]
assert len(prefill_calls) == 5
assert [call.call_func_name for call in prefill_calls] == [
    "subgraph0_prefill0",
    "subgraph0_prefill1",
    "subgraph0_prefill1",
    "subgraph0_prefill1",
    "subgraph0_prefill2",
]
for index, call in enumerate(prefill_calls):
    assert prefill_parameters[index].name in call.args
assert prefill_calls[2].args[0] == prefill_calls[1].name
assert prefill_calls[3].args[0] == prefill_calls[2].name
assert str(prefill_combined).count("call @subgraph0_prefill1") == 3

for func_name, template_id, expected in (
    ("forward_prefill", 0, "subgraph0_prefill0"),
    ("forward_decode", 1, "subgraph0_decode1"),
    ("forward_prefill_128", 0, "subgraph0_prefill_128_0"),
    ("forward_decode_128", 2, "subgraph0_decode_128_2"),
):
    symbol_graph, _, _ = five_region_graph(func_name)
    symbol_analysis, symbol_plan = materialization_analysis(symbol_graph)
    symbol_driver = TemplatePartitionedGraphDriver(
        symbol_graph, symbol_analysis.structure_index, symbol_plan
    )
    assert symbol_driver.template_symbol(template_id) == expected


# Tiered graphs retain every Region call while sharing unique template bodies.
tiered_graph, _, _ = five_region_graph("forward_decode_128")
tiered_analysis, tiered_plan = materialization_analysis(tiered_graph)
tiered_driver = TemplatePartitionedGraphDriver(
    tiered_graph, tiered_analysis.structure_index, tiered_plan
)
tiered_subgraphs = tiered_driver.build_template_subgraphs()
assert len(tiered_analysis.structure_index.regions) == 5
assert len(tiered_plan.templates) == 3
assert len(tiered_subgraphs) == 3
for subgraph in tiered_subgraphs:
    subgraph.lower_to_top_level_ir()
tiered_combined = tiered_driver.construct_template_combined_main_graph(True)
tiered_calls = [
    op for op in tiered_driver.combined_graph.body if isinstance(op, CallOp)
]
assert len(tiered_calls) == 5
assert [call.call_func_name for call in tiered_calls] == [
    "subgraph0_decode_128_0",
    "subgraph0_decode_128_1",
    "subgraph0_decode_128_1",
    "subgraph0_decode_128_1",
    "subgraph0_decode_128_2",
]
assert tiered_driver.combined_graph._func_name == "forward_decode_128"
assert "func.func @forward_decode_128" in str(tiered_combined)


# Output remapping changes only the public forward return order.
remap_graph, _, _ = five_region_graph(
    "forward_prefill", output_indices=(0, 2, 4)
)
remap_analysis, remap_plan = materialization_analysis(remap_graph)
ordered_outputs_before = [
    tuple(region.interface.ordered_outputs)
    for region in remap_analysis.structure_index.regions
]
remap_driver = TemplatePartitionedGraphDriver(
    remap_graph, remap_analysis.structure_index, remap_plan
)
remap_driver.build_template_subgraphs()
remap_driver.construct_template_combined_main_graph()
default_calls = [
    op for op in remap_driver.combined_graph.body if isinstance(op, CallOp)
]
default_output = next(
    op for op in remap_driver.combined_graph.body if isinstance(op, OutputOp)
)
assert default_output.args == [
    default_calls[0].name,
    default_calls[2].name,
    default_calls[4].name,
]
remap_driver.construct_template_combined_main_graph(
    output_remap=[2, 0, 1]
)
remapped_calls = [
    op for op in remap_driver.combined_graph.body if isinstance(op, CallOp)
]
remapped_output = next(
    op for op in remap_driver.combined_graph.body if isinstance(op, OutputOp)
)
assert remapped_output.args == [
    remapped_calls[4].name,
    remapped_calls[0].name,
    remapped_calls[2].name,
]
assert ordered_outputs_before == [
    tuple(region.interface.ordered_outputs)
    for region in remap_analysis.structure_index.regions
]
expect_value_error(
    "output_remap length",
    lambda: remap_driver.construct_template_combined_main_graph(
        output_remap=[0]
    ),
)
expect_value_error(
    "invalid output index",
    lambda: remap_driver.construct_template_combined_main_graph(
        output_remap=[0, 1, 3]
    ),
)


# The exporter analyzes and materializes each phase independently, then writes
# one manifest only after both complete.
repo_root = os.environ.get("BUDDY_SRC_ROOT", os.getcwd())
import_model_path = os.path.join(repo_root, "tools", "buddy-codegen", "import_model.py")
import_model_spec = importlib.util.spec_from_file_location(
    "template_materialization_import_model", import_model_path
)
assert import_model_spec is not None and import_model_spec.loader is not None
import_model_module = importlib.util.module_from_spec(import_model_spec)
import_model_spec.loader.exec_module(import_model_module)

export_prefill, _, _ = five_region_graph("forward_prefill")
export_decode, _, _ = five_region_graph("forward_decode")
with tempfile.TemporaryDirectory() as output_dir:
    manifest = import_model_module.export_template_partitioned_mlir(
        export_prefill, export_decode, output_dir
    )
    partition_dir = os.path.join(output_dir, "layer_partitioned")
    filenames = set(os.listdir(partition_dir))
    assert "forward_prefill.mlir" in filenames
    assert "forward_decode.mlir" in filenames
    assert "partition_manifest.json" in filenames
    prefill_files = sorted(
        name
        for name in filenames
        if name.startswith("subgraph0_prefill") and name.endswith(".mlir")
    )
    decode_files = sorted(
        name
        for name in filenames
        if name.startswith("subgraph0_decode") and name.endswith(".mlir")
    )
    assert prefill_files == [
        "subgraph0_prefill0.mlir",
        "subgraph0_prefill1.mlir",
        "subgraph0_prefill2.mlir",
    ]
    assert decode_files == [
        "subgraph0_decode0.mlir",
        "subgraph0_decode1.mlir",
        "subgraph0_decode2.mlir",
    ]
    with open(os.path.join(partition_dir, "partition_manifest.json")) as f:
        written_manifest = json.load(f)
    assert written_manifest == manifest
    assert manifest["prefill_regions"] == 5
    assert manifest["prefill_templates"] == 3
    assert manifest["prefill_subgraphs"] == len(prefill_files)
    assert manifest["decode_regions"] == 5
    assert manifest["decode_templates"] == 3
    assert manifest["decode_subgraphs"] == len(decode_files)
    assert manifest["prefill_main_graphs"] == 0
    assert manifest["decode_main_graphs"] == 0
    assert manifest["debug_wrappers"] is False
    assert manifest["template_materialization"] is True
    assert "prefill_template_materialization" not in manifest
