# RUN: %PYTHON %s

import operator

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

import buddy.compiler.frontend as frontend
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.source_meta import SourceMeta, extract_source_meta


def import_fx(graph_module, *inputs):
    """Run a captured FX graph through the production FX-to-Buddy loop."""

    compiler = DynamoCompiler()
    original_aot_module_simplified = frontend.aot_module_simplified

    def direct_aot_module_simplified(
        gm, example_inputs, fw_compiler, decompositions
    ):
        return fw_compiler(gm, example_inputs)

    frontend.aot_module_simplified = direct_aot_module_simplified
    try:
        compiler._compile_fx(graph_module, list(inputs))
    finally:
        frontend.aot_module_simplified = original_aot_module_simplified
    assert len(compiler._imported_graphs) == 1
    return compiler._imported_graphs[0]


def capture(module, *inputs):
    graph_module = symbolic_trace(module)
    output = next(
        node for node in graph_module.graph.nodes if node.op == "output"
    )
    if isinstance(output.args[0], torch.fx.Node):
        output.args = ((output.args[0],),)
        graph_module.recompile()
    ShapeProp(graph_module).propagate(*inputs)
    return graph_module


class AddLeaf(torch.nn.Module):
    def forward(self, lhs, rhs):
        return lhs + rhs


class AddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = AddLeaf()

    def forward(self, lhs, rhs):
        return self.block(lhs, rhs)


lhs = torch.ones(2)
rhs = torch.ones(2)
add_gm = capture(AddModel(), lhs, rhs)
add_fx = next(
    node
    for node in add_gm.graph.nodes
    if node.op == "call_function" and node.target is operator.add
)
add_fx.meta["original_aten"] = "aten.add.Tensor"
add_source = extract_source_meta(add_fx)
assert add_source == (
    SourceMeta("block", f"{__name__}.AddLeaf", "aten.add.Tensor"),
)
add_graph = import_fx(add_gm, lhs, rhs)
assert add_graph.node_table[add_fx.name]._source_meta == add_source


class BareAdd(torch.nn.Module):
    def forward(self, lhs, rhs):
        return lhs + rhs


empty_gm = capture(BareAdd(), lhs, rhs)
empty_fx = next(
    node
    for node in empty_gm.graph.nodes
    if node.op == "call_function" and node.target is operator.add
)
empty_fx.meta.pop("nn_module_stack", None)
empty_fx.meta.pop("original_aten", None)
assert extract_source_meta(empty_fx) == ()
empty_graph = import_fx(empty_gm, lhs, rhs)
assert empty_graph.node_table[empty_fx.name]._source_meta == ()


class GetItemLeaf(torch.nn.Module):
    def forward(self, value):
        return value[0]


class GetItemModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = GetItemLeaf()

    def forward(self, value):
        return self.block(value)


value = torch.ones(2, 3)
getitem_gm = capture(GetItemModel(), value)
getitem_fx = next(
    node
    for node in getitem_gm.graph.nodes
    if node.op == "call_function" and node.target is operator.getitem
)
getitem_source = extract_source_meta(getitem_fx)
assert getitem_source == (
    SourceMeta("block", f"{__name__}.GetItemLeaf", None),
)
getitem_graph = import_fx(getitem_gm, value)
assert getitem_graph.node_table[getitem_fx.name]._source_meta == getitem_source


# _create_node lowers nested FX operands and dtypes while parents come from args.
operand_fx_graph = torch.fx.Graph()
operand_lhs = operand_fx_graph.placeholder("operand_lhs")
operand_rhs = operand_fx_graph.placeholder("operand_rhs")
operand_kwarg = operand_fx_graph.placeholder("operand_kwarg")
operand_args = (
    [operand_lhs, (operand_rhs, {"value": operand_lhs})],
    operand_lhs,
    torch.float32,
)
operand_kwargs = {
    "nested": {"value": operand_kwarg, "items": (operand_rhs, [operand_lhs])},
    "dtype": torch.float64,
}
operand_compiler = DynamoCompiler()
operand_node = operand_compiler._create_node(
    "add.Tensor", "nested_operands", operand_args, [], node_kwargs=operand_kwargs
)
assert operand_node.args == [
    ["operand_lhs", ("operand_rhs", {"value": "operand_lhs"})],
    "operand_lhs",
    operand_compiler._torch_dtype_translate(str(torch.float32)),
]
assert operand_node.kwargs == {
    "nested": {
        "value": "operand_kwarg",
        "items": ("operand_rhs", ["operand_lhs"]),
    },
    "dtype": operand_compiler._torch_dtype_translate(str(torch.float64)),
}
assert operand_node.parents == [
    "operand_lhs",
    "operand_rhs",
    "operand_lhs",
    "operand_lhs",
]
