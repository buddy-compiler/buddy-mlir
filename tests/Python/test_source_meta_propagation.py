# RUN: %PYTHON %s

from torch.fx.immutable_collections import immutable_list

from buddy.compiler.frontend import _is_unsupported_get_attr
from buddy.compiler.graph.graph import Graph
from buddy.compiler.graph.operation import (
    AddMMOp,
    AddOp,
    CloneOp,
    ExpandOp,
    GQAAttentionFusedOp,
    IndexPutOp,
    MatmulOp,
    Op,
    PermuteOp,
    ReshapeOp,
    TransposeMatmulFusedOp,
    TransposeOp,
    UnsqueezeOp,
    ViewOp,
    ScaledDotProductFlashAttentionForCpuOp,
)
from buddy.compiler.graph.source_meta import SourceMeta, merge_source_meta
from buddy.compiler.graph.transform.eliminate_matmul_transpose_reshape import (
    eliminate_matmul_transpose_reshape,
)
from buddy.compiler.graph.transform.fuse_ops import (
    decompose_addmm_to_mm_add,
    replace_gqa_attention_with_fused_op,
    transpose_matmul_fusion,
)
from buddy.compiler.graph.type import TensorDType


def meta(name):
    return (SourceMeta(module_path=name),)


def node(cls, name, args=None, parents=None, children=None):
    value = cls()
    value.name = name
    value._arguments = list(args or [])
    value._parents = list(parents or [])
    value._children = list(children or [])
    return value


def graph_of(*nodes):
    graph = Graph.__new__(Graph)
    graph._body = list(nodes)
    graph.node_table = {value.name: value for value in nodes}
    graph._inputs = []
    graph._fake_params = []
    return graph


class FakeFXNode:
    def __init__(self, op, name):
        self.op = op
        self.name = name


# The production loop uses this predicate before SourceMeta extraction or
# graph.add_node. Exercise first/middle get_attr placement and later nodes.
fx_nodes = (
    FakeFXNode("get_attr", "ordinary_first"),
    FakeFXNode("placeholder", "input"),
    FakeFXNode("get_attr", "ordinary_middle"),
    FakeFXNode("get_attr", "_tensor_constant0"),
    FakeFXNode("call_function", "later_op"),
)
created = [value.name for value in fx_nodes if not _is_unsupported_get_attr(value)]
assert created == ["input", "_tensor_constant0", "later_op"]


a, b = SourceMeta(module_path="a"), SourceMeta(module_path="b")
assert merge_source_meta((a, b), (a,), (), (b,)) == (a, b)

old = node(Op, "old")
old._source_meta = meta("old")
new = node(Op, "new")
graph = graph_of(old)
graph.displace_node(old, new)
assert new._source_meta == meta("old")

bias = node(Op, "bias", children=["addmm"])
lhs = node(Op, "lhs", children=["addmm"])
rhs_parent = node(Op, "weight", children=["rhs"])
rhs = node(
    PermuteOp,
    "rhs",
    ["weight", immutable_list([1, 0])],
    ["weight"],
    ["addmm"],
)
addmm = node(AddMMOp, "addmm", ["bias", "lhs", "rhs"], ["bias", "lhs", "rhs"])
addmm._source_meta = meta("addmm")
graph = graph_of(bias, lhs, rhs_parent, rhs, addmm)
decompose_addmm_to_mm_add(graph)
assert isinstance(graph.node_table["addmm"], AddOp)
assert graph.node_table["addmm"]._source_meta == meta("addmm")
assert graph.node_table["addmm_decomposed_mm"]._source_meta == meta("addmm")

left = node(Op, "left", children=["mm"])
weight = node(Op, "weight", children=["permute"])
permute = node(
    PermuteOp,
    "permute",
    ["weight", immutable_list([1, 0])],
    ["weight"],
    ["mm"],
)
mm = node(MatmulOp, "mm", ["left", "permute"], ["left", "permute"])
permute._source_meta = meta("permute")
mm._source_meta = meta("mm")
graph = graph_of(left, weight, permute, mm)
transpose_matmul_fusion(graph, mm, permute, [left, permute], "transpose_matmul_fusion")
fused = graph.node_table["fusedmm"]
assert isinstance(fused, TransposeMatmulFusedOp)
assert fused._source_meta == meta("permute") + meta("mm")

# A shared permute survives fusion and must not be reported as absorbed.
left = node(Op, "left2", children=["mm2"])
weight = node(Op, "weight2", children=["permute2"])
extra = node(Op, "extra", ["permute2"], ["permute2"])
permute = node(
    PermuteOp,
    "permute2",
    ["weight2", immutable_list([1, 0])],
    ["weight2"],
    ["mm2", "extra"],
)
mm = node(MatmulOp, "mm2", ["left2", "permute2"], ["left2", "permute2"])
permute._source_meta = meta("permute2")
mm._source_meta = meta("mm2")
graph = graph_of(left, weight, permute, mm, extra)
transpose_matmul_fusion(graph, mm, permute, [left, permute], "transpose_matmul_fusion")
fused = graph.node_table["fusedmm2"]
assert permute in graph.body
assert fused._source_meta == meta("mm2")

query = node(Op, "query", children=["sdpa"])
k_index = node(IndexPutOp, "k_index", children=["k_unsqueeze"])
v_index = node(IndexPutOp, "v_index", children=["v_unsqueeze"])
k_unsqueeze = node(UnsqueezeOp, "k_unsqueeze", ["k_index"], ["k_index"], ["k_expand"])
k_expand = node(ExpandOp, "k_expand", ["k_unsqueeze"], ["k_unsqueeze"], ["k_clone"])
k_clone = node(CloneOp, "k_clone", ["k_expand"], ["k_expand"], ["k_view"])
k_view = node(ViewOp, "k_view", ["k_clone"], ["k_clone"], ["sdpa"])
v_unsqueeze = node(UnsqueezeOp, "v_unsqueeze", ["v_index"], ["v_index"], ["v_expand"])
v_expand = node(ExpandOp, "v_expand", ["v_unsqueeze"], ["v_unsqueeze"], ["v_clone"])
v_clone = node(CloneOp, "v_clone", ["v_expand"], ["v_expand"], ["v_view"])
v_view = node(ViewOp, "v_view", ["v_clone"], ["v_clone"], ["sdpa"])
sdpa = node(
    ScaledDotProductFlashAttentionForCpuOp,
    "sdpa",
    ["query", "k_view", "v_view"],
    ["query", "k_view", "v_view"],
)
for value in (
    k_index,
    k_unsqueeze,
    k_expand,
    k_clone,
    k_view,
    v_index,
    v_unsqueeze,
    v_expand,
    v_clone,
    v_view,
    sdpa,
):
    value._source_meta = meta(value.name)
graph = graph_of(
    query,
    k_index,
    k_unsqueeze,
    k_expand,
    k_clone,
    k_view,
    v_index,
    v_unsqueeze,
    v_expand,
    v_clone,
    v_view,
    sdpa,
)
replace_gqa_attention_with_fused_op(
    graph,
    sdpa,
    k_view,
    k_clone,
    k_expand,
    k_unsqueeze,
    k_index,
    v_view,
    v_clone,
    v_expand,
    v_unsqueeze,
    v_index,
    "gqa_attention_fusion",
)
gqa = graph.node_table["GQAAttentionFusedOp_1"]
assert isinstance(gqa, GQAAttentionFusedOp)
assert k_index in graph.body and v_index in graph.body
assert meta("k_index")[0] not in gqa._source_meta
assert meta("v_index")[0] not in gqa._source_meta
for removed in (k_unsqueeze, k_expand, k_clone, k_view, v_unsqueeze, v_expand, v_clone, v_view):
    assert removed not in graph.body
    assert meta(removed.name)[0] in gqa._source_meta

# Sharing an intermediate keeps that node and every upstream branch node alive;
# only the actually deleted downstream nodes contribute metadata.
query2 = node(Op, "query2", children=["sdpa2"])
k_index2 = node(IndexPutOp, "k_index2", children=["k_unsqueeze2"])
k_unsqueeze2 = node(UnsqueezeOp, "k_unsqueeze2", ["k_index2"], ["k_index2"], ["k_expand2"])
k_expand2 = node(ExpandOp, "k_expand2", ["k_unsqueeze2"], ["k_unsqueeze2"], ["k_clone2", "extra2"])
k_clone2 = node(CloneOp, "k_clone2", ["k_expand2"], ["k_expand2"], ["k_view2"])
k_view2 = node(ViewOp, "k_view2", ["k_clone2"], ["k_clone2"], ["sdpa2"])
extra2 = node(Op, "extra2", ["k_expand2"], ["k_expand2"])
v_index2 = node(IndexPutOp, "v_index2", children=["v_unsqueeze2"])
v_unsqueeze2 = node(UnsqueezeOp, "v_unsqueeze2", ["v_index2"], ["v_index2"], ["v_expand2"])
v_expand2 = node(ExpandOp, "v_expand2", ["v_unsqueeze2"], ["v_unsqueeze2"], ["v_clone2"])
v_clone2 = node(CloneOp, "v_clone2", ["v_expand2"], ["v_expand2"], ["v_view2"])
v_view2 = node(ViewOp, "v_view2", ["v_clone2"], ["v_clone2"], ["sdpa2"])
sdpa2 = node(
    ScaledDotProductFlashAttentionForCpuOp,
    "sdpa2",
    ["query2", "k_view2", "v_view2"],
    ["query2", "k_view2", "v_view2"],
)
branch2 = (
    k_index2, k_unsqueeze2, k_expand2, k_clone2, k_view2,
    v_index2, v_unsqueeze2, v_expand2, v_clone2, v_view2, sdpa2,
)
for value in branch2:
    value._source_meta = meta(value.name)
graph = graph_of(query2, *branch2[:-1], extra2, sdpa2)
replace_gqa_attention_with_fused_op(
    graph, sdpa2, k_view2, k_clone2, k_expand2, k_unsqueeze2, k_index2,
    v_view2, v_clone2, v_expand2, v_unsqueeze2, v_index2,
    "gqa_attention_fusion",
)
gqa = graph.node_table["GQAAttentionFusedOp_1"]
assert k_expand2 in graph.body and k_unsqueeze2 in graph.body and k_index2 in graph.body
assert k_view2 not in graph.body and k_clone2 not in graph.body
assert meta("k_view2")[0] in gqa._source_meta
assert meta("k_clone2")[0] in gqa._source_meta
for survivor in (k_expand2, k_unsqueeze2, k_index2):
    assert meta(survivor.name)[0] not in gqa._source_meta

input_node = node(Op, "input", children=["transpose"])
input_node.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
transpose = node(
    TransposeOp,
    "transpose",
    ["input", [0, 1]],
    ["input"],
    ["reshape"],
)
transpose.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
transpose._source_meta = meta("transpose")
reshape = node(
    ReshapeOp,
    "reshape",
    ["transpose", [2, 1]],
    ["transpose"],
)
reshape.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
reshape._source_meta = meta("reshape")
graph = graph_of(input_node, transpose, reshape)
eliminate_matmul_transpose_reshape(graph)
assert transpose not in graph.body
assert reshape._source_meta == meta("transpose") + meta("reshape")

# Multiple consumers replace the permute itself with one Reshape. Only the
# replaced permute is absorbed; its surviving consumers are not.
input_node = node(Op, "multi_input", children=["multi_permute"])
input_node.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
permute = node(
    PermuteOp, "multi_permute", ["multi_input", [0, 1]], ["multi_input"],
    ["multi_a", "multi_b"],
)
permute.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
permute._source_meta = meta("multi_permute")
multi_a = node(Op, "multi_a", ["multi_permute"], ["multi_permute"])
multi_b = node(Op, "multi_b", ["multi_permute"], ["multi_permute"])
multi_a._source_meta = meta("multi_a")
multi_b._source_meta = meta("multi_b")
graph = graph_of(input_node, permute, multi_a, multi_b)
eliminate_matmul_transpose_reshape(graph)
replacement = graph.node_table["multi_permute_reshaped"]
assert permute not in graph.body and isinstance(replacement, ReshapeOp)
assert replacement._source_meta == meta("multi_permute")
assert meta("multi_a")[0] not in replacement._source_meta
assert multi_a.args[0] == replacement.name and multi_b.args[0] == replacement.name

# An Unsqueeze replacement merges the transpose and replaced Unsqueeze in
# original body order.
input_node = node(Op, "unsq_input", children=["unsq_transpose"])
input_node.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
transpose = node(
    TransposeOp, "unsq_transpose", ["unsq_input", [0, 1]], ["unsq_input"],
    ["unsqueeze"],
)
transpose.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
transpose._source_meta = meta("unsq_transpose")
unsqueeze = node(UnsqueezeOp, "unsqueeze", ["unsq_transpose", 0], ["unsq_transpose"])
unsqueeze.tensor_meta = {"shape": (1, 2, 1), "dtype": TensorDType.Float32}
unsqueeze._source_meta = meta("unsqueeze")
graph = graph_of(input_node, transpose, unsqueeze)
eliminate_matmul_transpose_reshape(graph)
replacement = graph.node_table["unsqueeze"]
assert isinstance(replacement, ReshapeOp) and transpose not in graph.body
assert replacement._source_meta == meta("unsq_transpose") + meta("unsqueeze")

# A no-op view skipped before transpose is truly deleted and joins the
# surviving reshape metadata according to the original graph.body order.
input_node = node(Op, "skip_input", children=["skip_view"])
input_node.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
view = node(ViewOp, "skip_view", ["skip_input", [2, 1]], ["skip_input"], ["skip_transpose"])
view.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
view._source_meta = meta("skip_view")
transpose = node(
    TransposeOp, "skip_transpose", ["skip_view", [0, 1]], ["skip_view"],
    ["skip_reshape"],
)
transpose.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
transpose._source_meta = meta("skip_transpose")
reshape = node(ReshapeOp, "skip_reshape", ["skip_transpose", [2, 1]], ["skip_transpose"])
reshape.tensor_meta = {"shape": (2, 1), "dtype": TensorDType.Float32}
reshape._source_meta = meta("skip_reshape")
graph = graph_of(input_node, view, transpose, reshape)
eliminate_matmul_transpose_reshape(graph)
assert view not in graph.body and transpose not in graph.body
assert reshape._source_meta == meta("skip_view") + meta("skip_transpose") + meta("skip_reshape")
assert reshape.args[0] == "skip_input"
