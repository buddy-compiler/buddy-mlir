# ===- fuse_ops.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list

classicfuse_register = {
    "transpose_matmul_fusion": TransposeMatmulFusedOp,
    "flash_attention_prefill_fusion": FlashAttentionForCpuPrefillOp,
    "gqa_attention_fusion": GQAAttentionFusedOp,
}

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def classic_fuse_check(graph: Graph):
    """
    Function to identifies and fuses PermuteOp operations with preceding
    MatmulOp operations in a computation graph to optimize performance.

    Args:
        graph (Graph): The computation graph to analyze and optimize.

    Returns:
        None
    """
    for op in graph.body:
        pattern = None
        if isinstance(op, MatmulOp):
            parentop = [graph.node_table[str(i)] for i in op._parents]
            for target in parentop:
                if isinstance(target, PermuteOp) and target.args[
                    1
                ] == immutable_list([1, 0]):
                    pattern = target, parentop, "transpose_matmul_fusion"
        if pattern:
            transpose_matmul_fusion(
                graph, op, pattern[0], pattern[1], pattern[2]
            )


def transpose_matmul_fusion(
    graph: Graph, node, target: Op, parents: List[Op], pattern: str
):
    """
    Function to fuse some typical operations into one operation.
    Such as transpose + matmul
    Args:
    - graph (Graph): The input graph to be simplified.
    - node (Op): The operation to be fused.
    - target (Op): The target operation to be fused.
    - parents (List[Op]): The parents of the node to be fused.
    - pattern (str): The pattern of the fusion.
    Returns:
    - None: Modifies the input graph in place.
    """
    fused_op = classicfuse_register.get(pattern)()
    # matmulop -> fusedmatmulopnode
    fused_op.name = "fused" + node.name
    graph.displace_node(node, fused_op)
    fused_op.args.pop(fused_op.args.index(target.name))
    fused_op._parents.pop(fused_op._parents.index(target.name))
    fused_op.args.extend(target.args)

    fused_op._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fused_op.name)
    target._children.pop(target._children.index(fused_op.name))

    if graph.check_delete_node(target):
        graph.delete_node(target, targets_parent)


def apply_classic_fusion(graph: Graph):
    """
    Function to fuse some typical operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    # Run the first round of op fusion
    classic_fuse_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def flash_attention_prefill(graph: Graph):
    """
    Replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    new_op_group = []
    device = DeviceType.CPU
    replace_attention_op(graph)

    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)

    graph.op_groups = {"subgraph0": new_op_group}
    graph.group_map_device = {"subgraph0": device}


def replace_attention_op(graph: Graph):
    """
    replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    for op in list(graph.body):
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):
            new_op = classicfuse_register.get(
                "flash_attention_prefill_fusion"
            )()
            new_op.name = "FlashAttentionForCpuPrefillOp"
            graph.displace_node(op, new_op)


def gqa_attention_fusion(graph: Graph):
    """
    Function to fuse GQA Attention operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    gqa_attention_fusion_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def gqa_attention_fusion_check(graph: Graph):
    for op in graph.body:
        # === GQA Attention pattern ===
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):

            # get KV and View nodes
            k_view_node = graph.node_table.get(op._parents[1], None)
            v_view_node = graph.node_table.get(op._parents[2], None)

            if not (
                isinstance(k_view_node, ViewOp)
                and isinstance(v_view_node, ViewOp)
            ):
                continue

            # trace Key branch: View <- Clone <- Expand <- slice1 <- slice2 <- unsqueeze
            k_clone = graph.node_table.get(k_view_node._parents[0], None)
            if not isinstance(k_clone, CloneOp):
                continue
            k_expand = graph.node_table.get(k_clone._parents[0], None)
            if not isinstance(k_expand, ExpandOp):
                continue
            k_slice1 = graph.node_table.get(k_expand._parents[0], None)
            if not isinstance(k_slice1, SliceOp):
                continue
            k_slice2 = graph.node_table.get(k_slice1._parents[0], None)
            if not isinstance(k_slice2, SliceOp):
                continue
            k_cache_unsqueeze = graph.node_table.get(k_slice2._parents[0], None)
            if not isinstance(k_cache_unsqueeze, UnsqueezeOp):
                continue

            # trace Value branch: View <- Clone <- Expand <- slice1 <- slice2 <- unsqueeze
            v_clone = graph.node_table.get(v_view_node._parents[0], None)
            if not isinstance(v_clone, CloneOp):
                continue
            v_expand = graph.node_table.get(v_clone._parents[0], None)
            if not isinstance(v_expand, ExpandOp):
                continue
            v_slice1 = graph.node_table.get(v_expand._parents[0], None)
            if not isinstance(v_slice1, SliceOp):
                continue
            v_slice2 = graph.node_table.get(v_slice1._parents[0], None)
            if not isinstance(v_slice2, SliceOp):
                continue
            v_cache_unsqueeze = graph.node_table.get(v_slice2._parents[0], None)
            if not isinstance(v_cache_unsqueeze, UnsqueezeOp):
                continue
            replace_gqa_attention_with_fused_op(
                graph,
                op,
                k_view_node,
                k_clone,
                k_expand,
                k_slice1,
                k_slice2,
                k_cache_unsqueeze,
                v_view_node,
                v_clone,
                v_expand,
                v_slice1,
                v_slice2,
                v_cache_unsqueeze,
                "gqa_attention_fusion",
            )


def replace_gqa_attention_with_fused_op(
    graph: Graph,
    sdpa_node: Op,
    k_view: Op,
    k_clone: Op,
    k_expand: Op,
    k_slice1: Op,
    k_slice2: Op,
    k_cache_unsqueeze: Op,
    v_view: Op,
    v_clone: Op,
    v_expand: Op,
    v_slice1: Op,
    v_slice2: Op,
    v_cache_unsqueeze: Op,
    pattern: str,
):
    """
    Fuse GQA subgraph (Reshape -> Expand -> Clone -> Reshape)
    into one GQAAttentionFusedOp.
    """
    fused_cls = classicfuse_register.get(pattern)
    fused_op = fused_cls()
    fused_op.name = "GQAAttentionFusedOp"

    # replace SDPA node with GQAAttentionFusedOp
    graph.displace_node(sdpa_node, fused_op)

    # clear old KV View input inherited by SDPA
    # assume sdpa_node.args[0] is Query, keep unchanged
    # args[1] and args[2] are k_view and v_view, need to pop
    fused_op.args.pop(fused_op.args.index(k_view.name))
    fused_op._parents.pop(fused_op._parents.index(k_view.name))
    fused_op.args.pop(fused_op.args.index(v_view.name))
    fused_op._parents.pop(fused_op._parents.index(v_view.name))

    # 将输入重定向到 KV 分支的最顶端（Cache 的原始输入）
    for k_parent in k_cache_unsqueeze._parents:
        fused_op._parents.append(k_parent)
        fused_op.args.append(k_parent)
    for v_parent in v_cache_unsqueeze._parents:
        fused_op._parents.append(v_parent)
        fused_op.args.append(v_parent)

    k_view._children.clear()
    if graph.check_delete_node(k_view):
        graph.delete_node(k_view, [k_clone])
    if graph.check_delete_node(k_clone):
        graph.delete_node(k_clone, [k_expand])
    if graph.check_delete_node(k_expand):
        graph.delete_node(k_expand, [k_slice1])
    if graph.check_delete_node(k_slice1):
        graph.delete_node(k_slice1, [k_slice2])
    if graph.check_delete_node(k_slice2):
        graph.delete_node(k_slice2, [k_cache_unsqueeze])
    if graph.check_delete_node(k_cache_unsqueeze):
        k_orig_parents = [
            graph.node_table.get(p, None) for p in k_cache_unsqueeze._parents
        ]
        graph.delete_node(k_cache_unsqueeze, k_orig_parents)

    v_view._children.clear()
    if graph.check_delete_node(v_view):
        graph.delete_node(v_view, [v_clone])
    if graph.check_delete_node(v_clone):
        graph.delete_node(v_clone, [v_expand])
    if graph.check_delete_node(v_expand):
        graph.delete_node(v_expand, [v_slice1])
    if graph.check_delete_node(v_slice1):
        graph.delete_node(v_slice1, [v_slice2])
    if graph.check_delete_node(v_slice2):
        graph.delete_node(v_slice2, [v_cache_unsqueeze])
    if graph.check_delete_node(v_cache_unsqueeze):
        v_orig_parents = [
            graph.node_table.get(p, None) for p in v_cache_unsqueeze._parents
        ]
        graph.delete_node(v_cache_unsqueeze, v_orig_parents)
