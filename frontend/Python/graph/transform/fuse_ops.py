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
    "residual_fusion": MatmulWithAccOp,
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


def residual_fuse_check(graph: Graph):
    for op in graph.body:
        pattern = None
        if isinstance(op, MatmulOp):
            child_1op = [graph.node_table[str(i)] for i in op._children]
            for reshape in child_1op:
                if isinstance(reshape, (ViewOp, ReshapeOp)) and (
                    reshape.args[1] == immutable_list([1, 1, 1536])
                ):
                    child_2op = [
                        graph.node_table[str(i)] for i in reshape._children
                    ]
                    for add in child_2op:
                        if isinstance(add, AddOp):
                            pattern = (reshape, add, "residual_fusion")
                            break
                    else:
                        continue
                    break
        if pattern:
            residual_fusion(
                graph,
                op,
                pattern[0],
                pattern[1],
                pattern[2],
            )


def residual_fusion(graph: Graph, node, reshape, add: Op, pattern: str):
    fuse_op = classicfuse_register.get(pattern)()
    fuse_op.name = "fused_" + node.name
    graph.displace_node(node, fuse_op)

    reshape._children.extend(add._children)
    add_children = [
        graph.node_table[child_name] for child_name in add._children
    ]
    for child in add_children:
        if add.name in child._parents:
            parent_idx = child._parents.index(add.name)
            child._parents[parent_idx] = reshape.name

            if add.name in child.args:
                arg_idx = child.args.index(add.name)
                child.args[arg_idx] = reshape.name

    if add.name in reshape._children:
        reshape._children.pop(reshape._children.index(add.name))

    residual_parents = [p for p in add._parents if p != reshape.name]
    fuse_op._parents.extend(residual_parents)
    fuse_op.args.extend(residual_parents)

    fuse_op._parents = list(dict.fromkeys(fuse_op._parents))
    original_args = node.args.copy()
    residual_only = [p for p in fuse_op._parents if p not in original_args]
    fuse_op._parents = original_args + residual_only

    add._children.clear()

    add_parents = []
    for parent_name in add._parents:
        parent = graph.node_table[parent_name]
        if add.name in parent._children:
            add_parents.append(parent)

    if graph.check_delete_node(add) and add_parents:
        graph.delete_node(add, add_parents)


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
    residual_fuse_check(graph)
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
