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

classicfuse_register = {"transpose+mamtmul2D": TransposeMatmulFusedOp}

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def check_classicfusetype(graph: Graph, op: Op):
    pattern = None
    if isinstance(op, MatmulOp):
        parentop = [graph.node_table[str(i)] for i in op._parents]
        for target in parentop:
            if isinstance(target, PermuteOp) and target.args[
                1
            ] == immutable_list([1, 0]):
                pattern = target, parentop, "transpose+mamtmul2D"
    # TODO:other classic fusion pattern
    return pattern


def classic_fuse_check(graph: Graph):
    for op in graph.body:
        pattern = check_classicfusetype(graph, op)
        if pattern:
            do_classicfusion(graph, op, pattern[0], pattern[1], pattern[2])
        else:
            continue


def do_classicfusion(
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
    fusedop = classicfuse_register.get(pattern)()
    # matmulop -> fusedmatmulopnode
    fusedop.name = "fused" + node.name
    graph.displace_node(node, fusedop)
    fusedop.args.pop(fusedop.args.index(target.name))
    fusedop._parents.pop(fusedop._parents.index(target.name))
    fusedop.args.extend(target.args)

    fusedop._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fusedop.name)
    target._children.pop(target._children.index(fusedop.name))

    if graph.check_deletenode(target):
        graph.delete_node(target, targets_parent)


def classic_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.UNKNOW
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
    device = DeviceType.UNKNOW
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}
