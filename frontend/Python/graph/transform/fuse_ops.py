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

classicfuse_register = {"transpose_matmul_fusion": TransposeMatmulFusedOp,
                         "norm_fusion": NormOp
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
        # === LayerNorm pattern ===
        if isinstance(op, PowOp):
            # check LayerNorm pattern: pow -> mean -> add -> rsqrt -> mul -> mul
            if not op._children:
                continue
            mean_node = graph.node_table.get(op._children[0], None)
            if not isinstance(mean_node, MeanOp):
                continue

            if not mean_node._children:
                continue
            add_node = graph.node_table.get(mean_node._children[0], None)
            if not isinstance(add_node, AddOp):
                continue

            if not add_node._children:
                continue
            rsqrt_node = graph.node_table.get(add_node._children[0], None)
            if not isinstance(rsqrt_node, RsqrtOp):
                continue

            if not rsqrt_node._children:
                continue
            mul_node = graph.node_table.get(rsqrt_node._children[0], None)
            if not isinstance(mul_node, MulOp):
                continue

            if not mul_node._children:
                continue
            mul_2_node = graph.node_table.get(mul_node._children[0], None)
            if not isinstance(mul_2_node, MulOp):
                continue
            norm_fusion(graph, op, mean_node, add_node, rsqrt_node, mul_node, mul_2_node, "norm_fusion")


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


def norm_fusion(
    graph: Graph,
    pow_node: Op,
    mean_node: Op,
    add_node: Op,
    rsqrt_node: Op,
    mul_node: Op,      
    mul_2_node: Op,    
    pattern: str,
):
    """
    Fuse LayerNorm subgraph (Pow -> Mean -> Add -> Rsqrt -> Mul -> Mul)
    into one LayerNormFusedOp.
    """
    fused_cls = classicfuse_register.get(pattern)

    fused_op = fused_cls()

    fused_op.name = "NormOp"

    graph.displace_node(mul_2_node, fused_op)

    fused_op.args.pop(fused_op.args.index(mul_node.name))

    fused_op._parents.pop(fused_op._parents.index(mul_node.name))
    for parent_name in pow_node._parents:
        fused_op._parents.append(parent_name)
        fused_op.args.append(parent_name)

    mul_node._children.clear()

    if graph.check_delete_node(mul_node):
        graph.delete_node(mul_node,[graph.node_table.get(mul_node._parents[0], None),graph.node_table.get(mul_node._parents[1], None)])

    if graph.check_delete_node(rsqrt_node):
        graph.delete_node(rsqrt_node,[add_node])

    if graph.check_delete_node(add_node):
        graph.delete_node(add_node,[mean_node])

    if graph.check_delete_node(mean_node):
        graph.delete_node(mean_node,[pow_node])

    if graph.check_delete_node(pow_node):
        graph.delete_node(pow_node,[graph.node_table.get(mul_node._parents[0], None)])


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
