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

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = [] 

classicfuse_register = {
    "transpose+mamtmul2D": transpose_Matmul_fusedOp
}

def classic_fuse(graph : Graph):
    for op in graph.body:
        pattern = graph.check_classicfusetype(op)
        if (pattern):
            do_classicfusion(graph,op,pattern[0],pattern[1],pattern[2])
        else:
            continue

def do_classicfusion(graph : Graph,node,target : Op,parents : List[Op],pattern : str):
    """
    Function to fuse some typical operations into one operation.
    Such as transpose + matmul

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """

    fusedop = classicfuse_register.get(pattern)()
    fusedop.name = "fused"+node.name
    graph.displace_node(node,fusedop)
    fusedop.args.pop(fusedop.args.index(target.name))
    fusedop._parents.pop(fusedop._parents.index(target.name))
    fusedop.args.extend(target.args)
    fusedop._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fusedop.name)
    target._children.pop(target._children.index(fusedop.name))
    
    if(graph.check_deletenode(target)):
        graph.delete_node(target,targets_parent)

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
    classic_fuse(graph)
    
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}
