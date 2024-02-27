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

from typing import List
from .. import Graph
from ..operation import Op, PlaceholderOp, OpType
from .. import DeviceType

# TODO: classify op type for op fusion
OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType, OpType.TransposeType, OpType.GetItemType]
OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.PlaceholderType]
OP_TYPE_ELEMENTWISE_FUSABLE = [OpType.SliceLikeType]
ANCHOR_OP_TYPE = [OpType.ReduceType, OpType.SliceLikeType]

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

def has_anchor_op(op_group: List[Op]):
    for op in op_group:
        if op.op_type in ANCHOR_OP_TYPE:
            return True
    return False

def has_unfusable_op(op_group: List[Op]):
    for op in op_group:
        if op.op_type in OP_TYPE_UNFUSABLE:
            return True
    return False

def has_elementwise_fusable_op(op_group: List[Op]):
    for op in op_group:
        if op.op_type in OP_TYPE_ELEMENTWISE_FUSABLE:
            return True
    return False

def is_all_elementwise_op(op_group: List[Op]):
    for op in op_group:
        if op.op_type != OpType.BroadcastType or op.op_type != OpType.ElementwiseType:
            return False
    return True

def get_successor_list(op_group: List[Op], graph: Graph):
    elementwise_fusable_group = has_elementwise_fusable_op(op_group)
    successor_list = []
    for op in op_group:
        for child in op.children:
            child_node = graph.node_table[child]
            if child_node in op_group:
                continue
            child_group_name = graph.op_map_group[child]
            child_group = graph.op_groups[child_group_name]
            if has_anchor_op(child_group) or has_unfusable_op(child_group):
                continue
            if elementwise_fusable_group and is_all_elementwise_op(child_group):
                continue
            successor_list.append(child_group_name)
    return successor_list

def get_precursor_list(op_group: List[Op], graph: Graph):
    elementwise_fusable_group = has_elementwise_fusable_op(op_group)
    precursor_list = []
    for op in op_group:
        for parent in op.parents:
            parent_node = graph.node_table[parent]
            if isinstance(parent_node, PlaceholderOp):
                continue
            if parent_node in op_group:
                continue
            parent_group_name = graph.op_map_group[parent]
            parent_group = graph.op_groups[parent_group_name]
            if has_anchor_op(parent_group) or has_unfusable_op(parent_group):
                continue
            if elementwise_fusable_group and is_all_elementwise_op(parent_group):
                continue
            precursor_list.append(parent_group_name)
    return precursor_list

def forward_fuse(graph: Graph):
    for node in graph.body:
        print(node.__dict__)
    print("-------------------")
    while True:
        print("aaaaaaaaaaaaaaaaaaaaaaaaa")
        start = len(graph.op_groups)
        subgraph_names = list(graph.op_groups.keys())
        for subgraph_name in subgraph_names:
            if not has_anchor_op(graph.op_groups[subgraph_name]):
                continue
            op_group = graph.op_groups[subgraph_name]
            successor_list = get_successor_list(op_group, graph)
            for group_name in successor_list:
                subgraph_names.remove(group_name)
                successor_group = graph.op_groups[group_name]
                for op in successor_group:
                    op_group.append(op)
                    graph.op_map_group[op.name] = subgraph_name
                del graph.op_groups[group_name]
        if start == len(graph.op_groups):
            break
    print("-----------------------")
    for key in graph.op_groups:
        print(key)
        for op in graph.op_groups[key]:
            print(op.__dict__)

def backward_fuse(graph: Graph):
    while True:
        print("bbbbbbbbbbbbbbbbbbbbbbbbbb")
        start = len(graph.op_groups)
        subgraph_names = list(graph.op_groups.keys())
        for subgraph_name in subgraph_names:
            if not has_anchor_op(graph.op_groups[subgraph_name]):
                continue
            op_group = graph.op_groups[subgraph_name]
            precursor_list = get_precursor_list(op_group, graph)
            for group_name in precursor_list:
                subgraph_names.remove(group_name)
                precursor_group = graph.op_groups[group_name]
                for op in precursor_group:
                    op_group.insert(0, op)
                    graph.op_map_group[op.name] = subgraph_name
                del graph.op_groups[group_name]
        if start == len(graph.op_groups):
            break
    print("-----------------------")
    for key in graph.op_groups:
        print(key)
        for op in graph.op_groups[key]:
            print(op.__dict__)

