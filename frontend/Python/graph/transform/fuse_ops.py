from .. import Graph
from ..operation import PlaceholderOp
from .. import DeviceType


def simply_fuse(graph: Graph):
    new_op_group = []
    device = DeviceType.UNKNOW
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}
