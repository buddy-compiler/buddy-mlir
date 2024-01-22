from .. import Graph
from .. import DeviceType


def simply_fuse(graph: Graph):
    new_op_group = {}
    device = DeviceType.UNKNOW
    for subgraph_name in graph.op_groups.keys():
        if (
            device != DeviceType.UNKNOW
            and graph.group_map_device[subgraph_name] != device
            and graph.group_map_device[subgraph_name] != DeviceType.UNKNOW
        ):
            raise RuntimeError(
                "can't fuse ops from {} and {}".format(
                    device.value, graph.group_map_device[subgraph_name].value
                )
            )
        for op in graph.op_groups[subgraph_name]:
            new_op_group.append(op)
        if (
            device == DeviceType.UNKNOW
            and graph.group_map_device[subgraph_name] != device
        ):
            device = graph.group_map_device[subgraph_name]
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}
