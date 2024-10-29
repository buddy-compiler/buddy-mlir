import json
from pathlib import Path

from .graph import Graph, TensorDType, TensorMeta
from .graph_driver import GraphDriver
from .operation import *
from .type import *

from ..ops.linalg import ops_registry as linalg_ops_registry
from ..ops.tosa import ops_registry as tosa_ops_registry
from ..ops.math import ops_registry as math_ops_registry
from ..ops.func import ops_registry as func_ops_registry

def json_to_graph(json_str):
    """
    Converts a buddy graph JSON string to a Graph object.

    Args:
        json_str (str): The JSON string representing the buddy graph.

    Returns:
        Graph: The Graph object created from the JSON data.
    """
    def json_to_tensormeta(json_data):
        """
        Convert JSON data to a TensorMeta object.

        Args:
            json_data (dict): JSON data representing a TensorMeta object.

        Returns:
            TensorMeta: The TensorMeta object created from the JSON data.
        """
        if 'shape' in json_data:
            shape = json_data['shape']
            dtype = next(
                (member for member in TensorDType.__members__.values() 
                 if member.value.upper() == json_data['dtype'].upper()), None
            )
            return TensorMeta(shape, dtype)
        return {}
        
    json_data = json.loads(json_str)
    _graph = json_data
    graph_name = _graph['graph_name'] 
    inputs = []
    params = []
    for _input in _graph['inputs']:
        inputs.append(json_to_tensormeta(_input))
    for _param in _graph['params']:
        params.append(json_to_tensormeta(_param))
    ops_registry = {}
    ops_registry.update(func_ops_registry)
    ops_registry.update(linalg_ops_registry)
    ops_registry.update(tosa_ops_registry)
    ops_registry.update(math_ops_registry)
    graph = Graph(
        inputs, 
        params,
        ops_registry, 
        graph_name
    )
    graph.device = _graph['device']
    for _node in _graph['nodes']:
        op_class = _node['class']
        op = globals()[op_class]()

        op._name = _node['name']
        op._children = _node['children']
        op._parents = _node['parents']
        op._arguments = _node['arguments']
        op._keyword_arguments = _node['keyword_arguments']
        op._type = next(
            (member for member in OpType.__members__.values() if member.value == _node['type']), None
        )

        # TODO : node attr tensor_meta should be  Class TensorMeta
        if ('shape' not in _node['tensor_meta']):
            op._tensor_meta = _node['tensor_meta']
        else:
            op._tensor_meta = {
                'shape' : _node['tensor_meta']['shape'],
                'dtype' : next(
                    (member for member in TensorDType.__members__.values() 
                    if member.value.upper() == _node['tensor_meta']['dtype'].upper()), None
                )
            }
        graph.add_node(op)

    for i, device in enumerate(list(set(_graph['node_map_device'].values()))):
        subgraph_name = "subgraph{}".format(i)
        graph.op_groups[subgraph_name] = []
        graph.group_map_device[subgraph_name] = DeviceType(device)

    for node, op_device in _graph['node_map_device'].items():
        op = graph.node_table[node]
        for subgraph_name, group_device in graph.group_map_device.items():
            if op_device == group_device.value:
                graph.op_groups[subgraph_name].append(op)
                break

    return graph
