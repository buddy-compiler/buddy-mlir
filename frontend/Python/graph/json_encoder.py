import json
from .graph import Graph
from .graph_driver import GraphDriver
from .operation import Op


class GraphList:
    def __init__(self, graph : Graph) -> None:
        self._graphs = [graph]

class BuddyGraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Graph):
            return {
                'graph_name' : obj._func_name,
                'nodes' : obj._body,
                'device' : obj.device
            }
        elif isinstance(obj, Op):
            return {
                'name' : obj._name,
                'children' : obj._children,
                'parents' : obj._parents,
                'type' : obj._op_type._name_
            }
        elif isinstance(obj, GraphList):
            return {
                "graphs" : obj._graphs
            }
        else:
            return super().default(obj)