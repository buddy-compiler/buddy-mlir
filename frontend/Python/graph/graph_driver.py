from .graph import Graph


class GraphDriver:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        (
            self._subgraphs,
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = graph.build_subgraph_by_group()
        self.construct_main_graph()

    def construct_main_graph(self):
        pass