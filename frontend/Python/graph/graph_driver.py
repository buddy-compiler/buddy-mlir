from mlir import ir

from .graph import Graph, GraphImporter
from .operation import FuncOp, CallOp, PlaceholderOp, OutputOp, GetItemOp


class GraphDriver:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        (
            self._subgraphs,
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = graph.build_subgraph_by_group()

    def construct_main_graph(self, do_param_pack=False):
        main_graph = Graph(
            self._graph._inputs,
            self._graph._fake_params,
            self._graph._ops_registry,
            self._graph._func_name,
        )
        for subgraph_name in self._subgraphs.keys():
            func_node = FuncOp()
            func_node.name = subgraph_name
            func_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs[subgraph_name]._inputs:
                func_node.add_argument(inp)
            for output in self._subgraphs_outputs[subgraph_name]:
                func_node.tensor_meta["shape"].append(
                    self._graph.node_table[output].tensor_meta["shape"]
                )
                func_node.tensor_meta["dtype"].append(
                    self._graph.node_table[output].tensor_meta["dtype"]
                )
            main_graph.body.append(func_node)
        for op in self._graph.body:
            if isinstance(op, PlaceholderOp):
                main_graph.body.append(op)
        # TODO: analysis topology order to sort subgraph call.
        if len(self._subgraphs) == 1:
            call_node = CallOp()
            call_node.name = "call0"
            call_node.call_func_name = list(self._subgraphs.keys())[0]
            call_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in list(self._subgraphs_inputs.values())[0]:
                call_node.add_argument(inp)
            for output in list(self._subgraphs_outputs.values())[0]:
                call_node.tensor_meta["shape"].append(
                    self._graph.node_table[output].tensor_meta["shape"]
                )
                call_node.tensor_meta["dtype"].append(
                    self._graph.node_table[output].tensor_meta["dtype"]
                )
            main_graph.body.append(call_node)
            output_node = OutputOp()
            for i, output in enumerate(list(self._subgraphs_outputs.values())[0]):
                getitem_node = GetItemOp()
                getitem_node.add_argument(call_node.name)
                getitem_node.add_argument(i)
                getitem_node.name = "getitem{}".format(i)
                output_node.add_argument(getitem_node.name)
                main_graph.body.append(getitem_node)
            output_node.name = "output"
            main_graph.body.append(output_node)
            with ir.Location.unknown(ir.Context()):
                main_importer = GraphImporter(
                    main_graph.body,
                    main_graph._fake_params,
                    main_graph._inputs,
                    do_param_pack,
                    main_graph._func_name,
                    main_graph._ops_registry,
                )
                return main_importer.import_main_graph()
