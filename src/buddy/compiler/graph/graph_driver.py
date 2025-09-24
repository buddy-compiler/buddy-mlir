# ===- graph_driver.py ---------------------------------------------------------
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
# This is the graph driver to drive the input graph:
#     1. Split the input graph into subgraphs.
#     2. Construct a main graph to call subgraphs with right order.
#
# ===---------------------------------------------------------------------------

from mlir import ir
from collections import deque, defaultdict

from .graph import Graph, GraphImporter, TensorMeta
from .operation import FuncOp, CallOp, PlaceholderOp, OutputOp, GetItemOp


class GraphDriver:
    """
    Class responsible for managing and driving the execution of a computational
    graph.

    Attributes:
    - _graph (Graph): The computational graph associated with this driver.
    - _subgraphs (dict): A dictionary mapping subgraph names to their
    corresponding subgraphs.
    - _subgraphs_inputs (dict): A dictionary mapping subgraph names to their
    input placeholders.
    - _subgraphs_outputs (dict): A dictionary mapping subgraph names to their
    output op's result.
    """

    def __init__(self, graph: Graph) -> None:
        """
        Initialize the GraphDriver object with a given computational graph.

        Args:
        - graph (Graph): The computational graph to be associated with this
        driver.

        Returns:
        - None
        """
        self._graph = graph
        self._subgraph_dependencies = {
            subgraph_name: set()
            for subgraph_name in list(self._graph.op_groups.keys())
        }
        self._call_table = {}
        (
            self._subgraphs,
            self._subgraphs_inputs,
            self._subgraphs_outputs,
        ) = self.build_subgraph_by_group()

    @property
    def subgraphs(self):
        return list(self._subgraphs.values())

    def build_subgraph_by_group(self):
        """
        Builds subgraphs from a given graph based on groups.

        Args:
        - graph (Graph): The graph from which subgraphs are constructed.

        Returns:
        - tuple: A tuple containing dictionaries of subgraphs, subgraph inputs,
        and subgraph outputs.
        """

        subgraphs_inputs = {}

        # Identify inputs for each subgraph
        for subgraph_name in self._graph.op_groups.keys():
            subgraphs_inputs[subgraph_name] = []
            for op in self._graph.op_groups[subgraph_name]:
                for parent in op._parents:
                    if (
                        self._graph.node_table[parent]
                        not in self._graph.op_groups[subgraph_name]
                    ):
                        subgraphs_inputs[subgraph_name].append(parent)
        subgraphs_outputs = {}
        output_node = []

        # Identify output nodes of the entire graph
        for node in self._graph.body:
            if isinstance(node, OutputOp):
                for arg in node.args:
                    output_node.append(arg)

        # Identify outputs for each subgraph and build dependencies between subgraphs
        for subgraph_name in self._graph.op_groups.keys():
            subgraphs_outputs[subgraph_name] = []
            for op in self._graph.op_groups[subgraph_name]:
                for key in subgraphs_inputs.keys():
                    if op.name in subgraphs_inputs[key]:
                        subgraphs_outputs[subgraph_name].append(op.name)
                        self._subgraph_dependencies[subgraph_name].add(key)
                if (op.name in output_node) and (
                    op.name not in subgraphs_outputs[subgraph_name]
                ):
                    subgraphs_outputs[subgraph_name].append(op.name)
        subgraphs = {}

        # Construct each subgraph
        for subgraph_name in self._graph.op_groups.keys():
            subgraph_input = []
            subgraph_body = []
            subgraph_device = self._graph.group_map_device[subgraph_name]

            # Construct input placeholder nodes
            for inp in subgraphs_inputs[subgraph_name]:
                node = self._graph.node_table[inp]
                node_shape = node.tensor_meta["shape"]
                node_dtype = node.tensor_meta["dtype"]
                input_tensor_meta = TensorMeta(node_shape, node_dtype)
                subgraph_input.append(input_tensor_meta)
                placeholder_node = PlaceholderOp()
                placeholder_node.name = inp
                placeholder_node.tensor_meta = input_tensor_meta
                for op in self._graph.op_groups[subgraph_name]:
                    if inp in node._parents:
                        placeholder_node.add_children(op.name)
                subgraph_body.append(placeholder_node)

            # Add operations to subgraph body
            for op in self._graph.op_groups[subgraph_name]:
                subgraph_body.append(op)

            # Construct output node
            output_node = OutputOp()
            output_node.name = "output"
            for output in subgraphs_outputs[subgraph_name]:
                output_node.add_argument(output)
                output_node.add_parent(output)
            subgraph_body.append(output_node)

            # Create subgraph and add it to the dictionary
            subgraph = Graph(
                subgraph_input,
                [],
                self._graph._ops_registry,
                subgraph_name,
                subgraph_device,
                verbose=self._graph._verbose,
            )
            subgraph.body = subgraph_body
            for op in subgraph_body:
                subgraph.node_table[op.name] = op
            subgraphs[subgraph_name] = subgraph

        return subgraphs, subgraphs_inputs, subgraphs_outputs

    def topological_sort_subgraph(self):
        """
        Performs topological sorting on the subgraphs based on their dependencies.
        Args:
        - graph (Graph): The graph from which subgraphs are constructed.
        Returns:
        - list: A list of subgraph names in topological order if the graph is acyclic; otherwise, None.
        """
        # Calculate in degree of each subgraph
        in_degree = {
            subgraph_name: 0 for subgraph_name in list(self._subgraphs.keys())
        }
        for src, dests in self._subgraph_dependencies.items():
            for dest in dests:
                in_degree[dest] += 1
        # Topological sorting
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in self._subgraph_dependencies[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # TODO: If the custom subgraph partitioning is illegal, further partition the subgraph to make it valid.
        return (
            topo_order
            if len(topo_order) == len(list(self._subgraphs.keys()))
            else None
        )

    def construct_main_graph(self, do_param_pack=False):
        """
        Constructs the main computational graph by incorporating subgraphs' call
        and placeholder operations.

        Args:
        - do_param_pack (bool): Flag indicating whether parameter packing should
        be performed. Defaults to False.

        Returns:
        - Graph: The main computational graph constructed.

        Note: The actual call sequence and topology analysis are pending
        implementation.

        """
        main_graph = Graph(
            self._graph._inputs,
            self._graph._fake_params,
            self._graph._ops_registry,
            self._graph._func_name,
            self._graph._verbose,
        )

        # Adding FuncOp nodes for each subgraph
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
            main_graph.add_node(func_node)

        # Adding placeholder operations from the original graph
        for op in self._graph.body:
            if isinstance(op, PlaceholderOp):
                main_graph.add_node(op)
        # Analysis topology order to sort subgraph call.
        topo_order = self.topological_sort_subgraph()
        if topo_order == None:
            print("Error : Graph Partitioning is illegal!")
            return None
        # Adding CallOp to invoke the single subgraph
        for i, subgraph_name in enumerate(topo_order):
            call_node = CallOp()
            call_node.name = "call{}".format(i)
            call_node.call_func_name = subgraph_name
            call_node.tensor_meta = {"shape": [], "dtype": []}
            for inp in self._subgraphs_inputs[subgraph_name]:
                if inp in main_graph.node_table:
                    call_node.add_argument(inp)
                    continue
                for key, value in self._subgraphs_outputs.items():
                    if inp in value:
                        call_node.add_argument(
                            arg=self._call_table[key].name,
                            arg_index=value.index(inp),
                        )
                        break
            for output in self._subgraphs_outputs[subgraph_name]:
                call_node.tensor_meta["shape"].append(
                    self._graph.node_table[output].tensor_meta["shape"]
                )
                call_node.tensor_meta["dtype"].append(
                    self._graph.node_table[output].tensor_meta["dtype"]
                )
            self._call_table[subgraph_name] = call_node
            main_graph.add_node(call_node)
        # Adding GetItemOps to retrieve individual output tensors
        output_node = OutputOp()
        for i, output in enumerate(self._subgraphs_outputs[topo_order[-1]]):
            getitem_node = GetItemOp()
            getitem_node.add_argument(call_node.name)
            getitem_node.add_argument(i)
            getitem_node.name = "getitem{}".format(i)
            output_node.add_argument(getitem_node.name)
            main_graph.add_node(getitem_node)
        # Marking the final output of the main graph
        output_node.name = "output"
        main_graph.add_node(output_node)
        # Importing the main graph
        with ir.Location.unknown(ir.Context()):
            main_importer = GraphImporter(
                main_graph.body,
                main_graph._fake_params,
                main_graph._inputs,
                main_graph._func_name,
                main_graph._ops_registry,
                do_param_pack,
            )
            return main_importer.import_main_graph()
