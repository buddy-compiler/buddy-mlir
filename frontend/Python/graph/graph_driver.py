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
        Builds subgraphs from a given graph based on groups, and assigns a hardware type to each subgraph.

        Returns:
        - tuple: A tuple containing dictionaries of subgraphs, subgraph inputs, and subgraph outputs.
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
        
        # Identify outputs for each subgraph
        for subgraph_name in self._graph.op_groups.keys():
            subgraphs_outputs[subgraph_name] = []
            for op in self._graph.op_groups[subgraph_name]:
                for key in subgraphs_inputs.keys():
                    if op.name in subgraphs_inputs[key]:
                        subgraphs_outputs[subgraph_name].append(op.name)
                if (op.name in output_node) and (
                    op.name not in subgraphs_outputs[subgraph_name]
                ):
                    subgraphs_outputs[subgraph_name].append(op.name)
        
        subgraphs = {}

        # Construct each subgraph
        for subgraph_name in self._graph.op_groups.keys():
            subgraph_input = []
            subgraph_body = []

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

            # Assign hardware type to each subgraph (you can customize this logic)
            hardware_type = self.assign_hardware_type(subgraph_name)

            # Create subgraph and add it to the dictionary, with hardware type
            subgraph = Graph(
                subgraph_input, [], self._graph._ops_registry, subgraph_name
            )
            subgraph.body = subgraph_body
            for op in subgraph_body:
                subgraph.node_table[op.name] = op
            
            subgraph.device = hardware_type

            subgraphs[subgraph_name] = subgraph

        return subgraphs, subgraphs_inputs, subgraphs_outputs

    def assign_hardware_type(self, subgraph_name):
        """
        Assign a hardware type to the subgraph based on its name or operations.

        Args:
        - subgraph_name (str): The name of the subgraph.

        Returns:
        - str: The hardware type assigned to the subgraph (e.g., 'CPU', 'GPU', 'TPU').
        """
        # Example logic for assigning hardware type (this can be customized)
        if "GPU" in subgraph_name:
            return "GPU"
        elif "TPU" in subgraph_name:
            return "TPU"
        else:
            return "CPU"

