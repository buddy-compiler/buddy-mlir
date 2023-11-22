# ===- graph.py ----------------------------------------------------------------
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
# The DL graph from torch fx graph.
#
# ===---------------------------------------------------------------------------

from .node import Node

class Graph:
    """
    Graph is a middle IR for torch fx graph, which converts an FX Graph into an 
    equivalent Graph IR.

    Attributes:
        nodes_dict: The dict store torch fx nodes.
        front_node: The first node in Graph.
    """
    def __init__(self, torch_fx_graph) -> None:
        """
        Initializes the Graph.

        Args:
            torch_fx_graph (torch.fx.GraphModule): The torch_fx_graph to be 
            converted.
        """
        self.nodes_dict = {}
        self.front_node = None
        self.create_graph(torch_fx_graph.graph.nodes)
    
    def create_graph(self, node_list):
        """
        Convert the provided FX Graph nodes to Graph IR.

        Args:
            node_list (List[torch.fx.Node]): The Graph nodes to be converted.
        """
        for node in node_list:
            new_node = Node(node)
            self.nodes_dict[new_node.name] = new_node
        self.front_node = self.nodes_dict[next(iter(node_list)).name]
    
    def __iter__(self):
        """
        Initializes the Graph iterator.
        """
        self.current_node_name = self.front_node.name
        return self
    
    def __next__(self) -> Node:
        """
        Get next node in Graph.
        """
        if self.current_node_name == '':
            raise StopIteration
        temp_node = self.nodes_dict[self.current_node_name]
        self.current_node_name = temp_node.next_node
        return temp_node
