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
    def __init__(self, torch_fx_graph) -> None:
        self.nodes_dict = {}
        self.front_node = None
        self.create_graph(torch_fx_graph.graph.nodes)
    
    def create_graph(self, node_list):
        for node in node_list:
            new_node = Node(node)
            self.nodes_dict[new_node.name] = new_node
        self.front_node = self.nodes_dict[next(iter(node_list)).name]
    
    def __iter__(self):
        self.current_node_name = self.front_node.name
        return self
    
    def __next__(self) -> Node:
        if self.current_node_name == '':
            raise StopIteration
        temp_node = self.nodes_dict[self.current_node_name]
        self.current_node_name = temp_node.next_node
        return temp_node