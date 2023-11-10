# ===- maxpool2d_simplify.py ---------------------------------------------------
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
# simplify the maxpool2d with getitem.
#
# ===---------------------------------------------------------------------------

import operator
from .. import Graph, Node

def maxpool2d_simplify(graph: Graph):
    for node in graph:
        if node.op_name == 'max_pool2d_with_indices.default':
            getitem_num = 0
            for user in node.users.keys():
                if graph.nodes_dict[str(user)].target is operator.getitem:
                    getitem_num += 1
                    getitem_node = graph.nodes_dict[str(user)]
            if getitem_num == 1 and len(node.users.keys()) == 1:
                new_node = Node(None)
                new_node.name = getitem_node.name
                new_node.op = node.op
                new_node.op_name = 'maxpool2d.tosa_default'
                new_node.target = node.target
                new_node._input_nodes = node._input_nodes
                new_node.args = node.args
                new_node.kwargs = node.kwargs
                new_node.users = getitem_node.users
                new_node.prev_node = node.prev_node
                if node.next_node == getitem_node.name:
                    new_node.next_node = getitem_node.next_node
                else:
                    new_node.next_node = node.next_node
                new_node.meta = {}
                new_node.meta['val'] = getitem_node.meta['val']
                new_node.meta['tensor_meta'] = getitem_node.meta['tensor_meta']
                graph.nodes_dict[node.prev_node].next_node = new_node.name
                if node.next_node != getitem_node.name:
                    graph.nodes_dict[node.next_node].prev_node = new_node.name
                    graph.nodes_dict[getitem_node.next_node].prev_node = getitem_node.prev_node
                else:
                    graph.nodes_dict[getitem_node.next_node].prev_node = new_node.name
                del graph.nodes_dict[node.name]
                del graph.nodes_dict[getitem_node.name]
                graph.nodes_dict[new_node.name] = new_node


