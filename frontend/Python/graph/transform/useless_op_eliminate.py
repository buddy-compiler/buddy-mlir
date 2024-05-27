# ===- useless_op_eliminate.py ---------------------------------------------------
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
# eliminate the useless ops.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *


def maxpool2d_simplify(graph: Graph):
    """
    Fuse the maxpool op and getitem op to simpllify graph.

    Args:
        graph (Graph): The Graph to be simplified.
    """
    for i, node in enumerate(graph.body):
        if isinstance(node, MaxPool2dWithIndicesOp):
            getitem_num = 0
            for user in node._children:
                if isinstance(graph.node_table[user], GetItemOp):
                    getitem_num += 1
                    getitem_node = graph.node_table[user]
            if (
                getitem_num == 1
                and len(node._children) == 1
                and getitem_node.args[1] == 0
            ):
                new_node = MaxPool2dOp()
                new_node.name = getitem_node.name
                for arg in node.args:
                    new_node.add_argument(arg)
                for parent in node._parents:
                    new_node.add_parent(parent)
                for child in getitem_node._children:
                    new_node.add_children(child)
                new_node.tensor_meta["shape"] = getitem_node.tensor_meta[
                    "shape"
                ]
                new_node.tensor_meta["dtype"] = getitem_node.tensor_meta[
                    "dtype"
                ]
                new_node._layout = node._layout
                del graph.node_table[node.name]
                del graph.node_table[getitem_node.name]
                graph.node_table[new_node.name] = new_node
                del graph.body[i]
                for j, op in enumerate(graph.body):
                    if op == getitem_node:
                        graph.body[j] = new_node
                        break
