# ===- Node.py -----------------------------------------------------------------
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
# The DL operator node of DL graph.
#
# ===---------------------------------------------------------------------------

import torch


class Node:
    def __init__(self, node) -> None:
        """
        Initializes the Node.

        Args:
            node (torch.fx.Node): The torch fx node to be 
            converted.
        """
        if node is None:
            return
        self.name = str(node.name)
        self.op = node.op
        self.op_name = getattr(node.target, "__name__", None)
        self.target = node.target
        self._input_nodes = node._input_nodes
        self.args = node.args
        self.kwargs = node.kwargs
        self.users = node.users
        self.prev_node = str(node._prev)
        self.next_node = str(node._next)
        if "tensor_meta" not in node.meta.keys():
            node.meta["tensor_meta"] = None
        if "val" not in node.meta.keys():
            node.meta["val"] = None
        self.meta = {
            "val": node.meta["val"],
            "tensor_meta": node.meta["tensor_meta"],
        }
