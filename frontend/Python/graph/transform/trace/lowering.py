# ===- lowering.py -------------------------------------------------------------
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

import buddy_mlir.ir as ir


def has_trace(node) -> bool:
    return node.trace_meta is not None


def _set_trace_attrs(node, value) -> None:
    trace_meta = node.trace_meta
    if trace_meta is None:
        return

    owner = None
    if isinstance(value, ir.OpResult):
        owner = value.owner
    elif isinstance(value, ir.Operation):
        owner = value
    elif isinstance(value, ir.OpView):
        owner = value.operation

    if owner is None:
        raise TypeError(f"Cannot attach trace metadata to node {node.name}")

    i64 = ir.IntegerType.get_signless(64)
    owner.attributes["buddy.trace_id"] = ir.IntegerAttr.get(
        i64, trace_meta["id"]
    )
    if "tag" in trace_meta:
        owner.attributes["buddy.trace_tag"] = ir.StringAttr.get(
            trace_meta["tag"]
        )


def trace_op_result(node, op_ret):
    if not has_trace(node):
        return
    if isinstance(op_ret, tuple | list | ir.OpResultList):
        if len(op_ret) != 1:
            raise ValueError(f"Trace node {node.name} must lower to one result")
        _set_trace_attrs(node, op_ret[0])
        return
    _set_trace_attrs(node, op_ret)
