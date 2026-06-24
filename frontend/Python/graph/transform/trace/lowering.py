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


def _trace_attrs(node) -> dict:
    trace_meta = node.trace_meta
    if trace_meta is None:
        return {}

    i64 = ir.IntegerType.get_signless(64)
    attrs = {"id": ir.IntegerAttr.get(i64, trace_meta["id"])}
    if "tag" in trace_meta:
        attrs["tag"] = ir.StringAttr.get(trace_meta["tag"])
    return attrs


def _insert_op(op: ir.Operation) -> ir.Operation:
    return op


def _as_value(value):
    if isinstance(value, ir.OpResult):
        return value
    elif isinstance(value, ir.Operation):
        if len(value.results) != 1:
            raise ValueError("Trace op must lower to one result")
        return value.result
    elif isinstance(value, ir.OpView):
        if len(value.operation.results) != 1:
            raise ValueError("Trace op must lower to one result")
        return value.operation.result
    return value


def trace_op_start(node):
    if not has_trace(node):
        return
    _insert_op(
        ir.Operation.create(
            "buddy_trace.start",
            attributes=_trace_attrs(node),
        )
    )


def trace_op_result(node, op_ret):
    if not has_trace(node):
        return op_ret

    if isinstance(op_ret, tuple | list | ir.OpResultList):
        if len(op_ret) != 1:
            raise ValueError(f"Trace node {node.name} must lower to one result")
        value = _as_value(op_ret[0])
    else:
        value = _as_value(op_ret)

    if not isinstance(value, (ir.OpResult, ir.BlockArgument)):
        raise TypeError(f"Cannot trace result of node {node.name}")

    op = _insert_op(
        ir.Operation.create(
            "buddy_trace.end",
            results=[value.type],
            operands=[value],
            attributes=_trace_attrs(node),
        )
    )
    return op.result
