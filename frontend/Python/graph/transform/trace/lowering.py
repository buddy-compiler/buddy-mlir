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
    attrs = {
        "id": ir.IntegerAttr.get(i64, trace_meta["id"]),
        "cycle": ir.BoolAttr.get(trace_meta["cycle"]),
        "tensor": ir.BoolAttr.get(trace_meta["tensor"]),
    }
    if "id_path" in trace_meta:
        attrs["id_path"] = ir.ArrayAttr.get(
            [ir.IntegerAttr.get(i64, value) for value in trace_meta["id_path"]]
        )
    if "tag" in trace_meta:
        attrs["tag"] = ir.StringAttr.get(trace_meta["tag"])
    for extend in trace_meta.get("extend", []):
        dialect = extend["dialect"]
        attrs[f"trace_extend_{dialect}"] = ir.BoolAttr.get(True)
        include = extend.get("include", [])
        if include:
            attrs[f"trace_{dialect}_include"] = ir.ArrayAttr.get(
                [ir.StringAttr.get(name) for name in include]
            )
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

    multiple = isinstance(op_ret, tuple | list | ir.OpResultList)
    if multiple:
        if not len(op_ret) or (len(op_ret) != 1 and node.trace_meta["tensor"]):
            raise ValueError(
                f"Tensor trace node {node.name} requires one result"
            )
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
    if multiple and len(op_ret) > 1:
        # Cycle tracing observes completion without changing secondary results.
        return (op.result, *op_ret[1:])
    return op.result
