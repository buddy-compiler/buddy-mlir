#!/usr/bin/env python3
"""Render a saved Buddy op dump as an ordered, readable dataflow listing.

The operator-mapping inventory has to come from the real imported graph, not
from a wish list, so this prints every op in body order with its operand and
result shapes. Op *type* alone is not enough to identify a kernel: the same
``MatmulOp`` is a different Triton/AME case depending on shapes and on whether
the operand is a view, so both are shown.
"""
import argparse
import json
from pathlib import Path


def fmt_shape(shape):
    return "?" if shape is None else "x".join(str(d) for d in shape)


def short_dtype(dtype):
    if not isinstance(dtype, str):
        return str(dtype)
    return dtype.split(".")[-1].replace("TensorDType", "").lower() or "?"


def render(path, only=None, summary=False):
    ops = json.loads(Path(path).read_text())
    lines = []
    for op in ops:
        kind = op["op_type"]
        if only and kind not in only:
            continue
        out = op.get("tensor_meta", {})
        out_shape = out.get("shape")
        if isinstance(out_shape, list) and out_shape and isinstance(out_shape[0], list):
            out_text = " ; ".join(fmt_shape(s) for s in out_shape)
        else:
            out_text = fmt_shape(out_shape)
        parents = op.get("parents") or []
        pshapes = op.get("parent_shapes") or []
        ptypes = op.get("parent_types") or []
        operands = []
        for idx, parent in enumerate(parents):
            shape = pshapes[idx] if idx < len(pshapes) else None
            ptype = ptypes[idx] if idx < len(ptypes) else "?"
            marker = ""
            if ptype in ("ViewOp", "PermuteOp", "UnsqueezeOp", "SliceOp",
                         "ExpandOp", "CloneOp", "AliasOp"):
                marker = f":{ptype[:-2]}"
            operands.append(f"{parent}[{fmt_shape(shape)}]{marker}")
        lines.append(
            f"{op['index']:4d} {op['name']:<44} {kind:<38} "
            f"({', '.join(operands)}) -> {out_text}")
    if summary:
        counts = {}
        for op in ops:
            counts[op["op_type"]] = counts.get(op["op_type"], 0) + 1
        lines.append("")
        lines.append("op histogram:")
        for kind, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {count:6d}  {kind}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--only", action="append", default=None,
                        help="restrict to this op type (repeatable)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    print(render(args.dump, set(args.only) if args.only else None, args.summary))
