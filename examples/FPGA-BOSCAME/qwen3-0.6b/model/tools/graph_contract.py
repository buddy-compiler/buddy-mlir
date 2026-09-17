"""Record the compiler's model ABI and remaining dense work, without guessing."""
import math
import re


def describe_entry_abi(graph):
    """Call after lower_to_top_level_ir and before destructive LLVM lowering."""
    from buddy_mlir import ir
    function = next(op for op in graph._imported_module.body.operations
                    if op.operation.name == "func.func"
                    and ir.StringAttr(op.attributes["sym_name"]).value == graph._func_name)
    function_type = ir.FunctionType(ir.TypeAttr(function.attributes["function_type"]).value)
    nodes = list(graph.params) + list(graph.inputs)
    output_nodes = next(op for op in graph.body if type(op).__name__ == "OutputOp").args

    def entry(index, tensor_type, names):
        tensor = ir.RankedTensorType(tensor_type)
        shape = list(tensor.shape)
        rank = len(shape)
        return {"index": index, "node": names[index], "shape": shape,
                "dtype": str(tensor.element_type), "rank": rank,
                "descriptor_bytes_lp64": 8 * (3 + 2 * rank),
                "llvm_descriptor": "{ ptr, ptr, i64, [" + str(rank)
                                   + " x i64], [" + str(rank) + " x i64] }"}

    inputs = [entry(i, typ, [node.name for node in nodes])
              for i, typ in enumerate(function_type.inputs)]
    outputs = [entry(i, typ, output_nodes)
               for i, typ in enumerate(function_type.results)]
    offset = 0
    for output in outputs:
        output["aggregate_offset_bytes_lp64"] = offset
        offset += output["descriptor_bytes_lp64"]
    return {"entry": graph._func_name, "source": "actual imported func.func type",
            "inputs": inputs, "outputs": outputs, "parameter_count": len(graph.params),
            "result_descriptor_count": len(outputs),
            "result_aggregate_bytes_lp64": offset,
            "contract": "CIFACE result aggregate pointer first, then one descriptor pointer per input; descriptor offsets and strides must be honored"}


def remaining_dense_work(graph):
    """List every unreplaced matmul/attention, including small RoPE products."""
    from triton_call_replace import shape_of, parents_of
    remaining = []
    for op in graph.body:
        kind = type(op).__name__
        if kind not in ("MatmulOp", "BatchMatmulOp", "ScaledDotProductFlashAttentionForCpuOp"):
            continue
        shapes = [shape_of(parent) for parent in parents_of(graph, op)]
        macs = None
        if kind in ("MatmulOp", "BatchMatmulOp") and len(shapes) == 2:
            a, b = shapes
            if a and b and len(a) >= 2 and len(b) >= 2:
                macs = math.prod(a[:-2]) * a[-2] * a[-1] * b[-1]
        elif kind == "ScaledDotProductFlashAttentionForCpuOp" and len(shapes) >= 2:
            q, k = shapes[:2]
            if q and k:
                macs = 2 * math.prod(q[:-2]) * q[-2] * q[-1] * k[-2]
        remaining.append({"node": op.name, "kind": kind, "operand_shapes": shapes,
                          "estimated_macs": macs,
                          "large": macs is None or macs >= 1_000_000,
                          "fallback": "generic Buddy lowering; no Triton call",
                          "performance": "unverified; large entries block full kernel coverage"})
    return {"remaining": remaining,
            "large_uncovered": [entry for entry in remaining if entry["large"]]}


def llvm_entry_evidence(text, entry):
    """Keep exact LLVM signature and directly known malloc sizes.

    Static call-site sums are not peak estimates: loops, frees, and dynamic
    allocations require runtime measurement. Unknown sizes are kept explicit.
    """
    lines = text.splitlines()
    signatures = [line for line in lines if line.startswith("define ")
                  and re.search(r"@(?:_mlir_ciface_)?" + re.escape(entry) + r"\(", line)]
    mallocs = []
    for line in lines:
        if re.search(r"\bcall\b.*@(?:malloc|aligned_alloc)\(", line):
            constant = re.search(r"@malloc\(i64 (\d+)\)", line)
            mallocs.append({"llvm": line.strip(),
                            "constant_bytes": int(constant[1]) if constant else None})
    return {"llvm_signatures": signatures, "allocation_call_sites": mallocs,
            "allocation_peak_bytes": None,
            "peak_note": "Not derivable from call-site counting; measure scoped arena high-water mark"}
