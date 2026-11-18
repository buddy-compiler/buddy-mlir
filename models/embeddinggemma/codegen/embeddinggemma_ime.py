# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional FFN FP16 three-product compensation for the K3 A100.

Weights retain their F32 parameter slots, as in the YOLOv8 IME importer:
the slots contain packed FP16 bytes, not numerical F32 weights. Two FP16
components occupy exactly the original F32 byte count. The original graph's
remaining operations and the runner ABI are unchanged.
"""

import math
from collections import Counter

import numpy
import torch
from buddy.compiler.graph import TensorDType
from buddy.compiler.graph.operation import CallExternalOp, MatmulOp
from buddy.compiler.graph.transform import k3_w4 as k3
from buddy_mlir import ir
from buddy_mlir.dialects import arith, bufferization, cf, func, memref, scf
from buddy_mlir.dialects import math as mlir_math

KERNELS_MLIR = "embeddinggemma_ime.mlir"
MMA = "embeddinggemma_comp3_mma"


def pack_weight(weight):
    """[K,N] F32 -> same-sized F32 byte container, plus power-of-two exponent."""
    weight = torch.as_tensor(weight).detach()
    if weight.dtype != torch.float32 or not bool(torch.isfinite(weight).all()):
        raise ValueError("IME weights must be finite F32")
    k, n = weight.shape
    if k % 8 or n % 64:
        raise ValueError("IME weight dimensions require K % 8 == N % 64 == 0")
    peak = float(weight.abs().max())
    exponent = 14 - math.floor(math.log2(peak)) if peak else 0
    if not -120 <= exponent <= 120:
        raise ValueError("IME weight scale out of range")

    def half_ftz(x):
        h = x.half().float()
        if not bool(torch.isfinite(h).all()):
            raise ValueError("Nonfinite FP16 weight component")
        return torch.where(
            h.abs() < 2**-14, torch.copysign(torch.zeros_like(h), h), h
        )

    scaled = weight * (2.0**exponent)
    high = half_ftz(scaled)
    low = half_ftz((scaled - high) * 2048.0)
    parts = []
    for part in (high, low):
        # [N/64,K/8,8 column groups,8 columns,8 K elements], as the
        # existing 8x64 matrix loop reads its B operand.
        packed = (
            part.T.numpy()
            .astype(numpy.float16)
            .reshape(n // 64, 8, 8, k // 8, 8)
            .transpose(0, 3, 1, 2, 4)
        )
        parts.append(numpy.ascontiguousarray(packed).reshape(-1))
    packed = numpy.concatenate(parts).view(numpy.float32).reshape(k, n)
    return packed, exponent


def rewrite_graph(graph, rows):
    """Replace exactly the 24 gate/up/down projections after eliminate_transpose."""
    if rows < 8 or rows % 8:
        raise ValueError(
            "FFN IME requires batch_size * max_seq_len divisible by 8"
        )
    rw = k3._Rewriter(graph, rows, 0)
    counts = Counter()
    for node in list(graph.body):
        if not isinstance(node, MatmulOp) or not rw.is_linear(node):
            continue
        lhs, rhs = map(str, node.args[:2])
        weight = rw.weight(rhs)
        k, n = weight.shape
        if (k, n) not in ((768, 1152), (1152, 768)):
            continue
        if list(node.tensor_meta["shape"]) != [rows, n]:
            raise ValueError(f"Unexpected FFN output shape at {node.name}")
        if graph.node_table[rhs]._children != [node.name]:
            raise ValueError(f"FFN weight has unexpected uses: {rhs}")
        packed, exponent = pack_weight(weight)
        rw.set_param(rhs, packed, TensorDType.Float32)
        name = f"embeddinggemma_comp3_m{rows}_k{k}_n{n}_e{exponent}"
        rw.kernels[name] = {
            "name": name,
            "m": rows,
            "k": k,
            "n": n,
            "exponent": exponent,
        }
        call = CallExternalOp(
            call_func_name=name,
            args=[lhs, rhs],
            args_index=[0, 0],
            tensor_meta=dict(node.tensor_meta),
            name=node.name,
            written_args=[],
        )
        call.trace_meta = node.trace_meta
        rw.replace(node, call)
        rw.link(call, [lhs, rhs])
        counts[k, n] += 1
    if counts != {(768, 1152): 48, (1152, 768): 24}:
        raise RuntimeError(
            f"Expected all 72 FFN projections, found {dict(counts)}"
        )
    graph._enable_external_calls = True
    return list(rw.kernels.values())


def lower_calls(module):
    """Use explicit caller-owned outputs before bufferization.

    An opaque returned memref may alias any live buffer. Ownership-based
    deallocation then retains buffers for runtime alias checks, preventing
    the existing liveness pass from freeing them early. A visible allocation
    and a void call express the actual ownership without changing the math.
    """
    count = 0
    with module.context, ir.Location.unknown():
        declarations = {
            op.attributes["sym_name"].value: op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and not op.regions[0].blocks
            and op.attributes["sym_name"].value.startswith(
                "embeddinggemma_comp3_m"
            )
        }

        def buffer_type(tensor_type):
            tensor_type = ir.RankedTensorType(tensor_type)
            if tensor_type.rank != 2:
                raise ValueError("FFN IME calls require rank-two tensors")
            return k3._memref(
                list(tensor_type.shape), tensor_type.element_type, [None, None]
            )

        for caller in module.body.operations:
            for region in caller.regions:
                for block in region.blocks:
                    for call in list(block.operations):
                        if call.operation.name != "func.call":
                            continue
                        name = ir.FlatSymbolRefAttr(
                            call.attributes["callee"]
                        ).value
                        if name not in declarations:
                            continue
                        tensor_type = ir.RankedTensorType(call.results[0].type)
                        with ir.InsertionPoint(call), call.location:
                            args = [
                                bufferization.ToBufferOp(
                                    buffer_type(value.type),
                                    value,
                                    read_only=True,
                                ).result
                                for value in call.operands
                            ]
                            out = k3._alloc(
                                k3._memref(
                                    list(tensor_type.shape),
                                    tensor_type.element_type,
                                )
                            )
                            destination = memref.CastOp(
                                buffer_type(tensor_type), out
                            ).result
                            func.CallOp([], name + "_out", args + [destination])
                            result = bufferization.ToTensorOp(
                                tensor_type, out, restrict=True, writable=True
                            ).result
                        call.results[0].replace_all_uses_with(result)
                        call.erase()
                        count += 1
        if count != 72:
            raise RuntimeError(
                f"Expected 72 destination-passing calls, found {count}"
            )
        for name, declaration in declarations.items():
            signature = ir.FunctionType(
                ir.TypeAttr(declaration.attributes["function_type"]).value
            )
            inputs = [buffer_type(t) for t in signature.inputs]
            inputs.append(buffer_type(signature.results[0]))
            declaration.attributes["sym_name"] = ir.StringAttr.get(
                name + "_out"
            )
            declaration.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(inputs, [])
            )
            declaration.attributes["arg_attrs"] = ir.ArrayAttr.get(
                [
                    ir.DictAttr.get(
                        {"bufferization.access": ir.StringAttr.get(access)}
                    )
                    for access in ("read", "read", "read-write")
                ]
            )
        module.operation.verify()


def _half_ftz(fn, vector, value):
    """Round to FP16, flushing subnormals while preserving the sign of zero."""
    half = k3._vec(8, ir.F16Type.get())
    bits_type = k3._vec(8, ir.IntegerType.get_signless(16))
    rounded = arith.TruncFOp(half, value).result
    bits = arith.BitcastOp(bits_type, rounded).result
    exp = arith.AndIOp(bits, fn.const(bits_type, 0x7C00)).result
    tiny = arith.CmpIOp(
        arith.CmpIPredicate.eq, exp, fn.const(bits_type, 0)
    ).result
    sign = arith.AndIOp(bits, fn.const(bits_type, 0x8000)).result
    flushed = arith.SelectOp(tiny, sign, bits).result
    result = arith.BitcastOp(half, flushed).result
    return result, arith.ExtFOp(vector, result).result


def _scale(fn, ty, x, m, k):
    """Compute the same per-tensor power-of-two scale as the offline split."""
    vec = k3._vec(32, ty.f32)
    rows = scf.ForOp(fn.idx(0), fn.idx(m), fn.idx(1), [fn.const(vec, 0.0)])
    with ir.InsertionPoint(rows.body):
        cols = scf.ForOp(
            fn.idx(0), fn.idx(k), fn.idx(32), [rows.inner_iter_args[0]]
        )
        with ir.InsertionPoint(cols.body):
            v = k3._read(
                vec,
                x,
                [rows.induction_variable, cols.induction_variable],
                fn.const(ty.f32, 0.0),
            )
            peak = arith.MaxNumFOp(
                cols.inner_iter_args[0], mlir_math.AbsFOp(v).result
            ).result
            scf.YieldOp([peak])
        scf.YieldOp([cols.results[0]])
    peak = k3._reduce(ty.f32, "maxnumf", rows.results[0])
    bits = arith.BitcastOp(ty.i32, peak).result
    field = arith.ShRUIOp(bits, fn.const(ty.i32, 23)).result
    zero = arith.CmpFOp(
        arith.CmpFPredicate.OEQ, peak, fn.const(ty.f32, 0.0)
    ).result
    normal = arith.CmpIOp(
        arith.CmpIPredicate.uge, field, fn.const(ty.i32, 21)
    ).result
    finite = arith.CmpIOp(
        arith.CmpIPredicate.ult, field, fn.const(ty.i32, 255)
    ).result
    valid = arith.OrIOp(zero, arith.AndIOp(normal, finite).result).result
    cf.AssertOp(valid, "EmbeddingGemma IME activation scale out of range")
    exponent = arith.SelectOp(
        zero,
        fn.const(ty.i32, 0),
        arith.SubIOp(fn.const(ty.i32, 141), field).result,
    ).result

    def power(biased):
        bits = arith.ShLIOp(biased, fn.const(ty.i32, 23)).result
        return arith.BitcastOp(ty.f32, bits).result

    scale = power(arith.AddIOp(fn.const(ty.i32, 127), exponent).result)
    inverse = power(arith.SubIOp(fn.const(ty.i32, 127), exponent).result)
    return scale, inverse


def _kernel(ty, spec):
    m, k, n, exponent = (spec[x] for x in ("m", "k", "n", "exponent"))
    fn = k3._Fn(spec["name"] + "_out", [ty.mat, ty.mat, ty.mat], [])
    x, weight, out = fn.args
    v8 = k3._vec(8, ty.f32)
    with ir.InsertionPoint(fn.entry):
        pack_type = k3._memref([m * k], ty.f16)
        high, low = k3._alloc(pack_type), k3._alloc(pack_type)
        scale, inverse = _scale(fn, ty, x, m, k)
        scale_vector = k3._bcast(v8, scale)
        restore = k3._mulf(inverse, fn.const(ty.f32, 2.0**-exponent))
        body, (mb,) = k3._parallel(fn, [fn.idx(m // 8)])
        with ir.InsertionPoint(body):
            kl = scf.ForOp(fn.idx(0), fn.idx(k // 8), fn.idx(1))
            with ir.InsertionPoint(kl.body):
                kk = kl.induction_variable
                rl = scf.ForOp(fn.idx(0), fn.idx(8), fn.idx(1))
                with ir.InsertionPoint(rl.body):
                    row = rl.induction_variable
                    v = k3._read(
                        v8,
                        x,
                        [
                            k3._addi(k3._muli(mb, fn.idx(8)), row),
                            k3._muli(kk, fn.idx(8)),
                        ],
                        fn.const(ty.f32, 0.0),
                    )
                    scaled = k3._mulf(v, scale_vector)
                    hi, hi32 = _half_ftz(fn, v8, scaled)
                    residual = k3._mulf(
                        k3._subf(scaled, hi32), fn.const(v8, 2048.0)
                    )
                    lo, _ = _half_ftz(fn, v8, residual)
                    at = k3._muli(
                        k3._addi(
                            k3._muli(
                                k3._addi(k3._muli(mb, fn.idx(k // 8)), kk),
                                fn.idx(8),
                            ),
                            row,
                        ),
                        fn.idx(8),
                    )
                    k3._write(hi, high, [at])
                    k3._write(lo, low, [at])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.ReduceOp([], 0)

        # Parameters are subviews of arg0.data. ExtractAlignedPointer does
        # not include their offset: include it explicitly, in bytes.
        offset = memref.ExtractStridedMetadataOp(weight).offset
        weight_ptr = k3._addi(
            k3._ptr_i64(ty, weight),
            k3._muli(
                arith.IndexCastOp(ty.i64, offset).result, fn.const(ty.i64, 4)
            ),
        )
        ah, al = k3._ptr_i64(ty, high), k3._ptr_i64(ty, low)
        body, (tile,) = k3._parallel(fn, [fn.idx((m // 8) * (n // 64))])
        with ir.InsertionPoint(body):
            mb = k3._divui(tile, fn.idx(n // 64))
            nb = arith.RemUIOp(tile, fn.idx(n // 64)).result
            scope = memref.AllocaScopeOp([])
            block = scope.bodyRegion.blocks.append()
            with ir.InsertionPoint(block):
                scratch = [
                    memref.AllocaOp(
                        k3._memref([512], ty.f32), [], [], alignment=128
                    ).result
                    for _ in range(3)
                ]
                aoff = arith.IndexCastOp(
                    ty.i64, k3._muli(mb, fn.idx(16 * k))
                ).result
                boff = arith.IndexCastOp(
                    ty.i64, k3._muli(nb, fn.idx(128 * k))
                ).result
                bh = k3._addi(weight_ptr, boff)
                bl = k3._addi(bh, fn.const(ty.i64, 2 * k * n))
                for av, bv, cv in [
                    (ah, bh, scratch[0]),
                    (al, bh, scratch[1]),
                    (ah, bl, scratch[2]),
                ]:
                    func.CallOp(
                        [],
                        MMA,
                        [
                            k3._addi(av, aoff),
                            bv,
                            k3._ptr_i64(ty, cv),
                            fn.const(ty.i64, k // 8),
                            fn.const(ty.i64, 1024),
                        ],
                    )
                jl = scf.ForOp(fn.idx(0), fn.idx(8), fn.idx(1))
                with ir.InsertionPoint(jl.body):
                    j = jl.induction_variable
                    rl = scf.ForOp(fn.idx(0), fn.idx(8), fn.idx(1))
                    with ir.InsertionPoint(rl.body):
                        row = rl.induction_variable
                        at = k3._addi(
                            k3._muli(j, fn.idx(64)), k3._muli(row, fn.idx(8))
                        )
                        values = [
                            k3._read(v8, buf, [at], fn.const(ty.f32, 0.0))
                            for buf in scratch
                        ]
                        correction = k3._mulf(
                            k3._addf(values[1], values[2]),
                            fn.const(v8, 2.0**-11),
                        )
                        result = k3._mulf(
                            k3._addf(values[0], correction),
                            k3._bcast(v8, restore),
                        )
                        k3._write(
                            result,
                            out,
                            [
                                k3._addi(k3._muli(mb, fn.idx(8)), row),
                                k3._addi(
                                    k3._muli(nb, fn.idx(64)),
                                    k3._muli(j, fn.idx(8)),
                                ),
                            ],
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                memref.AllocaScopeReturnOp([])
            scf.ReduceOp([], 0)
        memref.DeallocOp(high)
        memref.DeallocOp(low)
        func.ReturnOp([])


def gen_kernels(specs):
    """Generate wrappers around the existing FP16/F32 IME matrix loop."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            ty = k3._Types()
            k3._attn_mma_fn(ty, MMA, False)
            for spec in specs:
                _kernel(ty, spec)
        module.operation.verify()
        return str(module)
