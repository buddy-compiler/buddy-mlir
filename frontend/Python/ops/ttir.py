# ===- ttir.py ----------------------------------------------------------------
#
# Lower Buddy Graph ops to TTIR dialect ops via the ttmlir Python API.
# Used by ``Graph.lower_to_ttir()`` (see ``graph/ttir_import.py``).
#
# Element type is fixed per import (BF16 or F32) via ``TTIRSandbox.elt_type``.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TTIRSandbox:
    """Per-import context: element type, MLIR context, location (ttmlir)."""

    ctx: object
    loc: object
    elt_type: object


def _tensor_meta_shape_dtype(node):
    tm = node.tensor_meta
    if isinstance(tm, dict):
        return list(tm["shape"]), tm["dtype"]
    return list(tm.shape), tm.dtype


def _ranked_type(shape: Sequence[int], sb: TTIRSandbox):
    from ttmlir.ir import RankedTensorType

    return RankedTensorType.get(list(shape), sb.elt_type)


def _mlir_element_type_for_tensor_dtype(ctx, dtype, default_float_elt):
    from ttmlir.ir import BF16Type, F16Type, F32Type, IntegerType

    if dtype is None:
        return default_float_elt
    name = str(getattr(dtype, "value", dtype)).lower()
    if name in ("bfloat16", "bf16") or "bfloat16" in name:
        return BF16Type.get()
    if name in ("float16", "f16") or "float16" in name:
        return F16Type.get()
    if name in ("float32", "f32") or "float32" in name:
        return F32Type.get()
    if name in ("int64", "i64") or "int64" in name:
        return IntegerType.get_signless(64, ctx)
    if name in ("int32", "i32") or "int32" in name:
        return IntegerType.get_signless(32, ctx)
    if name in ("bool", "i1") or "bool" in name:
        return IntegerType.get_signless(1, ctx)
    return default_float_elt


def _cast_to_element_type(val, element_type, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    if not hasattr(val, "type"):
        return val
    try:
        cur_elt = val.type.element_type
    except Exception:
        return val
    if str(cur_elt) == str(element_type):
        return val
    cast_rt = RankedTensorType.get(list(val.type.shape), element_type)
    return ttir.typecast(cast_rt, val, loc=sb.loc)


def _cast_to_sandbox_elt(val, sb: TTIRSandbox):
    """Ensure ``val`` has the sandbox default element type (e.g. bf16/f32).

    Shape-only ops (reshape, unsqueeze, squeeze, permute, transpose) build their
    output type using ``sb.elt_type`` but some producers (e.g. ``ttir.arange``)
    legitimately emit integer tensors (i64/i32). Feeding such a tensor directly
    into a shape op creates a type mismatch where the op claims to be bf16/f32
    but the underlying storage is still integer. This cascades through the
    TTIR -> TTNN -> runtime pipeline and produces kernels compiled for int
    dtypes against bf16-annotated tensors, which hits JIT kernel-selection
    bugs in tt-metal (e.g. eltwise_binary.cpp picked for SFPU int32 ge).
    Insert an explicit ``ttir.typecast`` to keep input/output dtypes honest.
    """
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    if not hasattr(val, "type"):
        return val
    try:
        cur_elt = val.type.element_type
    except Exception:
        return val
    if cur_elt == sb.elt_type:
        return val
    cast_rt = RankedTensorType.get(list(val.type.shape), sb.elt_type)
    return ttir.typecast(cast_rt, val, loc=sb.loc)


def _shape_op_type(shape: Sequence[int], node, sb: TTIRSandbox):
    from ttmlir.ir import RankedTensorType

    element_type = sb.elt_type
    if node is not None and os.environ.get("BUDDY_TTIR_PRESERVE_SHAPE_TYPES") == "1":
        _, dtype = _tensor_meta_shape_dtype(node)
        element_type = _mlir_element_type_for_tensor_dtype(
            sb.ctx, dtype, sb.elt_type
        )
    return RankedTensorType.get([int(x) for x in shape], element_type)


def _cast_for_shape_op(val, result_type, sb: TTIRSandbox):
    if os.environ.get("BUDDY_TTIR_PRESERVE_SHAPE_TYPES") == "1":
        return _cast_to_element_type(val, result_type.element_type, sb)
    return _cast_to_sandbox_elt(val, sb)


def _i32_array_attr(vals: Sequence[int], sb: TTIRSandbox):
    """Stride / padding / kernel arrays for TTIR conv and pool (``array<i32: ...>``)."""
    from ttmlir.ir import DenseI32ArrayAttr

    return DenseI32ArrayAttr.get([int(x) for x in vals], context=sb.ctx)


def _i32_attr(val: int, sb: TTIRSandbox):
    from ttmlir.ir import IntegerAttr, IntegerType

    t = IntegerType.get_signless(32, context=sb.ctx)
    return IntegerAttr.get(t, int(val))


def _bool_attr(val: bool, sb: TTIRSandbox):
    from ttmlir.ir import BoolAttr

    return BoolAttr.get(bool(val), context=sb.ctx)


def _dense_i64_perm(ctx, perm: Sequence[int]):
    from ttmlir.ir import DenseI64ArrayAttr

    return DenseI64ArrayAttr.get(list(perm), context=ctx)


def _permute_val(inp, perm: Sequence[int], out_shape: Sequence[int], node, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    if node is None and os.environ.get("BUDDY_TTIR_PRESERVE_SHAPE_TYPES") == "1":
        rt = RankedTensorType.get([int(x) for x in out_shape], inp.type.element_type)
    else:
        rt = _shape_op_type(out_shape, node, sb)
    inp = _cast_for_shape_op(inp, rt, sb)
    p = _dense_i64_perm(sb.ctx, perm)
    return ttir.permute(rt, inp, p)


def _nchw_to_nhwc(val, sb: TTIRSandbox):
    sh = list(val.type.shape)
    n, c, h, w = sh
    out_shape = [n, h, w, c]
    return _permute_val(val, [0, 2, 3, 1], out_shape, None, sb)


def _nhwc_to_nchw(val, sb: TTIRSandbox):
    sh = list(val.type.shape)
    n, h, w, c = sh
    out_shape = [n, c, h, w]
    return _permute_val(val, [0, 3, 1, 2], out_shape, None, sb)


def _reshape_to(val, new_shape: Sequence[int], _node, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    total = 1
    for d in val.type.shape:
        total *= int(d)
    shape_list = list(new_shape)
    neg_one_cnt = 0
    rest = 1
    for dim_siz in shape_list:
        if dim_siz == -1:
            neg_one_cnt += 1
            continue
        rest *= int(dim_siz)
    if neg_one_cnt != 0:
        if neg_one_cnt > 1 or total % rest != 0:
            raise ValueError("Cannot infer reshape -1 for TTIR reshape.")
        for i, _ in enumerate(shape_list):
            if shape_list[i] == -1:
                shape_list[i] = total // rest
    element_type = sb.elt_type
    if os.environ.get("BUDDY_TTIR_PRESERVE_SHAPE_TYPES") == "1":
        _, dtype = _tensor_meta_shape_dtype(_node)
        element_type = _mlir_element_type_for_tensor_dtype(
            sb.ctx, dtype, sb.elt_type
        )
        val = _cast_to_element_type(val, element_type, sb)
    else:
        val = _cast_to_sandbox_elt(val, sb)
    rt = RankedTensorType.get([int(x) for x in shape_list], element_type)
    return ttir.reshape(rt, val, [int(x) for x in shape_list])


def permute_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    input_tensor = symbol_table[(str(node.args[0]), 0)]
    perm = list(node.args[1])
    init_shape = list(input_tensor.type.shape)
    new_shape = [init_shape[i] for i in perm]
    rt = _shape_op_type(new_shape, node, sb)
    input_tensor = _cast_for_shape_op(input_tensor, rt, sb)
    p = _dense_i64_perm(sb.ctx, perm)
    return ttir.permute(rt, input_tensor, p)


def t_op(node, symbol_table, sb: TTIRSandbox):
    assert len(node.args) == 1
    input1 = symbol_table[(str(node.args[0]), 0)]
    input_shape = list(input1.type.shape)
    assert len(input_shape) == 2, "TOp TTIR: expect 2D tensor."
    out_shape = [input_shape[1], input_shape[0]]
    return _permute_val(input1, [1, 0], out_shape, node, sb)


def transpose_op(node, symbol_table, sb: TTIRSandbox):
    assert len(node.args) == 3
    input1 = symbol_table[(str(node.args[0]), 0)]
    dim1 = int(node.args[1])
    dim2 = int(node.args[2])
    input_shape = list(input1.type.shape)
    perm_list = list(range(len(input_shape)))
    perm_list[dim1], perm_list[dim2] = perm_list[dim2], perm_list[dim1]
    tm = node.tensor_meta
    out_shape = list(tm["shape"]) if isinstance(tm, dict) else list(tm.shape)
    return _permute_val(input1, perm_list, out_shape, node, sb)


def relu_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    assert len(node.args) == 1
    input1 = symbol_table[(str(node.args[0]), 0)]
    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_type(out_shape, sb)
    return ttir.relu(rt, input1)


def reshape_op(node, symbol_table, sb: TTIRSandbox):
    input1 = symbol_table[(str(node.args[0]), 0)]
    shape_arg = node.args[1]
    if isinstance(shape_arg, (list, tuple)):
        new_shape = [int(x) for x in shape_arg]
    else:
        new_shape = [int(shape_arg)]
    return _reshape_to(input1, new_shape, node, sb)


def view_op(node, symbol_table, sb: TTIRSandbox):
    return reshape_op(node, symbol_table, sb)


def add_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import (
        DenseElementsAttr,
        FloatAttr,
        IntegerType,
        RankedTensorType,
        IntegerAttr,
    )

    def _as_value(arg, peer):
        v = symbol_table.get((str(arg), 0), arg)
        if hasattr(v, "type"):
            return v
        if peer is not None and isinstance(v, (int, float, bool)):
            sh = list(peer.type.shape)
            elt = peer.type.element_type
            rt_s = RankedTensorType.get(sh, elt)
            if isinstance(elt, IntegerType):
                iv = int(v)
                attr = DenseElementsAttr.get_splat(
                    rt_s, IntegerAttr.get(elt, iv)
                )
            else:
                attr = DenseElementsAttr.get_splat(
                    rt_s, FloatAttr.get(elt, float(v))
                )
            return ttir.constant(rt_s, attr, loc=sb.loc)
        raise NotImplementedError(
            f"TTIR add: could not resolve operand {arg!r} (peer={peer!r})."
        )

    a0, a1 = node.args[0], node.args[1]
    input1 = symbol_table.get((str(a0), 0), a0)
    input2 = symbol_table.get((str(a1), 0), a1)
    if not hasattr(input1, "type") and hasattr(input2, "type"):
        input1 = _as_value(a0, input2)
    elif not hasattr(input2, "type") and hasattr(input1, "type"):
        input2 = _as_value(a1, input1)
    elif not hasattr(input1, "type") and not hasattr(input2, "type"):
        raise NotImplementedError("TTIR add: expected at least one tensor SSA operand.")
    out_shape, _ = _tensor_meta_shape_dtype(node)
    preserve_f32 = False
    if os.environ.get("BUDDY_TTIR_PRESERVE_F32_ADD") == "1":
        from ttmlir.ir import F32Type

        if (
            hasattr(input1, "type")
            and hasattr(input2, "type")
            and input1.type.element_type == input2.type.element_type
            and isinstance(input1.type.element_type, F32Type)
        ):
            preserve_f32 = True
    rt = (
        RankedTensorType.get(list(out_shape), input1.type.element_type)
        if preserve_f32
        else _ranked_type(out_shape, sb)
    )

    def _cast_to_result_elt(v):
        elt_out = rt.element_type
        if v.type.element_type == elt_out:
            return v
        cast_rt = RankedTensorType.get(list(v.type.shape), elt_out)
        return ttir.typecast(cast_rt, v, loc=sb.loc)

    input1 = _cast_to_result_elt(input1)
    input2 = _cast_to_result_elt(input2)
    return ttir.add(rt, input1, input2)


def addmm_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    inp = symbol_table[(str(node.args[0]), 0)]
    mat1 = symbol_table[(str(node.args[1]), 0)]
    mat2 = symbol_table[(str(node.args[2]), 0)]
    s0 = list(mat1.type.shape)
    s1 = list(mat2.type.shape)
    m, k0 = int(s0[0]), int(s0[1])
    k1, n = int(s1[0]), int(s1[1])
    assert k0 == k1, (s0, s1)
    out_shape, _ = _tensor_meta_shape_dtype(node)
    mm_type = _ranked_type([m, n], sb)
    mm = ttir.matmul(mm_type, mat1, mat2)
    rt = _ranked_type(out_shape, sb)
    return ttir.add(rt, inp, mm)


def _bias_to_conv_bias(bias_val, out_channels: int, sb: TTIRSandbox):
    """1D bias [O] -> [1,1,1,O] for NHWC conv."""
    from ttmlir.dialects import ttir

    bshape = list(bias_val.type.shape)
    if len(bshape) == 1 and int(bshape[0]) == out_channels:
        rt = _ranked_type([1, 1, 1, out_channels], sb)
        return ttir.reshape(rt, bias_val, [1, 1, 1, out_channels])
    return bias_val


def conv2d_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    assert len(node.args) == 9
    _input = node.args[0]
    weight = node.args[1]
    bias = node.args[2]
    stride = node.args[3]
    input_padding = node.args[4]
    dilation = node.args[5]
    _transposed = node.args[6]
    _out_pad = node.args[7]
    groups = int(node.args[8])

    if _transposed:
        raise NotImplementedError("TTIR conv2d: transposed conv not implemented.")

    input_val = symbol_table[(str(_input), 0)]
    weight_val = symbol_table[(str(weight), 0)]
    wshape = list(weight_val.type.shape)
    out_shape, _ = _tensor_meta_shape_dtype(node)
    out_ch = int(wshape[0])

    if len(node._parents) == 2:
        bias_tensor = None
    else:
        bias_tensor = _bias_to_conv_bias(
            symbol_table[(str(bias), 0)], out_ch, sb
        )

    if isinstance(stride, (list, tuple)):
        sy, sx = int(stride[0]), int(stride[1])
    else:
        sy = sx = int(stride)
    if isinstance(dilation, (list, tuple)):
        dy, dx = int(dilation[0]), int(dilation[1])
    else:
        dy = dx = int(dilation)

    if len(input_padding) == 1:
        t = b = l = r = int(input_padding[0])
    elif len(input_padding) == 2:
        t = b = int(input_padding[0])
        l = r = int(input_padding[1])
    else:
        t, b, l, r = (int(x) for x in input_padding)

    kh, kw = int(wshape[2]), int(wshape[3])

    def _adjust(input_size, kernel, dil, stride_val, pad0, pad1):
        base = input_size - 1 - (kernel - 1) * dil
        total = base + pad0 + pad1
        remainder = total % stride_val
        if remainder == 0:
            return pad0, pad1
        pad_needed = stride_val - remainder
        return pad0, pad1 + pad_needed

    if node._layout.find("NCHW") != -1:
        ishape = list(input_val.type.shape)
        input_h = int(ishape[2])
        input_w = int(ishape[3])
        t, b = _adjust(input_h, kh, dy, sy, t, b)
        l, r = _adjust(input_w, kw, dx, sx, l, r)
        input_padding_tt = [t, l, b, r]
        input_val = _nchw_to_nhwc(input_val, sb)
    else:
        ishape = list(input_val.type.shape)
        input_h = int(ishape[1])
        input_w = int(ishape[2])
        t, b = _adjust(input_h, kh, dy, sy, t, b)
        l, r = _adjust(input_w, kw, dx, sx, l, r)
        input_padding_tt = [t, l, b, r]

    # TTIR expects weight (O, C, KH, KW), same as PyTorch; do not apply TOSA FCHW permute.

    if node._layout.find("NCHW") != -1:
        conv_out_nhwc = [
            int(out_shape[0]),
            int(out_shape[2]),
            int(out_shape[3]),
            int(out_shape[1]),
        ]
    else:
        conv_out_nhwc = [int(x) for x in out_shape]
    output_type = _ranked_type(conv_out_nhwc, sb)

    res = ttir.conv2d(
        output_type,
        input_val,
        weight_val,
        _i32_array_attr([sy, sx], sb),
        _i32_array_attr(input_padding_tt, sb),
        _i32_array_attr([dy, dx], sb),
        _i32_attr(groups, sb),
        bias=bias_tensor,
        loc=sb.loc,
    )
    out = res.result if hasattr(res, "result") else res
    if node._layout.find("NCHW") != -1:
        out = _nhwc_to_nchw(out, sb)
    return out


def maxpool2d_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    if len(node.args) == 5:
        raise NotImplementedError("TTIR max_pool2d: 5-arg form not implemented.")
    input1 = symbol_table[(str(node.args[0]), 0)]
    kernel = node.args[1]
    stride = node.args[2]
    if len(node.args) > 3:
        pad = node.args[3]
    else:
        pad = [0, 0]
    k_h, k_w = int(kernel[0]), int(kernel[1])
    if isinstance(stride, (list, tuple)):
        s_h, s_w = int(stride[0]), int(stride[1])
    else:
        s_h = s_w = int(stride)

    if isinstance(pad, (list, tuple)):
        if len(pad) == 1:
            pt = pb = pl = pr = int(pad[0])
        elif len(pad) == 2:
            pt = pb = int(pad[0])
            pl = pr = int(pad[1])
        else:
            pt, pb, pl, pr = (int(x) for x in pad)
    else:
        pt = pb = pl = pr = int(pad)

    original_shape = list(input1.type.shape)
    if node._layout.find("NCHW") != -1:
        n, c, h, w = original_shape
        input1 = _nchw_to_nhwc(input1, sb)
        in_h, in_w = h, w
    else:
        n, in_h, in_w, c = original_shape

    tm = node.tensor_meta
    out_shape_meta = list(tm["shape"]) if isinstance(tm, dict) else list(tm.shape)
    if node._layout.find("NCHW") != -1:
        out_h, out_w = int(out_shape_meta[2]), int(out_shape_meta[3])
    else:
        out_h, out_w = int(out_shape_meta[1]), int(out_shape_meta[2])

    if not ((in_h + pt + pb - k_h) % s_h == 0):
        pad_total_h = max(
            ((out_h - 1) * s_h + k_h - in_h),
            0,
        )
        pt = pad_total_h // 2
        pb = pad_total_h - pt
    if not ((in_w + pl + pr - k_w) % s_w == 0):
        pad_total_w = max(
            ((out_w - 1) * s_w + k_w - in_w),
            0,
        )
        pl = pad_total_w // 2
        pr = pad_total_w - pl

    pad_tt = [pt, pl, pb, pr]
    out_nhwc = [n, out_h, out_w, c]
    output = _ranked_type(out_nhwc, sb)
    ceil_mode = bool(node.kwargs.get("ceil_mode", False)) if node.kwargs else False

    mp = ttir.max_pool2d(
        output,
        input1,
        _i32_array_attr([k_h, k_w], sb),
        _i32_array_attr([s_h, s_w], sb),
        _i32_array_attr([1, 1], sb),
        _i32_array_attr(pad_tt, sb),
        _bool_attr(ceil_mode, sb),
        loc=sb.loc,
    )
    out = mp.result if hasattr(mp, "result") else mp
    if node._layout.find("NCHW") != -1:
        out = _nhwc_to_nchw(out, sb)
    return out


from . import ttir_llm

# LLM entries first; CNN / LeNet entries below override on duplicate keys (e.g. ``AddOp``).
ops_registry = {
    **ttir_llm.llm_ops_registry,
    "PermuteOp": permute_op,
    "TOp": t_op,
    "TransposeOp": transpose_op,
    "Conv2dOp": conv2d_op,
    "ReluOp": relu_op,
    "MaxPool2dOp": maxpool2d_op,
    "ReshapeOp": reshape_op,
    "ViewOp": view_op,
    "AddMMOp": addmm_op,
    "AddOp": add_op,
}
