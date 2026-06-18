# ===- ttir_llm.py ------------------------------------------------------------
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
# Lowers LLM-style Buddy Graph ops to TTIR dialect ops.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import os

from ..graph.type import TensorDType
from .ttir import (
    TTIRSandbox,
    _bool_attr,
    _cast_to_sandbox_elt,
    _i32_attr,
    _ranked_type,
    _reshape_to,
    _tensor_meta_shape_dtype,
)


def _mlir_element_type_for_tensor_dtype(ctx, td, default_float_elt):
    """Same mapping as ``graph/ttir_import.py`` (kept local to avoid import cycles)."""
    from ttmlir.ir import BF16Type, F16Type, F32Type, IntegerType

    if td is None:
        return default_float_elt
    if isinstance(td, TensorDType):
        name = td.value
    else:
        name = str(td)
    if name in ("bfloat16", "bf16"):
        return BF16Type.get()
    if name in ("float16", "f16"):
        return F16Type.get()
    if name in ("float32", "f32"):
        return F32Type.get()
    if name in ("int64", "i64"):
        return IntegerType.get_signless(64, ctx)
    if name in ("int32", "i32"):
        return IntegerType.get_signless(32, ctx)
    if name in ("bool", "i1"):
        return IntegerType.get_signless(1, ctx)
    return default_float_elt


def _v(symbol_table, name):
    return symbol_table[(str(name), 0)]


def _scalar_promote(arg, peer, sb: TTIRSandbox):
    """Broadcast scalar Python values to ``peer``'s shape (element type)."""
    from ttmlir.dialects import ttir
    from ttmlir.ir import (
        DenseElementsAttr,
        FloatAttr,
        IntegerAttr,
        IntegerType,
        RankedTensorType,
    )

    sh = list(peer.type.shape)
    elt = peer.type.element_type
    rt_s = RankedTensorType.get(sh, elt)
    if isinstance(elt, IntegerType):
        attr = DenseElementsAttr.get_splat(rt_s, IntegerAttr.get(elt, int(arg)))
    else:
        attr = DenseElementsAttr.get_splat(rt_s, FloatAttr.get(elt, float(arg)))
    return ttir.constant(rt_s, attr, loc=sb.loc)


def _bin_operands(symbol_table, a0, a1, sb: TTIRSandbox):
    """Resolve two Buddy args to TTIR Values; supports one Python scalar + tensor."""

    def _get(arg, peer):
        key = (str(arg), 0)
        if key in symbol_table:
            return symbol_table[key]
        if hasattr(arg, "type"):
            return arg
        if peer is not None and isinstance(arg, (int, float, bool)):
            return _scalar_promote(arg, peer, sb)
        raise KeyError(key)

    try:
        left = _get(a0, None)
    except KeyError:
        left = None
    if left is not None:
        return left, _get(a1, left)
    right = _get(a1, None)
    return _get(a0, right), right


def _bitwise_operands(symbol_table, a0, a1, sb: TTIRSandbox):
    """Resolve bitwise / logical operands without forcing sandbox float casts."""

    def _get(arg, peer):
        key = (str(arg), 0)
        if key in symbol_table:
            return symbol_table[key]
        if hasattr(arg, "type"):
            return arg
        if peer is not None and isinstance(arg, (int, float, bool)):
            return _scalar_promote(arg, peer, sb)
        raise KeyError(key)

    try:
        left = _get(a0, None)
    except KeyError:
        left = None
    if left is not None:
        return left, _get(a1, left)
    right = _get(a1, None)
    return _get(a0, right), right


def _broadcast_to_result_shape(value, result_type, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    in_shape = [int(s) for s in value.type.shape]
    out_shape = [int(s) for s in result_type.shape]
    if in_shape == out_shape:
        return value
    rank_diff = len(out_shape) - len(in_shape)
    if rank_diff < 0:
        return value
    in_padded = [1] * rank_diff + in_shape
    if rank_diff > 0:
        reshape_type = RankedTensorType.get(in_padded, value.type.element_type)
        value = ttir.reshape(reshape_type, value, in_padded, loc=sb.loc)
        in_shape = in_padded
    bcast = []
    for ins, outs in zip(in_shape, out_shape):
        if ins == outs:
            bcast.append(1)
        elif ins == 1:
            bcast.append(outs)
        else:
            return value
    if os.environ.get("BUDDY_TTIR_MUL_BROADCAST_AS_REPEAT_OP") == "1":
        rt_same = RankedTensorType.get(out_shape, value.type.element_type)
        repeated = ttir.repeat(rt_same, value, bcast, loc=sb.loc)
        if str(value.type.element_type) != str(result_type.element_type):
            return ttir.typecast(result_type, repeated, loc=sb.loc)
        return repeated
    if os.environ.get("BUDDY_TTIR_MUL_BROADCAST_AS_REPEAT") == "1":
        cur_shape = list(in_shape)
        for dim, (ins, outs) in enumerate(zip(in_shape, out_shape)):
            if ins == 1 and outs != 1:
                if dim != len(out_shape) - 1:
                    break
                cur_shape[dim] = outs
                step_type = RankedTensorType.get(
                    cur_shape, value.type.element_type
                )
                value = ttir.repeat_interleave(
                    step_type, value, int(outs), int(dim), loc=sb.loc
                )
        if cur_shape == out_shape:
            if str(value.type.element_type) != str(result_type.element_type):
                return ttir.typecast(result_type, value, loc=sb.loc)
            return value
    rt_same = RankedTensorType.get(out_shape, value.type.element_type)
    expanded = ttir.broadcast(rt_same, value, bcast, loc=sb.loc)
    if str(value.type.element_type) != str(result_type.element_type):
        return ttir.typecast(result_type, expanded, loc=sb.loc)
    return expanded


def _compare_operands(symbol_table, a0, a1, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    a = _v(symbol_table, a0)
    b = _v(symbol_table, a1)
    if os.environ.get("BUDDY_TTIR_PRESERVE_COMPARE_TYPES") == "1":
        a_elt = a.type.element_type
        b_elt = b.type.element_type
        if str(a_elt) == str(b_elt):
            return a, b
        if isinstance(a_elt, IntegerType) and not isinstance(
            b_elt, IntegerType
        ):
            b_rt = RankedTensorType.get(
                [int(size) for size in b.type.shape], a_elt
            )
            return a, ttir.typecast(
                b_rt,
                b,
                conservative_folding=False,
                loc=sb.loc,
            )
        if isinstance(b_elt, IntegerType) and not isinstance(
            a_elt, IntegerType
        ):
            a_rt = RankedTensorType.get(
                [int(size) for size in a.type.shape], b_elt
            )
            return ttir.typecast(
                a_rt,
                a,
                conservative_folding=False,
                loc=sb.loc,
            ), b
    return _cast_to_sandbox_elt(a, sb), _cast_to_sandbox_elt(b, sb)


def _should_preserve_bool_tensor(val) -> bool:
    from ttmlir.ir import IntegerType

    et = val.type.element_type
    return (
        os.environ.get("BUDDY_TTIR_PRESERVE_BOOL_TENSORS") == "1"
        and IntegerType.isinstance(et)
        and IntegerType(et).width == 1
    )


def _ranked_from_meta(node, sb: TTIRSandbox):
    shape, dt = _tensor_meta_shape_dtype(node)
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    from ttmlir.ir import RankedTensorType

    return RankedTensorType.get(list(shape), mel)


def _maybe_i32_index(v, sb: TTIRSandbox):
    """``ttir.update_cache`` examples use ``tensor<...xi32>`` indices."""
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    et = v.type.element_type
    if IntegerType.isinstance(et) and IntegerType(et).width == 64:
        sh = [int(x) for x in v.type.shape]
        i32 = IntegerType.get_signless(32, sb.ctx)
        rt = RankedTensorType.get(sh, i32)
        return ttir.typecast(rt, v, loc=sb.loc)
    return v


def _index_tensor_to_ui32(v, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    et = v.type.element_type
    if IntegerType.isinstance(et):
        return _maybe_i32_index(v, sb)
    sh = [int(x) for x in v.type.shape]
    ui32 = IntegerType.get_unsigned(32, sb.ctx)
    rt = RankedTensorType.get(sh, ui32)
    return ttir.typecast(
        rt,
        v,
        conservative_folding=False,
        loc=sb.loc,
    )


def flash_attention_for_cpu_prefill_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F32Type, RankedTensorType

    q = _v(symbol_table, node.args[0])
    k = _v(symbol_table, node.args[1])
    vval = _v(symbol_table, node.args[2])
    attn_mask = node.kwargs.get("attn_mask")
    scale = node.kwargs.get("scale")
    mask_v = None
    if attn_mask is not None:
        mask_v = _v(symbol_table, attn_mask)

    tm = node.tensor_meta
    if isinstance(tm, dict):
        out_sh = list(tm["shape"][0])
        lse_sh = list(tm["shape"][1]) if len(tm["shape"]) > 1 else [1]
        lse_dt = (
            tm["dtype"][1]
            if isinstance(tm["dtype"], (list, tuple))
            else "float32"
        )
    else:
        out_sh = list(tm.shape)
        lse_sh = [1]
        lse_dt = TensorDType.Float32

    rt = RankedTensorType.get(out_sh, q.type.element_type)
    is_causal = mask_v is None
    scale_kw = float(scale) if scale is not None else None

    # --------------------------------------------------------------------
    # Workaround for tt-metal ``sdpa_flash_decode`` sfpi compile failure.
    #
    # When Q seq_len == 1 and a mask is provided (typical static-cache decode
    # step), ``tt-mlir``'s TTIRToTTNN converter promotes ttir.sdpa to
    # ttnn.scaled_dot_product_attention_decode, whose kernel
    # (``sdpa_flash_decode.cpp``) currently triggers the sfpi compiler bug:
    #   "cannot write sfpu vector to memory"
    # in ``_calculate_exponential_piecewise_.constprop.isra``.
    #
    # We pad Q's seq_len from 1 to TILE_HEIGHT (32) so TTIRToTTNN keeps the
    # standard ``scaled_dot_product_attention`` op instead. On a 32x32 tile
    # machine this adds essentially zero compute (both shapes occupy one
    # tile) but avoids the broken kernel entirely.
    # --------------------------------------------------------------------
    TILE = 32
    q_shape = list(q.type.shape)
    use_decode_workaround = (
        len(q_shape) == 4 and int(q_shape[-2]) == 1 and mask_v is not None
    )

    if use_decode_workaround:
        q_elt = q.type.element_type
        pad_q_shape = q_shape[:-2] + [TILE, int(q_shape[-1])]
        pad_q_ty = RankedTensorType.get(pad_q_shape, q_elt)
        bcast_q = [1, 1, TILE, 1]
        q_pad = ttir.broadcast(pad_q_ty, q, bcast_q, loc=sb.loc)

        m_shape = list(mask_v.type.shape)
        m_elt = mask_v.type.element_type
        pad_m_shape = m_shape[:-2] + [TILE, int(m_shape[-1])]
        pad_m_ty = RankedTensorType.get(pad_m_shape, m_elt)
        bcast_m = [1] * len(m_shape)
        bcast_m[-2] = TILE
        m_pad = ttir.broadcast(pad_m_ty, mask_v, bcast_m, loc=sb.loc)

        pad_out_shape = out_sh[:-2] + [TILE, int(out_sh[-1])]
        pad_out_ty = RankedTensorType.get(pad_out_shape, q_elt)
        out_pad = ttir.scaled_dot_product_attention(
            pad_out_ty,
            q_pad,
            k,
            vval,
            attention_mask=m_pad,
            is_causal=is_causal,
            scale=scale_kw,
            loc=sb.loc,
        )

        begins = [0] * len(pad_out_shape)
        ends = list(pad_out_shape)
        ends[-2] = 1
        step = [1] * len(pad_out_shape)
        out = ttir.slice_static(rt, out_pad, begins, ends, step, loc=sb.loc)
    else:
        out = ttir.scaled_dot_product_attention(
            rt,
            q,
            k,
            vval,
            attention_mask=mask_v,
            is_causal=is_causal,
            scale=scale_kw,
            loc=sb.loc,
        )
    lse_mel = _mlir_element_type_for_tensor_dtype(sb.ctx, lse_dt, F32Type.get())
    lse_ty = RankedTensorType.get(lse_sh, lse_mel)
    lse = ttir.empty(lse_ty, loc=sb.loc)
    return (out, lse)


def gqa_attention_fused_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    q = _v(symbol_table, node.args[0])
    k = _v(symbol_table, node.args[1])
    vval = _v(symbol_table, node.args[2])
    attn_mask = node.kwargs.get("attn_mask")
    scale = node.kwargs.get("scale")
    mask_v = None
    if attn_mask is not None:
        mask_v = _v(symbol_table, attn_mask)

    cur_key = (
        node.kwargs.get("cur_pos_tensor")
        or node.kwargs.get("cache_position")
        or node.kwargs.get("cur_pos")
    )
    if cur_key is None and len(node.args) > 3:
        cur_key = node.args[3]
    if cur_key is None:
        raise NotImplementedError(
            "GQAAttentionFusedOp → TTIR: set kwargs['cur_pos_tensor'] to the Buddy "
            "name of the cache-position placeholder (e.g. decode step index tensor) "
            "before lower_to_ttir()."
        )
    cur = _v(symbol_table, cur_key)

    tm = node.tensor_meta
    if isinstance(tm, dict):
        sh = tm["shape"]
        if len(sh) > 0 and isinstance(sh[0], (list, tuple)):
            out_shape = list(sh[0])
        else:
            out_shape = list(sh)
    else:
        out_shape = list(tm.shape)

    # ttir.scaled_dot_product_attention_decode expects:
    #   query: (1, B, nQueryHeads, headSize)
    #   key/value: (B, nKVHeads, maxSeqLen, headSize)
    #   cur_pos: (B,)
    #   result: (1, B, nQueryHeads, headSize)
    # but our Buddy graph hands us
    #   query / out_shape: (B, nQueryHeads, S=1, headSize)
    # Since S=1 in decode, we can reshape between the two layouts because
    # the leading "1" is just a degenerate dim with no stride implication.
    q_shape = list(q.type.shape)
    if len(q_shape) == 4 and q_shape[2] == 1 and q_shape[0] == 1:
        new_q_shape = [1, q_shape[0], q_shape[1], q_shape[3]]
        q_elt = q.type.element_type
        q_ty = RankedTensorType.get(new_q_shape, q_elt)
        q = ttir.reshape(q_ty, q, [int(x) for x in new_q_shape], loc=sb.loc)
    else:
        new_q_shape = q_shape

    if mask_v is not None and len(out_shape) == 4 and out_shape[2] == 1:
        m_shape = list(mask_v.type.shape)
        if len(m_shape) == 4 and m_shape[2] != out_shape[1]:
            new_m_shape = [m_shape[0], m_shape[1], out_shape[1], m_shape[3]]
            if new_m_shape != m_shape:
                m_ty = RankedTensorType.get(
                    new_m_shape, mask_v.type.element_type
                )
                bcast = [1] * 4
                if m_shape[2] == 1 and new_m_shape[2] != 1:
                    bcast[2] = new_m_shape[2]
                mask_v = ttir.broadcast(m_ty, mask_v, bcast, loc=sb.loc)

    new_out_shape = (
        [1, out_shape[0], out_shape[1], out_shape[3]]
        if len(out_shape) == 4 and out_shape[2] == 1 and out_shape[0] == 1
        else out_shape
    )
    rt = _ranked_type(new_out_shape, sb)
    scale_kw = float(scale) if scale is not None else None
    sdpa_out = ttir.scaled_dot_product_attention_decode(
        rt,
        q,
        k,
        vval,
        cur,
        attention_mask=mask_v,
        is_causal=False,
        scale=scale_kw,
        loc=sb.loc,
    )
    if new_out_shape != out_shape:
        out_ty = _ranked_type(out_shape, sb)
        sdpa_out = ttir.reshape(
            out_ty, sdpa_out, [int(x) for x in out_shape], loc=sb.loc
        )
    return sdpa_out


def index_put_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    if len(node.args) < 3:
        raise NotImplementedError(
            f"IndexPutOp TTIR: expected >=3 args, got {node.args!r}"
        )
    cache = _v(symbol_table, node.args[0])
    val = _v(symbol_table, node.args[2])
    spec = node.args[1]
    idx_names = [x for x in spec if x is not None]
    if len(idx_names) != 1:
        raise NotImplementedError(
            "IndexPutOp TTIR: only [None,…, index_tensor, …] with one index tensor "
            f"is supported; got {spec!r}"
        )
    update_index = _maybe_i32_index(_v(symbol_table, idx_names[0]), sb)
    out_ty = cache.type
    bo = _i32_attr(0, sb)
    return ttir.update_cache(
        out_ty, cache, val, update_index, batch_offset=bo, loc=sb.loc
    )


def fill_cache_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    if len(node.args) != 3:
        raise NotImplementedError(
            f"FillCacheOp TTIR: expected (cache, input, batch_offset), got {node.args!r}"
        )
    cache = _v(symbol_table, node.args[0])
    val = _v(symbol_table, node.args[1])
    batch_offset = int(node.args[2])
    return ttir.fill_cache(
        cache.type,
        cache,
        val,
        batch_offset=_i32_attr(batch_offset, sb),
        loc=sb.loc,
    )


def update_cache_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    if len(node.args) != 4:
        raise NotImplementedError(
            "UpdateCacheOp TTIR: expected "
            f"(cache, input, update_index, batch_offset), got {node.args!r}"
        )
    cache = _v(symbol_table, node.args[0])
    val = _v(symbol_table, node.args[1])
    update_index = _maybe_i32_index(_v(symbol_table, node.args[2]), sb)
    batch_offset = int(node.args[3])
    return ttir.update_cache(
        cache.type,
        cache,
        val,
        update_index,
        batch_offset=_i32_attr(batch_offset, sb),
        loc=sb.loc,
    )


def embedding_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    weight = _v(symbol_table, node.args[0])
    indices = _v(symbol_table, node.args[1])
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_EMBEDDING_AS_GATHER") == "1":
        out_shape = [int(size) for size in rt.shape]
        weight_shape = [int(size) for size in weight.type.shape]
        index_shape = [int(size) for size in indices.type.shape]
        flat_indices = 1
        for size in index_shape:
            flat_indices *= size

        weight_3d_shape = [1] + weight_shape
        weight_3d_rt = RankedTensorType.get(
            weight_3d_shape, weight.type.element_type
        )
        weight_3d = ttir.reshape(
            weight_3d_rt, weight, weight_3d_shape, loc=sb.loc
        )
        weight_2d_rt = RankedTensorType.get(
            weight_shape, weight.type.element_type
        )
        weight_2d = ttir.reshape(
            weight_2d_rt, weight_3d, weight_shape, loc=sb.loc
        )

        index_3d_shape = [1] + index_shape
        index_3d_rt = RankedTensorType.get(
            index_3d_shape, indices.type.element_type
        )
        index_3d = ttir.reshape(
            index_3d_rt, indices, index_3d_shape, loc=sb.loc
        )
        index_flat_rt = RankedTensorType.get(
            [flat_indices], indices.type.element_type
        )
        index_flat = ttir.reshape(
            index_flat_rt, index_3d, [flat_indices], loc=sb.loc
        )
        ui32 = IntegerType.get_unsigned(32, sb.ctx)
        index_ui32_rt = RankedTensorType.get([flat_indices], ui32)
        index_ui32 = ttir.typecast(
            index_ui32_rt,
            index_flat,
            conservative_folding=False,
            loc=sb.loc,
        )

        gather_rt = RankedTensorType.get(
            [flat_indices, weight_shape[1]],
            weight.type.element_type,
        )
        gathered = ttir.gather(
            gather_rt,
            weight_2d,
            index_ui32,
            [1],
            [0],
            [],
            [],
            [0],
            1,
            [1, weight_shape[1]],
            False,
            loc=sb.loc,
        )
        return ttir.reshape(rt, gathered, out_shape, loc=sb.loc)
    return ttir.embedding(rt, indices, weight, loc=sb.loc)


def matmul_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a = _v(symbol_table, node.args[0])
    b = _v(symbol_table, node.args[1])
    out_shape, _ = _tensor_meta_shape_dtype(node)
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_MATMUL_AS_DOT_GENERAL") == "1":
        a_rank = len(a.type.shape)
        b_rank = len(b.type.shape)
        if a_rank == 2 and b_rank == 2:
            return ttir.dot_general(
                rt,
                a,
                b,
                [],
                [1],
                [],
                [0],
                loc=sb.loc,
            )
        if a_rank == 3 and b_rank == 3:
            return ttir.dot_general(
                rt,
                a,
                b,
                [0],
                [2],
                [0],
                [1],
                loc=sb.loc,
            )
    return ttir.matmul(rt, a, b, loc=sb.loc)


def batch_matmul_op(node, symbol_table, sb: TTIRSandbox):
    return matmul_op(node, symbol_table, sb)


def mul_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_EXPLICIT_MUL_BROADCAST") == "1":
        a = _broadcast_to_result_shape(a, rt, sb)
        b = _broadcast_to_result_shape(b, rt, sb)
    return ttir.multiply(rt, a, b, loc=sb.loc)


def div_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.div(rt, a, b, loc=sb.loc)


def sub_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.subtract(rt, a, b, loc=sb.loc)


def rsub_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.subtract(rt, b, a, loc=sb.loc)


def silu_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F32Type, RankedTensorType

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_SILU_AS_SIGMOID_MUL") == "1":
        f32_rt = RankedTensorType.get(list(x.type.shape), F32Type.get())
        x_f32 = ttir.typecast(f32_rt, x, loc=sb.loc)
        sig = ttir.sigmoid(f32_rt, x_f32, loc=sb.loc)
        prod = ttir.multiply(f32_rt, x_f32, sig, loc=sb.loc)
        return ttir.typecast(rt, prod, loc=sb.loc)
    return ttir.silu(rt, x, loc=sb.loc)


def gelu_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.gelu(rt, x, loc=sb.loc)


def pow_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bin_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.pow(rt, a, b, loc=sb.loc)


def rsqrt_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_RSQRT_AS_SQRT_RECIPROCAL") == "1":
        sqrted = ttir.sqrt(rt, x, loc=sb.loc)
        return ttir.reciprocal(rt, sqrted, loc=sb.loc)
    return ttir.rsqrt(rt, x, loc=sb.loc)


def sqrt_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return ttir.sqrt(rt, x, loc=sb.loc)


def _unary_ttir(op_name: str, node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    return getattr(ttir, op_name)(rt, x, loc=sb.loc)


def cos_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("cos", node, symbol_table, sb)


def sin_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("sin", node, symbol_table, sb)


def tan_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("tan", node, symbol_table, sb)


def exp_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("exp", node, symbol_table, sb)


def log_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("log", node, symbol_table, sb)


def neg_op(node, symbol_table, sb: TTIRSandbox):
    return _unary_ttir("neg", node, symbol_table, sb)


def mean_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    x = _v(symbol_table, node.args[0])
    dims = node.args[1]
    keepdim = bool(node.args[2]) if len(node.args) > 2 else False
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    dim_list: list[int] = []
    rank = len(x.type.shape)
    for d in dims:
        di = int(d)
        if di < 0:
            di += rank
        dim_list.append(di)
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_MEAN_AS_SUM") == "1":
        dim_set = set(dim_list)
        reduced_shape = [
            int(size)
            for idx, size in enumerate(x.type.shape)
            if idx not in dim_set
        ]
        reduce_elems = 1
        for idx in dim_set:
            reduce_elems *= int(x.type.shape[idx])
        sum_rt = RankedTensorType.get(reduced_shape, x.type.element_type)
        summed = ttir.sum(
            sum_rt,
            x,
            _bool_attr(False, sb),
            dim_arg=dim_list,
            loc=sb.loc,
        )
        scale = _scalar_promote(1.0 / float(reduce_elems), summed, sb)
        scaled = ttir.multiply(sum_rt, summed, scale, loc=sb.loc)
        if keepdim:
            return ttir.reshape(
                rt,
                scaled,
                [int(size) for size in rt.shape],
                loc=sb.loc,
            )
        return scaled
    return ttir.mean(
        rt,
        x,
        _bool_attr(keepdim, sb),
        dim_arg=dim_list,
        loc=sb.loc,
    )


def argmax_op(node, symbol_table, sb: TTIRSandbox):
    """Lower ``aten.argmax.default`` to ``ttir.argmax``.

    ``ttir.argmax`` natively emits ``i32`` indices; if the FX node's expected
    output dtype is ``i64`` (PyTorch default), append a ``ttir.typecast`` to
    widen the result. Without ``dim_arg`` the op reduces over all dims.
    """
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    x = _v(symbol_table, node.args[0])
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = bool(node.args[2]) if len(node.args) > 2 else False

    in_shape = list(x.type.shape)
    rank = len(in_shape)
    dim_arg = None
    if dim is not None:
        di = int(dim)
        if di < 0:
            di += rank
        dim_arg = [di]

    if dim_arg is None:
        out_shape: list[int] = [1] * rank if keepdim else []
    else:
        d = dim_arg[0]
        if keepdim:
            out_shape = list(in_shape)
            out_shape[d] = 1
        else:
            out_shape = in_shape[:d] + in_shape[d + 1 :]

    i32 = IntegerType.get_signless(32, sb.ctx)
    i32_rt = RankedTensorType.get(out_shape, i32)
    res = ttir.argmax(
        i32_rt,
        x,
        _bool_attr(keepdim, sb),
        dim_arg=dim_arg,
        loc=sb.loc,
    )

    expected_rt = _ranked_from_meta(node, sb)
    expected_elt = expected_rt.element_type
    expected_shape = list(expected_rt.shape)

    if (
        IntegerType.isinstance(expected_elt)
        and IntegerType(expected_elt).width != 32
    ):
        cast_rt = RankedTensorType.get(out_shape, expected_elt)
        res = ttir.typecast(
            cast_rt, res, conservative_folding=False, loc=sb.loc
        )

    if expected_shape != out_shape:
        res = _reshape_to(res, expected_shape, node, sb)
    return res


def softmax_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F64Type, IntegerType, RankedTensorType

    x = _v(symbol_table, node.args[0])
    dim = int(node.args[1])
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_SOFTMAX_AS_EXPLICIT") == "1":
        in_shape = [int(size) for size in x.type.shape]
        rank = len(in_shape)
        if dim < 0:
            dim += rank
        red_shape = in_shape[:dim] + in_shape[dim + 1 :]
        keep_shape = list(in_shape)
        keep_shape[dim] = 1
        red_rt = RankedTensorType.get(red_shape, x.type.element_type)
        keep_rt = RankedTensorType.get(keep_shape, x.type.element_type)
        in_rt = RankedTensorType.get(in_shape, x.type.element_type)
        bcast_dims = [
            int(in_size // keep_size) if int(keep_size) != int(in_size) else 1
            for in_size, keep_size in zip(in_shape, keep_shape)
        ]

        safe_mask = None
        if os.environ.get("BUDDY_TTIR_SAFE_SOFTMAX_MASK") == "1":
            i1 = IntegerType.get_signless(1, sb.ctx)
            in_bool_rt = RankedTensorType.get(in_shape, i1)
            red_bool_rt = RankedTensorType.get(red_shape, i1)
            keep_bool_rt = RankedTensorType.get(keep_shape, i1)
            f64_rt = RankedTensorType.get(in_shape, F64Type.get())
            x_f64 = ttir.typecast(
                f64_rt,
                x,
                conservative_folding=False,
                loc=sb.loc,
            )
            neg_inf = _scalar_promote(float("-inf"), x_f64, sb)
            is_inf = ttir.eq(in_bool_rt, x_f64, neg_inf, loc=sb.loc)
            is_not_inf = ttir.logical_not(in_bool_rt, is_inf, loc=sb.loc)
            any_not_inf = ttir.reduce_or(
                red_bool_rt,
                is_not_inf,
                _bool_attr(False, sb),
                dim_arg=[dim],
                loc=sb.loc,
            )
            any_not_inf_keep = ttir.reshape(
                keep_bool_rt,
                any_not_inf,
                keep_shape,
                loc=sb.loc,
            )
            all_inf_keep = ttir.logical_not(
                keep_bool_rt,
                any_not_inf_keep,
                loc=sb.loc,
            )
            all_inf_red = ttir.reshape(
                red_bool_rt,
                all_inf_keep,
                red_shape,
                loc=sb.loc,
            )
            all_inf_keep = ttir.reshape(
                keep_bool_rt,
                all_inf_red,
                keep_shape,
                loc=sb.loc,
            )
            safe_mask = ttir.broadcast(
                in_bool_rt,
                all_inf_keep,
                bcast_dims,
                loc=sb.loc,
            )

        max_red = ttir.max(
            red_rt,
            x,
            _bool_attr(False, sb),
            dim_arg=[dim],
            loc=sb.loc,
        )
        max_keep = ttir.reshape(keep_rt, max_red, keep_shape, loc=sb.loc)
        max_bcast = ttir.broadcast(
            in_rt,
            max_keep,
            bcast_dims,
            loc=sb.loc,
        )
        shifted = ttir.subtract(in_rt, x, max_bcast, loc=sb.loc)
        expv = ttir.exp(in_rt, shifted, loc=sb.loc)
        sum_red = ttir.sum(
            red_rt,
            expv,
            _bool_attr(False, sb),
            dim_arg=[dim],
            loc=sb.loc,
        )
        sum_keep = ttir.reshape(keep_rt, sum_red, keep_shape, loc=sb.loc)
        sum_bcast = ttir.broadcast(
            in_rt,
            sum_keep,
            bcast_dims,
            loc=sb.loc,
        )
        probs = ttir.div(rt, expv, sum_bcast, loc=sb.loc)
        if safe_mask is not None:
            zero = _scalar_promote(0.0, probs, sb)
            return ttir.where(rt, safe_mask, zero, probs, loc=sb.loc)
        return probs
    return ttir.softmax(rt, x, dim, numeric_stable=False, loc=sb.loc)


def concat_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    inputs_names = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    tensors = [_v(symbol_table, n) for n in inputs_names]

    def _numel(ty):
        n = 1
        for d in ty.shape:
            n *= int(d)
        return n

    # ``torch.cat`` can include empty tensors; TTIR rejects invalid rank/shape mixes.
    non_empty = [t for t in tensors if _numel(t.type) > 0]
    out_shape, _ = _tensor_meta_shape_dtype(node)
    out_shape = [int(s) for s in out_shape]
    if dim < 0:
        dim += len(out_shape)
    if not non_empty:
        raise NotImplementedError(
            "concat TTIR: all operands are empty tensors."
        )
    if len(non_empty) == 1:
        only = non_empty[0]
        if [int(d) for d in only.type.shape] == out_shape:
            return only
        return _reshape_to(only, out_shape, node, sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.concat(rt, non_empty, dim, loc=sb.loc)


def where_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    cond = _v(symbol_table, node.args[0])
    a = _v(symbol_table, node.args[1])
    b = _v(symbol_table, node.args[2])
    rt = _ranked_from_meta(node, sb)
    return ttir.where(rt, cond, a, b, loc=sb.loc)


def logical_and_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _bitwise_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    a = _broadcast_to_result_shape(a, rt, sb)
    b = _broadcast_to_result_shape(b, rt, sb)
    return ttir.logical_and(rt, a, b, loc=sb.loc)


def bitwise_and_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType

    a, b = _bitwise_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    a = _broadcast_to_result_shape(a, rt, sb)
    b = _broadcast_to_result_shape(b, rt, sb)
    if (
        IntegerType.isinstance(rt.element_type)
        and IntegerType(rt.element_type).width == 1
    ):
        return ttir.logical_and(rt, a, b, loc=sb.loc)
    return ttir.bitwise_and(rt, a, b, loc=sb.loc)


def bitwise_and_scalar_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType

    a, b = _bitwise_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    a = _broadcast_to_result_shape(a, rt, sb)
    b = _broadcast_to_result_shape(b, rt, sb)
    if (
        IntegerType.isinstance(rt.element_type)
        and IntegerType(rt.element_type).width == 1
    ):
        return ttir.logical_and(rt, a, b, loc=sb.loc)
    return ttir.bitwise_and(rt, a, b, loc=sb.loc)


def index_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    if len(node.args) != 2:
        raise NotImplementedError(
            f"IndexOp TTIR: expected (input, indices), got {node.args!r}"
        )

    input_val = _v(symbol_table, node.args[0])
    index_spec = node.args[1]
    if not isinstance(index_spec, (list, tuple)):
        raise NotImplementedError(
            "IndexOp TTIR: expected a list/tuple of indices, "
            f"got {index_spec!r}"
        )

    tensor_indices = []
    tensor_positions = []
    for pos, idx in enumerate(index_spec):
        if idx is None:
            continue
        tensor_indices.append(_v(symbol_table, idx))
        tensor_positions.append(pos)

    out_ty = _ranked_from_meta(node, sb)
    out_shape = [int(d) for d in out_ty.shape]

    if (
        os.environ.get("BUDDY_TTIR_BYPASS_BOOL_IDENTITY_INDEX") == "1"
        and all(idx is not None for idx in index_spec)
        and tensor_indices
    ):
        from ttmlir.ir import IntegerType

        input_shape = [int(d) for d in input_val.type.shape]
        input_elt = input_val.type.element_type
        out_elt = out_ty.element_type
        if (
            input_shape == out_shape
            and len(index_spec) == len(input_shape)
            and IntegerType.isinstance(input_elt)
            and IntegerType(input_elt).width == 1
            and str(input_elt) == str(out_elt)
        ):
            return input_val

    # Fast path: one tensor index and the rest are full slices. This covers the
    # common Llama cache-position / prompt-mask style advanced indexing pattern.
    if (
        len(tensor_indices) == 1
        and len(tensor_positions) == 1
        and any(idx is None for idx in index_spec)
    ):
        indexed_dim = int(tensor_positions[0])
        idx = _index_tensor_to_ui32(tensor_indices[0], sb)
        idx_ty = RankedTensorType.get(out_shape, idx.type.element_type)
        idx = _broadcast_to_result_shape(idx, idx_ty, sb)
        return ttir.gather(out_ty, input_val, idx, indexed_dim, loc=sb.loc)

    if all(idx is not None for idx in index_spec) and tensor_indices:
        indexed_rank = len(index_spec)
        if indexed_rank > len(input_val.type.shape):
            raise NotImplementedError(
                "IndexOp TTIR: more tensor indices than input rank are not "
                f"supported; got {index_spec!r}"
            )

        input_shape = [int(d) for d in input_val.type.shape]
        remaining_shape = input_shape[indexed_rank:]
        broadcast_rank = len(out_shape) - len(remaining_shape)
        if broadcast_rank < 0:
            raise NotImplementedError(
                "IndexOp TTIR: output rank is smaller than the preserved "
                f"tail dimensions; got {index_spec!r}"
            )
        broadcast_shape = out_shape[:broadcast_rank]

        if (
            len(input_shape) == 2
            and indexed_rank == 2
            and not remaining_shape
            and out_shape == [input_shape[0], 1, 1, input_shape[1]]
        ):
            from ttmlir.ir import IntegerType

            input_elt = input_val.type.element_type
            if (
                IntegerType.isinstance(input_elt)
                and IntegerType(input_elt).width == 1
            ):
                return ttir.reshape(out_ty, input_val, out_shape, loc=sb.loc)

        indexed_volume = 1
        for dim in input_shape[:indexed_rank]:
            indexed_volume *= int(dim)
        remaining_volume = 1
        for dim in remaining_shape:
            remaining_volume *= int(dim)

        linear = None
        for dim, idx in enumerate(tensor_indices):
            idx = _index_tensor_to_ui32(idx, sb)
            idx_rt = RankedTensorType.get(
                broadcast_shape, idx.type.element_type
            )
            idx = _broadcast_to_result_shape(idx, idx_rt, sb)

            stride = 1
            for tail_dim in input_shape[dim + 1 : indexed_rank]:
                stride *= int(tail_dim)
            stride_val = _scalar_promote(stride, idx, sb)
            term = ttir.multiply(idx_rt, idx, stride_val, loc=sb.loc)
            linear = (
                term
                if linear is None
                else ttir.add(idx_rt, linear, term, loc=sb.loc)
            )

        full_idx_ty = RankedTensorType.get(out_shape, linear.type.element_type)
        linear = _broadcast_to_result_shape(linear, full_idx_ty, sb)

        flat_input_shape = (
            [indexed_volume]
            + [1] * max(0, broadcast_rank - 1)
            + remaining_shape
        )
        flat_input_ty = RankedTensorType.get(
            flat_input_shape,
            input_val.type.element_type,
        )
        flat_input = ttir.reshape(
            flat_input_ty,
            input_val,
            flat_input_shape,
            loc=sb.loc,
        )
        gather_rt = RankedTensorType.get(out_shape, input_val.type.element_type)
        return ttir.gather(gather_rt, flat_input, linear, 0, loc=sb.loc)

    raise NotImplementedError(
        "IndexOp TTIR: only one-tensor index with the remaining dimensions "
        f"left as None is currently supported; got {index_spec!r}"
    )


def le_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _compare_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.le(rt, a, b, loc=sb.loc)


def lt_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _compare_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.lt(rt, a, b, loc=sb.loc)


def eq_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _compare_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.eq(rt, a, b, loc=sb.loc)


def ne_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    a, b = _compare_operands(symbol_table, node.args[0], node.args[1], sb)
    rt = _ranked_from_meta(node, sb)
    return ttir.ne(rt, a, b, loc=sb.loc)


def unsqueeze_op(node, symbol_table, sb: TTIRSandbox):
    x = _v(symbol_table, node.args[0])
    out_shape, _ = _tensor_meta_shape_dtype(node)
    return _reshape_to(x, out_shape, node, sb)


def squeeze_op(node, symbol_table, sb: TTIRSandbox):
    x = _v(symbol_table, node.args[0])
    out_shape, _ = _tensor_meta_shape_dtype(node)
    return _reshape_to(x, out_shape, node, sb)


def slice_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    x = _v(symbol_table, node.args[0])
    dim = int(node.args[1])
    start_idx = int(node.args[2])
    end_idx = int(node.args[3])
    sizes = [int(s) for s in x.type.shape]
    rank = len(sizes)
    if dim < 0:
        dim += rank
    if start_idx < 0:
        start_idx += sizes[dim]
    if end_idx < 0:
        end_idx += sizes[dim]
    begins = [0] * rank
    ends = list(sizes)
    step = [1] * rank
    begins[dim] = max(0, min(start_idx, sizes[dim]))
    ends[dim] = max(begins[dim], min(end_idx, sizes[dim]))
    out_shape, out_dtype = _tensor_meta_shape_dtype(node)
    out_elt = _mlir_element_type_for_tensor_dtype(
        sb.ctx, out_dtype, sb.elt_type
    )
    if IntegerType.isinstance(out_elt):
        if str(x.type.element_type) != str(out_elt):
            cast_rt = RankedTensorType.get(list(x.type.shape), out_elt)
            x = ttir.typecast(cast_rt, x, loc=sb.loc)
        rt = RankedTensorType.get([int(s) for s in out_shape], out_elt)
    else:
        x = _cast_to_sandbox_elt(x, sb)
        rt = _ranked_type(out_shape, sb)
    return ttir.slice_static(rt, x, begins, ends, step, loc=sb.loc)


def select_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import IntegerType, RankedTensorType

    x = _v(symbol_table, node.args[0])
    dim = int(node.args[1])
    index = int(node.args[2])
    sizes = [int(s) for s in x.type.shape]
    rank = len(sizes)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(f"SelectOp dim {dim} out of range for rank {rank}")
    if index < 0:
        index += sizes[dim]
    index = max(0, min(index, sizes[dim] - 1))

    begins = [0] * rank
    ends = list(sizes)
    step = [1] * rank
    begins[dim] = index
    ends[dim] = index + 1

    out_shape, out_dtype = _tensor_meta_shape_dtype(node)
    out_elt = _mlir_element_type_for_tensor_dtype(
        sb.ctx, out_dtype, sb.elt_type
    )
    slice_shape = list(sizes)
    slice_shape[dim] = 1
    if IntegerType.isinstance(out_elt):
        if str(x.type.element_type) != str(out_elt):
            cast_rt = RankedTensorType.get(list(x.type.shape), out_elt)
            x = ttir.typecast(cast_rt, x, loc=sb.loc)
        slice_rt = RankedTensorType.get(slice_shape, out_elt)
        rt = RankedTensorType.get([int(s) for s in out_shape], out_elt)
    else:
        x = _cast_to_sandbox_elt(x, sb)
        slice_rt = _ranked_type(slice_shape, sb)
        rt = _ranked_type(out_shape, sb)
    sliced = ttir.slice_static(slice_rt, x, begins, ends, step, loc=sb.loc)
    return ttir.reshape(rt, sliced, [int(s) for s in out_shape], loc=sb.loc)


def expand_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    x = _v(symbol_table, node.args[0])
    preserve_bool = _should_preserve_bool_tensor(x)
    preserve_shape_types = (
        os.environ.get("BUDDY_TTIR_PRESERVE_SHAPE_TYPES") == "1"
    )
    if not preserve_bool and not preserve_shape_types:
        x = _cast_to_sandbox_elt(x, sb)
    out_shape, _ = _tensor_meta_shape_dtype(node)
    out_shape = [int(s) for s in out_shape]
    in_shape = [int(s) for s in x.type.shape]

    def _numel(sh):
        n = 1
        for d in sh:
            n *= int(d)
        return n

    # Pure rank / layout change with same elements (e.g. [8,8] → [1,8,8]).
    if _numel(in_shape) == _numel(out_shape) and tuple(in_shape) != tuple(
        out_shape
    ):
        return _reshape_to(x, out_shape, node, sb)

    rank_diff = len(out_shape) - len(in_shape)
    if rank_diff < 0:
        raise NotImplementedError(
            f"expand TTIR: output rank < input rank {in_shape} → {out_shape}"
        )
    in_padded = [1] * rank_diff + in_shape
    bcast = []
    for ins, outs in zip(in_padded, out_shape):
        if ins == outs:
            bcast.append(1)
        elif ins == 1:
            bcast.append(outs)
        else:
            raise NotImplementedError(
                f"expand TTIR: cannot broadcast {in_shape} → {out_shape}"
            )
    rt = (
        _ranked_from_meta(node, sb)
        if preserve_bool or preserve_shape_types
        else _ranked_type(out_shape, sb)
    )
    in_elt = x.type.element_type
    out_elt = rt.element_type
    # ttmlir-opt can assert when ``broadcast`` mixes dtypes (e.g. f32 in → bf16 out).
    # Broadcast in the input's element type, then typecast to the graph output type.
    if str(in_elt) != str(out_elt):
        from ttmlir.ir import RankedTensorType

        rt_same = RankedTensorType.get(list(out_shape), in_elt)
        expanded = ttir.broadcast(rt_same, x, bcast, loc=sb.loc)
        return ttir.typecast(rt, expanded, loc=sb.loc)
    return ttir.broadcast(rt, x, bcast, loc=sb.loc)


def lift_fresh_copy_op(node, symbol_table, sb: TTIRSandbox):
    """``lift_fresh_copy`` → same as identity-ish clone (multiply by one)."""
    return clone_op(node, symbol_table, sb)


def alias_op(node, symbol_table, sb: TTIRSandbox):
    """``aten.alias`` / tensor alias → identity path (multiply by 1)."""
    return clone_op(node, symbol_table, sb)


def clone_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    x = _v(symbol_table, node.args[0])
    rt = x.type
    if os.environ.get("BUDDY_TTIR_CLONE_AS_TO_LAYOUT") == "1":
        out = ttir.empty(rt, loc=sb.loc)
        return ttir.to_layout([rt], x, out, loc=sb.loc)
    elt = rt.element_type
    one = DenseElementsAttr.get_splat(rt, FloatAttr.get(elt, 1.0))
    ones_t = ttir.constant(rt, one, loc=sb.loc)
    return ttir.multiply(rt, x, ones_t, loc=sb.loc)


def convert_element_type_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import BF16Type, F32Type

    x = _v(symbol_table, node.args[0])
    rt = _ranked_from_meta(node, sb)
    if os.environ.get("BUDDY_TTIR_SKIP_RMSNORM_BF16_SCALAR_CAST") == "1":
        shape = [int(size) for size in x.type.shape]
        if (
            shape
            and shape[-1] == 1
            and isinstance(x.type.element_type, F32Type)
            and isinstance(rt.element_type, BF16Type)
        ):
            return x
    return ttir.typecast(rt, x, loc=sb.loc)


def to_copy_op(node, symbol_table, sb: TTIRSandbox):
    """``aten._to_copy`` / dtype-device copies → ``ttir.typecast`` when dtype changes."""
    return convert_element_type_op(node, symbol_table, sb)


def tensor_constant_op(node, symbol_table, sb: TTIRSandbox):
    import numpy as np
    import torch
    from ttmlir.dialects import ttir
    from ttmlir.ir import BF16Type, DenseElementsAttr

    rt = _ranked_from_meta(node, sb)
    raw = node.args[0]
    arr = np.asarray(raw)
    elt = rt.element_type
    if isinstance(elt, BF16Type) and arr.dtype != np.uint16:
        # Frontend may supply float32 NumPy after folding bfloat16 tensors.
        t = torch.from_numpy(arr.astype(np.float32, copy=False)).to(
            torch.bfloat16
        )
        arr = t.view(torch.uint16).numpy()
    try:
        attr = DenseElementsAttr.get(arr, type=rt)
    except (TypeError, ValueError):
        attr = DenseElementsAttr.get(np.asarray(arr), type=rt)
    return ttir.constant(rt, attr, loc=sb.loc)


def full_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    if isinstance(node.args[0], (list, tuple)):
        shape_arg = node.args[0]
        fill = node.args[1]
    else:
        fill = node.args[0]
        shape_arg = node.args[1]
    if isinstance(shape_arg, (list, tuple)):
        sh = [int(x) for x in shape_arg]
    else:
        sh = [int(shape_arg)]
    if os.environ.get("BUDDY_TTIR_PRESERVE_FULL_TYPES") == "1":
        rt = _ranked_from_meta(node, sb)
    else:
        rt = _ranked_type(sh, sb)
    attr = DenseElementsAttr.get_splat(
        rt, FloatAttr.get(rt.element_type, float(fill))
    )
    return ttir.constant(rt, attr, loc=sb.loc)


def ones_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    sh = [int(x) for x in node.args[0]]
    rt = _ranked_type(sh, sb)
    return ttir.ones(rt, sh, loc=sb.loc)


def zeros_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    sh = [int(x) for x in node.args[0]]
    rt = _ranked_type(sh, sb)
    return ttir.zeros(rt, sh, loc=sb.loc)


def scalar_tensor_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import DenseElementsAttr, FloatAttr

    fill = node.args[0]
    rt = _ranked_from_meta(node, sb)
    attr = DenseElementsAttr.get_splat(
        rt, FloatAttr.get(rt.element_type, float(fill))
    )
    return ttir.constant(rt, attr, loc=sb.loc)


def iota_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir

    count = int(node.args[0])
    start = int(node.kwargs.get("start", 0))
    step = int(node.kwargs.get("step", 1))
    out_shape, dt = _tensor_meta_shape_dtype(node)
    if len(out_shape) != 1:
        raise NotImplementedError("IotaOp TTIR: only 1-D output is supported.")
    end = start + count * step
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    from ttmlir.ir import RankedTensorType

    rt = RankedTensorType.get(out_shape, mel)
    return ttir.arange(rt, start, end, step, 0, loc=sb.loc)


def arange_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import RankedTensorType

    out_shape, dt = _tensor_meta_shape_dtype(node)
    if len(out_shape) != 1:
        raise NotImplementedError("ArangeOp TTIR: only 1-D is supported.")
    n = int(out_shape[0])
    start = int(node.kwargs.get("start", 0))
    step = int(node.kwargs.get("step", 1))
    if "end" in node.kwargs:
        end_excl = int(node.kwargs["end"])
    else:
        end_excl = start + n * step
    mel = _mlir_element_type_for_tensor_dtype(sb.ctx, dt, sb.elt_type)
    rt = RankedTensorType.get(out_shape, mel)
    return ttir.arange(rt, start, end_excl, step, 0, loc=sb.loc)


def arange_start_step_op(node, symbol_table, sb: TTIRSandbox):
    return arange_op(node, symbol_table, sb)


def native_layer_norm_op(node, symbol_table, sb: TTIRSandbox):
    from ttmlir.dialects import ttir
    from ttmlir.ir import F32Type, RankedTensorType

    x = _v(symbol_table, node.args[0])
    normalized_shape = [int(x) for x in node.args[1]]
    w = _v(symbol_table, node.args[2]) if node.args[2] is not None else None
    b = _v(symbol_table, node.args[3]) if node.args[3] is not None else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5
    rt = _ranked_from_meta(node, sb)
    y = ttir.layer_norm(
        rt,
        x,
        normalized_shape,
        weight=w,
        bias=b,
        epsilon=eps,
        loc=sb.loc,
    )
    tm = node.tensor_meta
    if (
        isinstance(tm, dict)
        and isinstance(tm.get("shape"), (list, tuple))
        and len(tm["shape"]) == 3
    ):
        shp = tm["shape"]
        dtp = tm["dtype"]
        msh = list(shp[1])
        rsh = list(shp[2])
        md0 = dtp[1] if isinstance(dtp, (list, tuple)) else TensorDType.Float32
        rd0 = dtp[2] if isinstance(dtp, (list, tuple)) else TensorDType.Float32
        mel_m = _mlir_element_type_for_tensor_dtype(sb.ctx, md0, F32Type.get())
        mel_r = _mlir_element_type_for_tensor_dtype(sb.ctx, rd0, F32Type.get())
        mean_e = ttir.empty(RankedTensorType.get(msh, mel_m), loc=sb.loc)
        rstd_e = ttir.empty(RankedTensorType.get(rsh, mel_r), loc=sb.loc)
        return (y, mean_e, rstd_e)
    return y


llm_ops_registry = {
    "FlashAttentionForCpuPrefillOp": flash_attention_for_cpu_prefill_op,
    "GQAAttentionFusedOp": gqa_attention_fused_op,
    "IndexPutOp": index_put_op,
    "FillCacheOp": fill_cache_op,
    "UpdateCacheOp": update_cache_op,
    "EmbeddingOp": embedding_op,
    "MatmulOp": matmul_op,
    "BatchMatmulOp": batch_matmul_op,
    "MulOp": mul_op,
    "DivOp": div_op,
    "SubOp": sub_op,
    "RsubOp": rsub_op,
    "SiluOp": silu_op,
    "GeluOp": gelu_op,
    "PowOp": pow_op,
    "RsqrtOp": rsqrt_op,
    "SqrtOp": sqrt_op,
    "CosOp": cos_op,
    "SinOp": sin_op,
    "TanOp": tan_op,
    "ExpOp": exp_op,
    "LogOp": log_op,
    "NegOp": neg_op,
    "MeanOp": mean_op,
    "ArgMaxOp": argmax_op,
    "SoftmaxOp": softmax_op,
    "CatOp": concat_op,
    "WhereOp": where_op,
    "LogicalAndOp": logical_and_op,
    "BitwiseAndTensorOp": bitwise_and_op,
    "BitwiseAndScalarOp": bitwise_and_scalar_op,
    "IndexOp": index_op,
    "LeTensorOp": le_tensor_op,
    "LtTensorOp": lt_tensor_op,
    "EqTensorOp": eq_tensor_op,
    "NeTensorOp": ne_tensor_op,
    "UnsqueezeOp": unsqueeze_op,
    "SqueezeOp": squeeze_op,
    "SliceOp": slice_op,
    "SelectOp": select_op,
    "ExpandOp": expand_op,
    "CloneOp": clone_op,
    "LiftFreshCopyOp": lift_fresh_copy_op,
    "AliasOp": alias_op,
    "ConvertElementTypeOp": convert_element_type_op,
    "ToCopyOp": to_copy_op,
    "TensorConstantOp": tensor_constant_op,
    "FullOp": full_op,
    "OnesOp": ones_op,
    "ZerosOp": zeros_op,
    "ScalarTensorOp": scalar_tensor_op,
    "IotaOp": iota_op,
    "ArangeOp": arange_op,
    "ArangeStartStepOp": arange_start_step_op,
    "NativeLayerNormOp": native_layer_norm_op,
}
