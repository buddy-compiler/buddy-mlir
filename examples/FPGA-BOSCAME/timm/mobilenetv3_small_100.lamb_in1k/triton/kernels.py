"""MobileNetV3 Triton kernels with static NCHW specializations."""

import triton
import triton.language as tl


@triton.jit
def residual_add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    y = tl.load(Y + index, valid, other=0.0)
    tl.store(Out + index, x + y, valid)


@triton.jit
def relu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    tl.store(Out + index, tl.maximum(x, 0.0), valid)


@triton.jit
def hardsigmoid(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    shifted = x + 3.0
    lower = tl.maximum(shifted, 0.0, propagate_nan=tl.PropagateNan.ALL)
    clamped = tl.minimum(lower, 6.0, propagate_nan=tl.PropagateNan.ALL)
    # Round the FP32 division, rather than multiplying by a rounded reciprocal.
    tl.store(Out + index, tl.div_rn(clamped, 6.0), valid)


@triton.jit
def hardswish(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    shifted = x + 3.0
    lower = tl.maximum(shifted, 0.0, propagate_nan=tl.PropagateNan.ALL)
    clamped = tl.minimum(lower, 6.0, propagate_nan=tl.PropagateNan.ALL)
    # Keep aten's FP32 multiply-then-divide order, including overflow behavior.
    tl.store(Out + index, tl.div_rn(x * clamped, 6.0), valid)


@triton.jit
def se_mul(X, Scale, Out, COUNT: tl.constexpr, SPATIAL: tl.constexpr,
           BLOCK: tl.constexpr):
    channel = tl.program_id(1)
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_channel = channel < COUNT // SPATIAL
    valid = valid_channel & (spatial < SPATIAL)
    index = channel * SPATIAL + spatial
    x = tl.load(X + index, valid, other=0.0)
    scale = tl.load(Scale + channel, valid_channel, other=0.0)
    tl.store(Out + index, x * scale, valid)


@triton.jit
def mean_hw(X, Out, COUNT: tl.constexpr, SPATIAL: tl.constexpr,
            BLOCK: tl.constexpr):
    channel = tl.program_id(0)
    spatial = tl.arange(0, BLOCK)
    valid_channel = channel < COUNT // SPATIAL
    x = tl.load(X + channel * SPATIAL + spatial,
                valid_channel & (spatial < SPATIAL), other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(Out + channel, tl.div_rn(total, SPATIAL), valid_channel)


@triton.jit
def linear(X, Weight, Bias, Out, M: tl.constexpr, N: tl.constexpr,
           K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    column_start = tl.program_id(1) * BN
    store_limit = N
    if N >= BN and N % BN != 0:
        # Keep every loaded weight tile entirely inside physical [N,K].
        # The final tile overlaps the preceding tile's reads; stores remain
        # disjoint: preceding tiles end at N-BN, final tile owns [N-BN,N).
        # This avoids materializing a padded/transposed B tile for the tail.
        column_start = tl.minimum(column_start, N - BN)
        store_limit = tl.where(tl.program_id(1) == tl.cdiv(N, BN) - 1, N, N - BN)
    cols = column_start + tl.arange(0, BN)
    k = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        if M % BM == 0 and K % BK == 0:
            a = tl.load(X + rows[:, None] * K + kk[None, :])
        else:
            a = tl.load(X + rows[:, None] * K + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        # Physical row-major [N,K]; logical dot operand [K,BN]. No packing
        # or transposed weight tensor is supplied by the caller.
        if N >= BN and K % BK == 0:
            b = tl.load(Weight + cols[None, :] * K + kk[:, None])
        else:
            b = tl.load(Weight + cols[None, :] * K + kk[:, None],
                        (cols[None, :] < N) & (kk[:, None] < K), other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    bias = tl.load(Bias + cols, cols < N, other=0.0)
    tl.store(Out + rows[:, None] * N + cols[None, :],
             accumulator + bias[None, :],
             (rows[:, None] < M) & (cols[None, :] < store_limit))


@triton.jit
def depthwise_conv2d(X, Weight, Bias, Out, C: tl.constexpr,
                     H: tl.constexpr, W: tl.constexpr,
                     KH: tl.constexpr, KW: tl.constexpr,
                     SH: tl.constexpr, SW: tl.constexpr,
                     PH: tl.constexpr, PW: tl.constexpr,
                     OH: tl.constexpr, OW: tl.constexpr,
                     BLOCK: tl.constexpr):
    channel = tl.program_id(1)
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oh = spatial // OW
    ow = spatial % OW
    valid_output = (channel < C) & (spatial < OH * OW)
    accumulator = tl.full((BLOCK,), 0.0, tl.float32)
    for kh in tl.static_range(KH):
        ih = oh * SH + kh - PH
        for kw in tl.static_range(KW):
            iw = ow * SW + kw - PW
            valid_input = valid_output & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x = tl.load(X + channel * H * W + ih * W + iw,
                        valid_input, other=0.0)
            # OIHW [C,1,KH,KW]: input channel index is always zero.
            weight = tl.load(Weight + (channel * KH + kh) * KW + kw,
                             channel < C, other=0.0)
            accumulator = accumulator + x * weight
    bias = tl.load(Bias + channel, channel < C, other=0.0)
    tl.store(Out + channel * OH * OW + spatial, accumulator + bias, valid_output)


@triton.jit
def pointwise_conv2d(X, Weight, Bias, Out, CIN: tl.constexpr,
                     COUT: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    spatial = tl.program_id(0) * BM + tl.arange(0, BM)
    channels = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(CIN, BK)):
        kk = start * BK + k
        # Logical A[m,k] comes from NCHW X[0,k,h,w], NOT X[m*CIN+k].
        a = tl.load(X + kk[None, :] * (H * W) + spatial[:, None],
                    (spatial[:, None] < H * W) & (kk[None, :] < CIN), other=0.0)
        # Logical B[k,n] comes directly from OIHW Weight[n,k,0,0].
        b = tl.load(Weight + channels[None, :] * CIN + kk[:, None],
                    (channels[None, :] < COUT) & (kk[:, None] < CIN), other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    bias = tl.load(Bias + channels, channels < COUT, other=0.0)
    # Write the logical [M,N] dot tile back to physical NCHW [N,M].
    tl.store(Out + channels[None, :] * (H * W) + spatial[:, None],
             accumulator + bias[None, :],
             (spatial[:, None] < H * W) & (channels[None, :] < COUT))


@triton.jit
def conv_stem(X, Weight, Bias, Out, CIN: tl.constexpr, COUT: tl.constexpr,
              H: tl.constexpr, W: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
              SH: tl.constexpr, SW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
              OH: tl.constexpr, OW: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # The single MobileNetV3 stem specialization has K=3*3*3=27, BK=32.
    tl.static_assert(BK >= CIN * KH * KW)
    spatial = tl.program_id(0) * BM + tl.arange(0, BM)
    channels = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ic = k // (KH * KW)
    kh = (k // KW) % KH
    kw = k % KW
    ih = (spatial // OW)[:, None] * SH + kh[None, :] - PH
    iw = (spatial % OW)[:, None] * SW + kw[None, :] - PW
    valid = ((spatial[:, None] < OH * OW) & (k[None, :] < CIN * KH * KW)
             & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W))
    # Gather only this dot tile from NCHW; padded pixels and K lanes are zero.
    a = tl.load(X + (ic[None, :] * H + ih) * W + iw, valid, other=0.0)
    # OIHW: Weight[oc,ic,kh,kw], with k=(ic*KH+kh)*KW+kw.
    b = tl.load(Weight + channels[None, :] * (CIN * KH * KW) + k[:, None],
                (channels[None, :] < COUT) & (k[:, None] < CIN * KH * KW), other=0.0)
    result = tl.dot(a, b, input_precision="ieee")
    bias = tl.load(Bias + channels, channels < COUT, other=0.0)
    tl.store(Out + channels[None, :] * (OH * OW) + spatial[:, None],
             result + bias[None, :],
             (spatial[:, None] < OH * OW) & (channels[None, :] < COUT))
