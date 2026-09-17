"""Real Triton front-end definitions for the Qwen3 operator examples.

The host only specializes shapes and launches grid programs. Every tensor load,
arithmetic operation, reduction, dot product and store lives in these JIT bodies.
No CUDA runtime or GPU is needed by the triton-riscv CPU front-end.
"""
import triton
import triton.language as tl


@triton.jit
def linear(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           INTEGER: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    if INTEGER:
        accumulator = tl.full((BM, BN), 0, tl.int32)
    else:
        accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        if M % BM == 0 and K % BK == 0:
            a = tl.load(A + rows[:, None] * K + kk[None, :])
        else:
            a = tl.load(A + rows[:, None] * K + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < K), other=0)
        # Model linear weights are physically [N,K]; the logical B is [K,N].
        if N % BN == 0 and K % BK == 0:
            b = tl.load(B + cols[None, :] * K + kk[:, None])
        else:
            b = tl.load(B + cols[None, :] * K + kk[:, None],
                        (cols[None, :] < N) & (kk[:, None] < K), other=0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    if M % BM == 0 and N % BN == 0:
        previous = tl.load(address)
        tl.store(address, accumulator + previous)
    else:
        previous = tl.load(address, valid, other=0)
        tl.store(address, accumulator + previous, valid)


@triton.jit
def attention_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    k = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(A + head * M * K + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + head * K * N + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    tl.store(C + head * M * N + rows[:, None] * N + cols[None, :], accumulator,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def binary(X, Y, Out, COUNT: tl.constexpr, MULTIPLY: tl.constexpr,
           BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < COUNT, other=0)
    y = tl.load(Y + index, index < COUNT, other=0)
    if MULTIPLY:
        result = x * y
    else:
        result = x + y
    tl.store(Out + index, result, index < COUNT)


@triton.jit
def silu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < COUNT, other=0)
    result = tl.div_rn(x, 1.0 + tl.exp(-x))
    tl.store(Out + index, result, index < COUNT)


@triton.jit
def rmsnorm(X, Weight, Sums, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + channel, channel < WIDTH, other=0)
    weight = tl.load(Weight + channel, channel < WIDTH, other=0)
    squares = tl.sum(x * x, axis=0)
    denominator = tl.sqrt(squares / WIDTH + 1.0e-6)
    result = tl.div_rn(x, denominator) * weight
    tl.store(Sums + row, squares)
    tl.store(Out + row * WIDTH + channel, result, channel < WIDTH)


@triton.jit
def rope(X, Cosine, Sine, Out, HEADS: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    channel = tl.arange(0, 64)
    index = (token * HEADS + head) * 128 + channel
    first = tl.load(X + index)
    second = tl.load(X + index + 64)
    cosine = tl.load(Cosine + token * 64 + channel)
    sine = tl.load(Sine + token * 64 + channel)
    tl.store(Out + index, first * cosine - second * sine)
    tl.store(Out + index + 64, second * cosine + first * sine)


@triton.jit
def softmax(X, Maxima, Sums, Out, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * LENGTH + column, column < LENGTH, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponentials = tl.exp(x - maximum)
    total = tl.sum(exponentials, axis=0)
    probabilities = tl.div_rn(exponentials, total)
    tl.store(Maxima + row, maximum)
    tl.store(Sums + row, total)
    tl.store(Out + row * LENGTH + column, probabilities, column < LENGTH)


@triton.jit
def attention_scale_mask(X, Out, SEQUENCE: tl.constexpr, TOTAL: tl.constexpr,
                         PAST: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count: tl.constexpr = 16 * SEQUENCE * TOTAL
    value = tl.load(X + index, index < count, other=0)
    query = (index // TOTAL) % SEQUENCE
    key = index % TOTAL
    result = tl.where(key <= PAST + query, value * 0.08838834764831845, -float("inf"))
    tl.store(Out + index, result, index < count)


@triton.jit
def attention_scale_mask_position(X, Position, Out, SEQUENCE: tl.constexpr,
                                  TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    """Scale + causal mask with the query position supplied at run time.

    The static ``PAST`` constexpr in ``attention_scale_mask`` only describes a
    cache that is exactly as long as the sequence it holds. A decoder serving
    many steps from a fixed-capacity cache does not have that property: the
    number of valid keys grows every step while ``TOTAL`` stays 512, so the mask
    boundary has to come from the runtime cache position. This kernel matches
    the masking the compiled Qwen3 graph actually performs, where the boundary
    is ``cache_position`` compared against the slot index.
    """
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count: tl.constexpr = 16 * SEQUENCE * TOTAL
    value = tl.load(X + index, index < count, other=0)
    query = (index // TOTAL) % SEQUENCE
    key = index % TOTAL
    boundary = tl.load(Position + query, query < SEQUENCE, other=0)
    result = tl.where(key <= boundary, value * 0.08838834764831845,
                      -float("inf"))
    tl.store(Out + index, result, index < count)


@triton.jit
def embedding(Ids, Out, Weight, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    token_index = tl.program_id(0)
    token = tl.load(Ids + token_index)
    channel = tl.arange(0, BLOCK)
    value = tl.load(Weight + token * WIDTH + channel, channel < WIDTH, other=0)
    tl.store(Out + token_index * WIDTH + channel, value, channel < WIDTH)


@triton.jit
def embedding_w8a8(Ids, Out, Weight, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Int8 row lookup rescaled to f32: out[t, :] = weight[ids[t], :] * scale[ids[t]]."""
    token_index = tl.program_id(0)
    token = tl.load(Ids + token_index)
    channel = tl.arange(0, BLOCK)
    value = tl.load(Weight + token * WIDTH + channel, channel < WIDTH, other=0)
    scale = tl.load(Scale + token)
    tl.store(Out + token_index * WIDTH + channel,
             value.to(tl.float32) * scale, channel < WIDTH)


@triton.jit
def kv_cache_update(X, Cache, PAST: tl.constexpr, CAPACITY: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    channel = tl.arange(0, 128)
    value = tl.load(X + (token * 8 + head) * 128 + channel)
    tl.store(Cache + (head * CAPACITY + PAST + token) * 128 + channel, value)


@triton.jit
def kv_cache_update_position(X, Position, Cache, CAPACITY: tl.constexpr):
    """KV write whose destination slot comes from a run-time position tensor.

    ``PAST`` is a constexpr, so serving step 2..8 of a decode loop would need a
    separate compiled kernel per step. The compiled Qwen3 graph instead adds a
    runtime ``cache_position`` to a per-token index and stores there, so the
    deployable form takes the absolute slot for each token as data. ``Position``
    holds one absolute destination slot per incoming token.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    channel = tl.arange(0, 128)
    slot = tl.load(Position + token)
    value = tl.load(X + (token * 8 + head) * 128 + channel)
    tl.store(Cache + (head * CAPACITY + slot) * 128 + channel, value)


@triton.jit
def gqa_repeat(X, Out, TOTAL: tl.constexpr):
    query_head = tl.program_id(0)
    token = tl.program_id(1)
    channel = tl.arange(0, 128)
    value = tl.load(X + ((query_head // 2) * TOTAL + token) * 128 + channel)
    tl.store(Out + (query_head * TOTAL + token) * 128 + channel, value)


@triton.jit
def transpose(X, Out, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
              P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
              BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = index // (D1 * D2)
    b = (index // D2) % D1
    c = index % D2
    if P0 == 0:
        o0 = a
    elif P0 == 1:
        o0 = b
    else:
        o0 = c
    if P1 == 0:
        o1 = a
        S1: tl.constexpr = D0
    elif P1 == 1:
        o1 = b
        S1: tl.constexpr = D1
    else:
        o1 = c
        S1: tl.constexpr = D2
    if P2 == 0:
        o2 = a
        S2: tl.constexpr = D0
    elif P2 == 1:
        o2 = b
        S2: tl.constexpr = D1
    else:
        o2 = c
        S2: tl.constexpr = D2
    valid = index < D0 * D1 * D2
    value = tl.load(X + index, valid, other=0)
    tl.store(Out + (o0 * S1 + o1) * S2 + o2, value, valid)


@triton.jit
def quantize(X, Q, Scale, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + channel, channel < WIDTH, other=0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.div_rn(x, scale)
    rounded = value + tl.where(value >= 0.0, 0.5, -0.5)
    saturated = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    integer = saturated.to(tl.int32).to(tl.int8)
    tl.store(Scale + row, scale)
    tl.store(Q + row * WIDTH + channel, integer, channel < WIDTH)


@triton.jit
def dequantize(X, Row, Column, Out, ROWS: tl.constexpr, COLS: tl.constexpr,
               BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    row = tl.load(Row + index // COLS, valid, other=0)
    column = tl.load(Column + index % COLS, valid, other=0)
    tl.store(Out + index, (value * row) * column, valid)
