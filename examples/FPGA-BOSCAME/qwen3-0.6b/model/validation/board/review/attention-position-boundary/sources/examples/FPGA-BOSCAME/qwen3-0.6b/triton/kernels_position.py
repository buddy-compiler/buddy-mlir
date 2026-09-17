"""Optional attention kernels that retain physical capacity but skip unused KV."""
import triton
import triton.language as tl


@triton.jit
def attention_dot_position(A, B, Position, C, M: tl.constexpr, N: tl.constexpr,
                           K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr, QK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    lane = tl.arange(0, BK)
    capacity: tl.constexpr = N if QK else K
    # The caller enforces 0 <= position < capacity. Clamp the read bound for
    # memory safety as well; an invalid caller input is not a supported result.
    valid = tl.minimum(tl.maximum(tl.load(Position + M - 1) + 1, 0), capacity)
    if QK:
        result = tl.full((BM, BN), 0, tl.float32)
        if tl.program_id(1) * BN < valid:
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + lane
                # Inactive cache columns must not be read: zero * NaN is NaN.
                a = tl.load(A + head * M * K + rows[:, None] * K + kk[None, :],
                            (rows[:, None] < M) & (kk[None, :] < K), other=0)
                b = tl.load(B + head * K * N + kk[:, None] * N + cols[None, :],
                            (kk[:, None] < K) & (cols[None, :] < valid), other=0)
                result = tl.dot(a, b, result, input_precision="ieee")
        # Fully define inactive output tiles as zero. The existing causal mask
        # turns their positions into -inf before softmax.
        tl.store(C + head * M * N + rows[:, None] * N + cols[None, :], result,
                 (rows[:, None] < M) & (cols[None, :] < N))
    else:
        result = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(valid, BK)):
            kk = block * BK + lane
            a = tl.load(A + head * M * K + rows[:, None] * K + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < valid), other=0)
            b = tl.load(B + head * K * N + kk[:, None] * N + cols[None, :],
                        (kk[:, None] < valid) & (cols[None, :] < N), other=0)
            result = tl.dot(a, b, result, input_precision="ieee")
        tl.store(C + head * M * N + rows[:, None] * N + cols[None, :], result,
                 (rows[:, None] < M) & (cols[None, :] < N))
