"""Optional QK with original [head, capacity, head_dim] cache storage."""
import triton
import triton.language as tl


@triton.jit
def attention_qk_position_native(A, B, Position, C, M: tl.constexpr,
                                 N: tl.constexpr, K: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr,
                                 BK: tl.constexpr, QK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    lane = tl.arange(0, BK)
    valid = tl.minimum(tl.maximum(tl.load(Position + M - 1) + 1, 0), N)
    result = tl.full((BM, BN), 0, tl.float32)
    if tl.program_id(1) * BN < valid:
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + lane
            a = tl.load(A + head * M * K + rows[:, None] * K + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < K), other=0)
            # Only the current active K tile is read/transposed. The backing
            # cache remains [head, N, K], exactly as provided by the model.
            b = tl.load(B + head * N * K + cols[None, :] * K + kk[:, None],
                        (kk[:, None] < K) & (cols[None, :] < valid), other=0)
            result = tl.dot(a, b, result, input_precision="ieee")
    tl.store(C + head * M * N + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))
