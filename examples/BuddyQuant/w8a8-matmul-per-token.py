#!/usr/bin/env python3
# ===- w8a8-matmul-per-token.py ------------------------------------------------
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
# W8A8 quantized MatMul example: compares per-tensor and per-token dynamic
# activation quantization against FP32 on CPU.
#
# ===---------------------------------------------------------------------------

"""Compare FP32, per-tensor W8A8, and per-token W8A8 MatMul on CPU.

Per-tensor W8A8 uses one activation scale for the whole batch, so rows with
small magnitudes lose resolution. Per-token W8A8 computes one scale per row
of the activation, keeping the full int8 range for every row.

This script compiles a static 2D MatMul through the buddy graph pass
`w8a8_channel_wise` in both activation granularities, executes the generated
MLIR on CPU via the MLIR ExecutionEngine, and reports the quantization error
of both modes against FP32.

Requires the buddy python packages and the MLIR runner utils libraries:
    ninja -C build python-package-buddy
    ninja -C llvm/build mlir_runner_utils mlir_c_runner_utils

Run with:
    python examples/BuddyQuant/w8a8-matmul-per-token.py
"""

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform.quantization import w8a8_channel_wise
from buddy.compiler.ops import tosa

SEED = 42


class MatMulModel(torch.nn.Module):
    """Minimal static 2D MatMul: x[M, K] @ weight[K, N]."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


def quantize_weight(weight):
    """Channel-wise (per output channel N) symmetric int8 quantization."""
    amax = weight.abs().amax(dim=0, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-10)
    weight_i8 = torch.clamp(torch.round(weight / scale), -128, 127)
    return weight_i8.to(torch.int8), scale


def compile_w8a8_matmul(x, weight, activation_granularity):
    """Import x @ weight, apply W8A8, and JIT-compile for CPU execution."""
    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry, func_name="forward"
    )
    with torch.no_grad():
        graphs = dynamo.importer(MatMulModel(weight), x)
    assert len(graphs) == 1
    w8a8_channel_wise(graphs[0], activation_granularity=activation_granularity)
    return dynamo.dynamo_run()


def relative_l2(q, ref):
    """||q - ref||_2 / ||ref||_2; a zero reference defines 0/0 as 0."""
    num = torch.linalg.vector_norm(q - ref).item()
    den = torch.linalg.vector_norm(ref).item()
    if den == 0:
        return 0.0 if num == 0 else float("inf")
    return num / den


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    M, K, N = 3, 17, 13  # non-square, odd contraction dim
    weight = torch.randn(K, N)
    x = torch.randn(M, K)
    weight_i8, weight_scale = quantize_weight(weight)

    out_fp32 = x @ weight

    rows = [("fp32 (torch)", 0.0, 0.0)]
    compiled = {}
    for mode in ("per_tensor", "per_token"):
        fn = compile_w8a8_matmul(x, weight, mode)
        compiled[mode] = fn
        out = fn(weight_i8, weight_scale.float(), x)[0]
        rows.append(
            (
                f"w8a8 {mode}",
                (out - out_fp32).abs().max().item(),
                relative_l2(out, out_fp32),
            )
        )

    print(f"MatMul [{M}x{K}] @ [{K}x{N}], seed={SEED}, ", end="")
    print(f"torch_threads={torch.get_num_threads()}")
    print(f"{'mode':16s} {'max abs err':>12s} {'rel L2':>12s}")
    for name, mae, rl2 in rows:
        print(f"{name:16s} {mae:12.6f} {rl2:12.6f}")

    # Motivation case: rows of very different magnitude. Per-tensor shares
    # one activation scale across rows; per-token gives each row its own
    # scale, so small-magnitude rows keep their int8 resolution.
    row_scales = torch.tensor([100.0, 1.0, 0.01]).unsqueeze(1)
    x_mixed = torch.randn(M, K) * row_scales
    ref_mixed = x_mixed @ weight
    print("\nmixed-magnitude input (rows scaled by [100, 1, 0.01]):")
    print("per-row max abs error vs FP32")
    for mode, fn in compiled.items():
        out = fn(weight_i8, weight_scale.float(), x_mixed)[0]
        err = (out - ref_mixed).abs().amax(dim=1)
        err_str = ", ".join(f"row {m}: {e:.6f}" for m, e in enumerate(err))
        print(f"  w8a8 {mode:10s} {err_str}")


if __name__ == "__main__":
    main()
