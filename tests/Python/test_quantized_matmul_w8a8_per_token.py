# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform.quantization import w8a8_channel_wise
from buddy.compiler.graph.operation import QuantizedMatmulOp
from buddy.compiler.ops import tosa

TOL = 1e-5


def quantize_weight(W):
    amax = W.abs().amax(dim=0, keepdim=True)
    s_W = (amax / 127.0).clamp(min=1e-10)
    W_i8 = torch.clamp(torch.round(W / s_W), -128, 127).to(torch.int8)
    return W_i8, s_W


def oracle_per_token(x, W_i8, s_W):
    amax = x.abs().amax(dim=1, keepdim=True)
    c127 = torch.full_like(amax, 127.0)
    eps = torch.full_like(amax, 1e-10)
    s_A = torch.maximum(amax * torch.reciprocal(c127), eps)
    x_i8 = torch.clamp(torch.round(x / s_A), -128, 127).to(torch.int8)
    # i32 accumulation is exact in the compiled graph, so the oracle
    # accumulates in int64 - bit-identical, zero f32 rounding noise.
    acc = x_i8.to(torch.int64) @ W_i8.to(torch.int64)
    out = acc.to(torch.float32) * (s_A * s_W)
    return out, x_i8, s_A


def run_case(name, x, W, check=None):
    torch.manual_seed(0)

    class MatMulModel(torch.nn.Module):
        def __init__(self, W):
            super().__init__()
            self.weight = torch.nn.Parameter(W)

        def forward(self, x):
            return torch.matmul(x, self.weight)

    model = MatMulModel(W)

    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry, func_name="forward"
    )
    with torch.no_grad():
        graphs = dynamo.importer(model, x)
    assert len(graphs) == 1
    graph = graphs[0]

    w8a8_channel_wise(graph, activation_granularity="per_token")

    qops = [op for op in graph.body if isinstance(op, QuantizedMatmulOp)]
    assert len(qops) == 1, (
        f"{name}: Expected 1 QuantizedMatmulOp, found {len(qops)}"
    )
    assert qops[0].kwargs.get("activation_granularity") == "per_token", (
        f"{name}: Expected activation_granularity to be 'per_token'"
    )

    graph.lower_to_top_level_ir()
    print(f"TEST: {name}")
    print(graph._imported_module)

    W_i8, s_W = quantize_weight(W)
    out = dynamo.dynamo_run()(W_i8, s_W.float(), x)[0]

    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    out_ref, _, _ = oracle_per_token(x, W_i8, s_W)
    diff = (out - out_ref).abs().max().item()
    assert diff < TOL, f"{name}: max diff {diff} exceeds tolerance {TOL}"
    if check is not None:
        check(out, out_ref, x, W)


K, N = 17, 13

# CHECK-LABEL: TEST: decode_m1
# CHECK: func.func @forward(%{{.*}}: tensor<17x13xi8>, %{{.*}}: tensor<1x13xf32>, %{{.*}}: tensor<1x17xf32>)
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32} : (tensor<1x17xf32>) -> tensor<1x1xf32>
# CHECK-NOT: tosa.reduce_max
# CHECK: linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%{{.*}}, %{{.*}} : tensor<1x17xi8>, tensor<17x13xi8>) outs(%{{.*}} : tensor<1x13xi32>)
torch.manual_seed(1)
run_case("decode_m1", torch.randn(1, K), torch.randn(K, N))

# CHECK-LABEL: TEST: prefill_m3
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32} : (tensor<3x17xf32>) -> tensor<3x1xf32>
# CHECK-NOT: tosa.reduce_max
# CHECK: linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%{{.*}}, %{{.*}} : tensor<3x17xi8>, tensor<17x13xi8>) outs(%{{.*}} : tensor<3x13xi32>)
# CHECK: tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<3x1xf32>, tensor<1x13xf32>, tensor<1xi8>) -> tensor<3x13xf32>
# CHECK: tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<3x13xf32>, tensor<3x13xf32>, tensor<1xi8>) -> tensor<3x13xf32>
torch.manual_seed(42)
run_case("prefill_m3", torch.randn(3, K), torch.randn(K, N))

# All-zero activation row: scale clamps to eps, row quantizes to 0,
# output row must be exactly zero and finite.
# CHECK-LABEL: TEST: zero_row
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32}
torch.manual_seed(2)
_x = torch.randn(3, K)
_x[1] = 0.0


def _check_zero_row(out, ref, x, W):
    assert (out[1] == 0).all(), "zero row must produce exact zeros"


run_case("zero_row", _x, torch.randn(K, N), check=_check_zero_row)

# All-zero weight channel: that output column must be exactly zero.
# CHECK-LABEL: TEST: zero_channel
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32}
torch.manual_seed(3)
_W = torch.randn(K, N)
_W[:, 5] = 0.0


def _check_zero_channel(out, ref, x, W):
    assert (out[:, 5] == 0).all(), "zero channel must produce exact zeros"


run_case("zero_channel", torch.randn(3, K), _W, check=_check_zero_channel)

# Rows of wildly different magnitude: the motivation case.
# Per-row error vs FP32 is printed for the record; correctness is vs oracle.
# CHECK-LABEL: TEST: mixed_magnitude
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32}
torch.manual_seed(4)
_x = torch.randn(3, K) * torch.tensor([100.0, 1.0, 0.01]).unsqueeze(1)
_W = torch.randn(K, N)


def _report(out, ref, x, W):
    err = (out - x @ W).abs().amax(dim=1)
    print(f"mixed_magnitude per-row |compiled - fp32|: {err.tolist()}")


run_case("mixed_magnitude", _x, _W, check=_report)

# Rounding boundaries: x/s_A hits exact halves (s = 2^-7 makes the
# division exact in f32). torch.round is round-half-to-even; if the
# lowering ever truncates instead, compiled != oracle by ~1 quantum.
# CHECK-LABEL: TEST: rounding_boundary
# CHECK: tosa.reduce_max %{{.*}} {axis = 1 : i32}
_s = 2.0**-7
_units = torch.tensor(
    [
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        126.5,
        -126.5,
        3.5,
        -4.5,
        4.5,
        -3.5,
        0.5,
        -0.5,
        1.5,
        -1.5,
        127.0,
    ]
)
torch.manual_seed(5)
run_case("rounding_boundary", (_units * _s).unsqueeze(0), torch.randn(K, N))
