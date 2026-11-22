# RUN: %PYTHON %s buddy-opt mlir-opt
"""Check the layout read by GEMV and compile mixed panel widths to RISC-V."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from buddy.compiler.graph.operation import MatmulOp, PlaceholderOp
from buddy.compiler.graph.transform.pack_decode_weights import (
    pack_decode_matmul_weights,
)

sys.path.insert(
    0, str(Path(os.environ["BUDDY_SRC_ROOT"]) / "tools/buddy-codegen")
)
from compile_pipeline import build_stages  # noqa: E402
from gen_config import (  # noqa: E402
    decode_panel_widths,
    derive_decode_pack,
    derive_fp16_optimizations,
)

rules = []
functions = []
weights = []
nodes = []
params = []
widths = [8, 16, 24, 32]
for i, width in enumerate(widths):
    k, n = 24 + 8 * i, 2 * width
    rules.append(
        {"weight_shape": [k, n], "packed_shape": [2, k // 8, width // 8, 8, 8]}
    )
    weight = PlaceholderOp()
    weight.name = f"w{i}"
    op = MatmulOp()
    op.add_argument("x")
    op.add_argument(weight.name)
    nodes.append(op)
    params.append(weight)
    weights.append((torch.arange(k * n).reshape(k, n) % 127).half())
    functions.append(f"""
      func.func @gemv_{width}(%a: memref<1x{k}xf16>, %b: memref<{k}x{n}xf16>,
                               %c: memref<1x{n}xf16>) {{
        linalg.matmul ins(%a, %b: memref<1x{k}xf16>, memref<{k}x{n}xf16>)
                      outs(%c: memref<1x{n}xf16>)
        return
      }}
    """)

spec = {"ime_decode_packing": rules, "decode_pack_vector_size": 32}
ime = derive_fp16_optimizations(spec, "f16")["ime"]
pack = derive_decode_pack({}, spec)
graph = SimpleNamespace(
    params=params,
    body=nodes,
    node_table={p.name: p for p in params},
    _params_ref=weights.copy(),
)
pack_decode_matmul_weights(
    graph, ime_shapes=pack["ime_shapes"], ime_widths=pack["ime_widths"]
)
for width, weight, packed in zip(widths, weights, graph._params_ref):
    k, n = weight.shape
    # Read the actual serialized bytes in the kernel's loop order.
    blocks = packed.reshape(n // width, k // 8, width, 8).float()
    x = torch.arange(k).float() % 7
    actual = (blocks * x.reshape(1, k // 8, 1, 8)).sum(dim=(1, 3)).flatten()
    assert torch.equal(actual, x @ weight.float())

stages = build_stages(
    "subgraph_decode",
    2,
    "-mtriple=riscv64 -mattr=+v,+zvfh",
    variant="f16",
    decode_pack=pack,
    ime=ime,
)
ir = "\n".join(functions).encode()
for tool, args in stages:
    executable = (
        sys.argv[1]
        if tool == "buddy-opt"
        else Path(sys.argv[2]).with_name(tool)
    )
    ir = subprocess.run(
        [executable, *args], input=ir, stdout=subprocess.PIPE, check=True
    ).stdout
    if tool == "mlir-translate":
        kernels = [line for line in ir.splitlines() if b"bnez t6" in line]
        assert len(kernels) == len(widths)
        opcodes = [0x9E81482B, 0x9EA1492B, 0x9EC14A2B, 0x9EE14B2B]
        for width in widths:
            groups = width // 8
            stride = f"addi t2, t2, {groups * 128}".encode()
            (body,) = [line for line in kernels if stride in line]
            # Preserve the original panel-address operands and their order.
            moves = [
                f"mv t{2 + panel}, ${1 + panel}" for panel in range(groups)
            ]
            moves.append(f"mv t6, ${groups + 2}")
            positions = [body.index(move.encode()) for move in moves]
            assert positions == sorted(positions)
            for panel, opcode in enumerate(opcodes):
                assert (f".word {opcode:#x}".encode() in body) == (
                    panel < groups
                )
assert ir.startswith(b"\x7fELF")
assert int.from_bytes(ir[18:20], "little") == 243  # EM_RISCV


# Old specs and generated configurations use the original four-tile layout.
legacy = {
    "ime_decode_shapes": ["48x64"],
    "ime_decode_kernel": "gemv",
    "decode_pack_vector_size": 32,
}
legacy_pack = derive_decode_pack({}, legacy)
assert legacy_pack["ime_layout"] == "n32k8"
graph._params_ref = weights.copy()
graph.body = [nodes[-1]]
pack_decode_matmul_weights(graph, ime_shapes=legacy_pack["ime_shapes"])
reference = weights[-1].reshape(6, 8, 2, 4, 8).permute(2, 0, 3, 4, 1)
assert torch.equal(graph._params_ref[-1].reshape(reference.shape), reference)

for invalid in (
    [rules[0], rules[0]],
    [{"weight_shape": [24, 16], "packed_shape": [1, 3, 1, 8, 8]}],
    [{"weight_shape": [24, 40], "packed_shape": [1, 3, 5, 8, 8]}],
):
    try:
        decode_panel_widths(invalid)
    except ValueError:
        continue
    raise AssertionError(f"accepted invalid Decode layout: {invalid}")
