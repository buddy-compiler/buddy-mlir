# RUN: %PYTHON %s buddy-opt %openmp_runtime_dir 2>&1 | FileCheck %s
#
# The matmul kernels of graph/transform/a210_w4.py (variant w4g32, kernels
# "a210"), lowered with the "kernels" pipeline of compile_pipeline.py up to
# LLVM IR and run on the host against numpy references: plain, with bias,
# with a fused RMSNorm, q / k / v, gate / up / SiLU; one decode row and a
# block of prefill rows; equal and weighted work splits.

import ctypes
import os
import platform
import re
import subprocess
import sys

import numpy
from buddy.compiler.graph.transform import a210_w4, k3_w4
from buddy_mlir import ir
from buddy_mlir.execution_engine import ExecutionEngine
from buddy_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402

BUDDY_OPT, OMP_DIR = sys.argv[1], sys.argv[2]
THREADS = 3
rng = numpy.random.default_rng(0)


def jit(specs):
    """The kernels of `specs`, lowered like compile_pipeline.py does."""
    stages = compile_pipeline.build_stages("kernels", THREADS, "", "w4g32")
    passes = next(args for tool, args in stages if tool == "buddy-opt")
    llvm = subprocess.run(
        [BUDDY_OPT, *passes],
        input=a210_w4.gen_kernels(specs),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with ir.Context():
        module = ir.Module.parse(llvm)
        return ExecutionEngine(
            module,
            opt_level=2,
            shared_libs=[os.path.join(OMP_DIR, "libomp.so")],
            enable_pic=platform.machine().lower() == "riscv64",
        )


def ptr(x):
    return ctypes.pointer(ctypes.pointer(x))


def results(n, ranks):
    """A struct of n result descriptors (one descriptor if n == 1)."""
    descs = [make_nd_memref_descriptor(r, ctypes.c_float) for r in ranks]
    if n == 1:
        return descs[0]()

    class Results(ctypes.Structure):
        _fields_ = [(f"r{i}", d) for i, d in enumerate(descs)]

    return Results()


def call(ee, name, n_results, ranks, *arrays):
    res = results(n_results, ranks)
    args = [ptr(get_ranked_memref_descriptor(a)) for a in arrays]
    ee.invoke(name, ptr(res), *args)
    if n_results == 1:
        return [ranked_memref_to_numpy([res])]
    return [
        ranked_memref_to_numpy([getattr(res, f"r{i}")])
        for i in range(n_results)
    ]


def report(label, got, want, exact):
    scale = numpy.abs(want).max()
    err = numpy.abs(got - want).max() / scale
    print(f"{label}: {got.shape} error {'ok' if err < exact else err}")


def quantize_x(x):
    """The kernels' activation quantization: int8 per 32-element group."""
    m, k = x.shape
    g = x.reshape(m, k // 32, 32).astype(numpy.float32)
    mx = numpy.abs(g).max(axis=2, keepdims=True)
    inv = numpy.where(
        mx > 0, numpy.float32(127) / numpy.where(mx > 0, mx, 1), 0
    )
    q = numpy.clip(numpy.rint(g * inv), -127, 127)
    return (q * (mx / numpy.float32(127))).reshape(m, k)


def ref_matmul(x, w):
    """x (quantized like the kernels) times w (int4 like the packing)."""
    wq = k3_w4.dequantize_q4(*k3_w4.quantize_q4(w))
    return quantize_x(x).astype(numpy.float64) @ wq


def rmsnorm(x, nw, eps):
    x64 = x.astype(numpy.float64)
    return x64 / numpy.sqrt((x64**2).mean(axis=1, keepdims=True) + eps) * nw


def silu(v):
    return v / (1 + numpy.exp(-v))


# The layout holds as many bytes as the K3 one (gen_config sizes it).
print(
    "packed bytes:",
    a210_w4.packed_bytes(96, 256) == k3_w4.packed_bytes(96, 256),
)
# CHECK: packed bytes: True

EPS = 1e-6
# (rows, thread weights, thread cores)
cases = (
    (1, None, None),
    (8, None, None),
    (1, [1, 2, 4], None),
    (8, [1, 3, 5], None),
    (8, [1, 3, 5], ["c908", "c920", "c908"]),
)
for m, weights, cores in cases:
    k = 96
    x = rng.standard_normal((m, k)).astype(numpy.float32)
    w = (rng.standard_normal((k, 256)) * 0.1).astype(numpy.float32)
    nw = (1 + 0.1 * rng.standard_normal(k)).astype(numpy.float32)
    bias = rng.standard_normal(512).astype(numpy.float32)
    wq, wk, wv = (
        (rng.standard_normal((k, n)) * 0.1).astype(numpy.float32)
        for n in (256, 128, 128)
    )
    wg, wu = (
        (rng.standard_normal((k, 256)) * 0.1).astype(numpy.float32)
        for _ in range(2)
    )

    def spec(kind, m, k, ns, bias, weights=weights, cores=cores):
        return a210_w4.kernel_spec(
            kind, m, k, ns, bias, THREADS, weights, cores
        )

    plain = spec("plain", m, k, [256], False)
    rms = dict(plain, name=plain["name"] + "_rms", norm_eps=EPS)
    multi = spec("multi", m, k, [256, 128, 128], True)
    glu = spec("glu", m, k, [256], False)
    ee = jit([plain, rms, multi, glu])
    tag = f"m{m} {'weighted' if weights else 'equal'}"
    if cores:
        tag += " mixed"

    (y,) = call(ee, plain["name"], 1, [2], x, a210_w4.pack_q4(w))
    report(f"plain {tag}", y, ref_matmul(x, w), 1e-5)

    (y,) = call(ee, rms["name"], 1, [2], x, a210_w4.pack_q4(w), nw)
    xn = rmsnorm(x, nw, EPS).astype(numpy.float32)
    report(f"rmsnorm {tag}", y, ref_matmul(xn, w), 1e-5)

    packed = a210_w4.pack_q4(numpy.concatenate([wq, wk, wv], axis=1))
    ys = call(ee, multi["name"], 3, [2, 2, 2], x, packed, bias)
    want = ref_matmul(x, numpy.concatenate([wq, wk, wv], axis=1)) + bias
    for name, got, lo, hi in zip("qkv", ys, (0, 256, 384), (256, 384, 512)):
        report(f"multi {name} {tag}", got, want[:, lo:hi], 1e-5)

    (y,) = call(ee, glu["name"], 1, [2], x, a210_w4.pack_glu(wg, wu))
    want = silu(ref_matmul(x, wg)) * ref_matmul(x, wu)
    report(f"glu {tag}", y, want, 1e-5)
# CHECK: plain m1 equal: (1, 256) error ok
# CHECK-NEXT: rmsnorm m1 equal: (1, 256) error ok
# CHECK-NEXT: multi q m1 equal: (1, 256) error ok
# CHECK-NEXT: multi k m1 equal: (1, 128) error ok
# CHECK-NEXT: multi v m1 equal: (1, 128) error ok
# CHECK-NEXT: glu m1 equal: (1, 256) error ok
# CHECK-NEXT: plain m8 equal: (8, 256) error ok
# CHECK-NEXT: rmsnorm m8 equal: (8, 256) error ok
# CHECK-NEXT: multi q m8 equal: (8, 256) error ok
# CHECK-NEXT: multi k m8 equal: (8, 128) error ok
# CHECK-NEXT: multi v m8 equal: (8, 128) error ok
# CHECK-NEXT: glu m8 equal: (8, 256) error ok
# CHECK-NEXT: plain m1 weighted: (1, 256) error ok
# CHECK-NEXT: rmsnorm m1 weighted: (1, 256) error ok
# CHECK-NEXT: multi q m1 weighted: (1, 256) error ok
# CHECK-NEXT: multi k m1 weighted: (1, 128) error ok
# CHECK-NEXT: multi v m1 weighted: (1, 128) error ok
# CHECK-NEXT: glu m1 weighted: (1, 256) error ok
# CHECK-NEXT: plain m8 weighted: (8, 256) error ok
# CHECK-NEXT: rmsnorm m8 weighted: (8, 256) error ok
# CHECK-NEXT: multi q m8 weighted: (8, 256) error ok
# CHECK-NEXT: multi k m8 weighted: (8, 128) error ok
# CHECK-NEXT: multi v m8 weighted: (8, 128) error ok
# CHECK-NEXT: glu m8 weighted: (8, 256) error ok
# CHECK-NEXT: plain m8 weighted mixed: (8, 256) error ok
# CHECK-NEXT: rmsnorm m8 weighted mixed: (8, 256) error ok
# CHECK-NEXT: multi q m8 weighted mixed: (8, 256) error ok
# CHECK-NEXT: multi k m8 weighted mixed: (8, 128) error ok
# CHECK-NEXT: multi v m8 weighted mixed: (8, 128) error ok
# CHECK-NEXT: glu m8 weighted mixed: (8, 256) error ok

# Mixed cores: one tile function per shape, each thread calling its own.
mixed = a210_w4.kernel_spec(
    "plain", 8, 96, [256], False, THREADS, None, ["c908", "c920", "c908"]
)
text = a210_w4.gen_kernels([mixed])
print("tiles:", sorted(set(re.findall(r"@(\w+__tile\w*)\(", text))))
# CHECK: tiles: ['a210_q4_plain_m8_k96_n256__tile_r4a2_generic', 'a210_q4_plain_m8_k96_n256__tile_r4a2_sifive_u74']

# Thread ranges cover the items once, in order, about in proportion.
print("split:", a210_w4.split_bounds(96, 8, [1] * 4 + [3.8] * 4))
# CHECK: split: [0, 5, 10, 15, 20, 39, 58, 77, 96]

# The lm_head of a prefill chunk ("prefill_logits") computes nothing after
# buddy_set_prefill_logits(0), and its logits again after (1).
lm = a210_w4.kernel_spec("plain", 1, 64, [256], False, THREADS)
lm.update(name=lm["name"] + "_prefill_logits", prefill_logits=True)
ee = jit([lm])
x = rng.standard_normal((1, 64)).astype(numpy.float32)
w = (rng.standard_normal((64, 256)) * 0.1).astype(numpy.float32)
want = ref_matmul(x, w)
for flag in (0, 1):
    ee.invoke("buddy_set_prefill_logits", ctypes.pointer(ctypes.c_int32(flag)))
    (y,) = call(ee, lm["name"], 1, [2], x, a210_w4.pack_q4(w))
    print(
        f"prefill logits {flag}: computed {numpy.allclose(y, want, atol=1e-4)}"
    )
# CHECK: prefill logits 0: computed False
# CHECK-NEXT: prefill logits 1: computed True
