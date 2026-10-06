# RUN: %PYTHON %s buddy-opt %openmp_runtime_dir 2>&1 | FileCheck %s
#
# The kernels of graph/transform/k3_w4.py (variant w4g32), lowered with the
# "kernels" pipeline of compile_pipeline.py up to LLVM IR and run on the host
# against numpy references: the int4 matmul kernels (plain, with bias, with a
# fused RMSNorm, q / k / v, gate / up / SiLU), one decode row and a block of
# prefill rows; and the attention kernels (RoPE, KV cache update, causal
# attention), decode and prefill, and the prefill one of "prefill_ime" with
# its two matrix loops emulated (the host has no IME).

import ctypes
import os
import subprocess
import sys

import numpy
from buddy.compiler.graph.transform import k3_w4
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


def jit(specs, emulate_ime=False):
    """The kernels of `specs`, lowered like compile_pipeline.py does."""
    stages = compile_pipeline.build_stages("kernels", THREADS, "", "w4g32")
    passes = next(args for tool, args in stages if tool == "buddy-opt")
    llvm = subprocess.run(
        [BUDDY_OPT, *passes],
        input=k3_w4.gen_kernels(specs, emulate_ime=emulate_ime),
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
    """Largest error relative to the largest output, against the reference of
    the kernel's arithmetic (exact) and against the float model (float)."""
    scale = numpy.abs(want).max()
    err = numpy.abs(got - want).max() / scale
    print(f"{label}: {got.shape} error {'ok' if err < exact else err}")


# ── matmul kernels ──────────────────────────────────────────────────────────


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


EPS = 1e-6
for m in (1, 8):
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
    plain = k3_w4.kernel_spec("plain", m, k, [256], False, THREADS)
    rms = dict(plain, name=plain["name"] + "_rms", norm_eps=EPS)
    multi = k3_w4.kernel_spec("multi", m, k, [256, 128, 128], True, THREADS)
    glu = k3_w4.kernel_spec("glu", m, k, [256], False, THREADS)
    ee = jit([plain, rms, multi, glu])

    (y,) = call(ee, plain["name"], 1, [2], x, k3_w4.pack_q4(w))
    report(f"plain m{m}", y, ref_matmul(x, w), 1e-5)
    # int4 weights: within about 10% of the float matmul (RMS)
    f = x @ w
    rms_err = numpy.sqrt(((y - f) ** 2).mean() / (f**2).mean())
    print(f"plain m{m} vs float: {rms_err < 0.15}")

    (y,) = call(ee, rms["name"], 1, [2], x, k3_w4.pack_q4(w), nw)
    xn = rmsnorm(x, nw, EPS).astype(numpy.float32)
    report(f"rmsnorm m{m}", y, ref_matmul(xn, w), 1e-5)

    packed = k3_w4.pack_q4(numpy.concatenate([wq, wk, wv], axis=1))
    ys = call(ee, multi["name"], 3, [2, 2, 2], x, packed, bias)
    want = ref_matmul(x, numpy.concatenate([wq, wk, wv], axis=1)) + bias
    for name, got, lo, hi in zip("qkv", ys, (0, 256, 384), (256, 384, 512)):
        report(f"multi {name} m{m}", got, want[:, lo:hi], 1e-5)

    (y,) = call(ee, glu["name"], 1, [2], x, k3_w4.pack_glu(wg, wu))
    want = silu(ref_matmul(x, wg)) * ref_matmul(x, wu)
    report(f"glu m{m}", y, want, 1e-5)
# CHECK: plain m1: (1, 256) error ok
# CHECK-NEXT: plain m1 vs float: True
# CHECK-NEXT: rmsnorm m1: (1, 256) error ok
# CHECK-NEXT: multi q m1: (1, 256) error ok
# CHECK-NEXT: multi k m1: (1, 128) error ok
# CHECK-NEXT: multi v m1: (1, 128) error ok
# CHECK-NEXT: glu m1: (1, 256) error ok
# CHECK-NEXT: plain m8: (8, 256) error ok
# CHECK-NEXT: plain m8 vs float: True
# CHECK-NEXT: rmsnorm m8: (8, 256) error ok
# CHECK-NEXT: multi q m8: (8, 256) error ok
# CHECK-NEXT: multi k m8: (8, 128) error ok
# CHECK-NEXT: multi v m8: (8, 128) error ok
# CHECK-NEXT: glu m8: (8, 256) error ok

# The lm_head of a prefill chunk ("prefill_logits") computes nothing after
# buddy_set_prefill_logits(0), and its logits again after (1).
lm = k3_w4.kernel_spec("plain", 1, 64, [256], False, THREADS)
lm.update(name=lm["name"] + "_prefill_logits", prefill_logits=True)
ee = jit([lm])
x = rng.standard_normal((1, 64)).astype(numpy.float32)
w = (rng.standard_normal((64, 256)) * 0.1).astype(numpy.float32)
want = ref_matmul(x, w)
for flag in (0, 1):
    ee.invoke("buddy_set_prefill_logits", ctypes.pointer(ctypes.c_int32(flag)))
    (y,) = call(ee, lm["name"], 1, [2], x, k3_w4.pack_q4(w))
    print(
        f"prefill logits {flag}: computed {numpy.allclose(y, want, atol=1e-4)}"
    )
# CHECK: prefill logits 0: computed False
# CHECK-NEXT: prefill logits 1: computed True


# ── attention kernels ───────────────────────────────────────────────────────


def rope(x, pos, inv_freq):
    """HF rotate_half RoPE of x [..., rows, d] at positions pos [rows]."""
    ang = pos[:, None].astype(numpy.float64) * inv_freq[None, :]
    cos = numpy.concatenate([numpy.cos(ang)] * 2, axis=1)
    sin = numpy.concatenate([numpy.sin(ang)] * 2, axis=1)
    half = x.shape[-1] // 2
    rot = numpy.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def heads(x, d):
    """[m, heads * d] -> [1, heads, m, d]"""
    m = x.shape[0]
    return x.reshape(m, -1, d).transpose(1, 0, 2)[None]


def f16(x):
    return x.astype(numpy.float16).astype(numpy.float64)


def ref_attention(q, k, v, kc, vc, start, inv_freq, scale, ime=False):
    """q / k / v as the projections give them ([m, heads * d]); returns the
    output in the same layout and the updated caches. With `ime`, rounded
    like the IME kernel: the scaled q, the keys, the values and the
    probabilities in fp16."""
    d = kc.shape[3]
    q, k, v = heads(q, d), heads(k, d), heads(v, d)
    m, group = q.shape[2], q.shape[1] // k.shape[1]
    pos = numpy.arange(start, start + m)
    kc, vc = kc.astype(numpy.float64), vc.astype(numpy.float64)
    kc[0, :, start : start + m] = rope(k[0], pos, inv_freq)
    vc[0, :, start : start + m] = v[0]
    qr = rope(q[0], pos, inv_freq)
    out = numpy.zeros(q.shape)
    for hh in range(q.shape[1]):
        for i in range(m):
            keys = kc[0, hh // group, : start + i + 1]
            vals = vc[0, hh // group, : start + i + 1]
            if ime:
                s = f16(keys) @ f16(qr[hh, i] * scale)
                p = numpy.exp(s - s.max())
                out[0, hh, i] = f16(p) @ f16(vals) / p.sum()
                continue
            s = keys @ qr[hh, i] * scale
            p = numpy.exp(s - s.max())
            out[0, hh, i] = p @ vals / p.sum()
    return out[0].transpose(1, 0, 2).reshape(m, -1), kc, vc


H, KVH, CTX = 4, 2, 96
# (m, start, ctx, head dim, ime); the IME kernel needs ctx % 64 == 0 and a
# head dim that is a multiple of 64
CASES = [(1, 0, CTX, 128, False), (1, 70, CTX, 128, False)]
CASES += [(32, 0, CTX, 128, False), (32, 41, CTX, 128, False)]
CASES += [(64, 0, 128, 128, True), (64, 41, 128, 128, True)]
CASES += [(64, 64, 128, 128, True), (64, 41, 128, 64, True)]
for m, start, ctx, D, ime in CASES:
    inv_freq = (10000.0 ** (-numpy.arange(0, D, 2) / D)).astype(numpy.float32)
    spec = {
        "name": f"k3_attn_m{m}_s{start}_d{D}",
        "kind": "attn",
        "m": m,
        "heads": H,
        "kv_heads": KVH,
        "dim": D,
        "scale": D**-0.5,
        "ctx": ctx,
    }
    if ime:
        spec.update(name=spec["name"] + "_ime", ime=True)
    ee = jit([spec], emulate_ime=ime)
    q = rng.standard_normal((m, H * D)).astype(numpy.float32)
    k = rng.standard_normal((m, KVH * D)).astype(numpy.float32)
    v = rng.standard_normal((m, KVH * D)).astype(numpy.float32)
    kc = numpy.zeros((1, KVH, ctx, D), numpy.float32)
    vc = numpy.zeros((1, KVH, ctx, D), numpy.float32)
    kc[0, :, :start] = rng.standard_normal((KVH, start, D))
    vc[0, :, :start] = rng.standard_normal((KVH, start, D))
    want, want_k, want_v = ref_attention(
        q, k, v, kc, vc, start, inv_freq, D**-0.5, ime
    )
    pos = numpy.array([start], numpy.int64)
    o, _, kco, vco = call(
        ee, spec["name"], 4, [2, 3, 4, 4], q, k, v, kc, vc, pos, inv_freq
    )
    label = f"attention m{m} start {start}" + (" ime" if ime else "")
    report(label, o, want, 5e-4 if ime else 1e-5)
    # the caches are updated in place and returned
    print(
        f"  caches: k {numpy.allclose(kc, want_k, atol=1e-5)}, "
        f"v {numpy.allclose(vc, want_v, atol=1e-6)}, "
        f"returned {numpy.array_equal(kco, kc) and numpy.array_equal(vco, vc)}"
    )
# CHECK: attention m1 start 0: (1, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m1 start 70: (1, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m32 start 0: (32, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m32 start 41: (32, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m64 start 0 ime: (64, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m64 start 41 ime: (64, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m64 start 64 ime: (64, 512) error ok
# CHECK-NEXT: caches: k True, v True, returned True
# CHECK-NEXT: attention m64 start 41 ime: (64, 256) error ok
# CHECK-NEXT: caches: k True, v True, returned True
