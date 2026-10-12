# RUN: %PYTHON %s buddy-opt %openmp_runtime_dir 2>&1 | FileCheck %s
#
# The kernels of graph/transform/k3_w4.py for Qwen3-MoE models (variant
# w4g32), lowered with the "kernels" pipeline of compile_pipeline.py and run
# on the host against numpy references: the experts of a MoE block
# (buddy::moe_experts), one decode row and a block of prefill rows; and the
# attention with Qwen3's RMSNorm over each head of q and k (q_norm /
# k_norm), decode and prefill, and the prefill one of "prefill_ime" with its
# two matrix loops emulated.

import ctypes
import os
import platform
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
            enable_pic=platform.machine().lower() == "riscv64",
        )


def ptr(x):
    return ctypes.pointer(ctypes.pointer(x))


def call(ee, name, n_results, ranks, *arrays):
    descs = [make_nd_memref_descriptor(r, ctypes.c_float) for r in ranks]
    if n_results == 1:
        res = descs[0]()
    else:

        class Results(ctypes.Structure):
            _fields_ = [(f"r{i}", d) for i, d in enumerate(descs)]

        res = Results()
    args = [ptr(get_ranked_memref_descriptor(a)) for a in arrays]
    ee.invoke(name, ptr(res), *args)
    if n_results == 1:
        return [ranked_memref_to_numpy([res])]
    return [
        ranked_memref_to_numpy([getattr(res, f"r{i}")])
        for i in range(n_results)
    ]


def report(label, got, want, exact):
    """Largest error relative to the largest output."""
    err = numpy.abs(got - want).max() / numpy.abs(want).max()
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


def int4(w):
    return k3_w4.dequantize_q4(*k3_w4.quantize_q4(w)).astype(numpy.float64)


def silu(v):
    return v / (1 + numpy.exp(-v))


# ── MoE blocks ──────────────────────────────────────────────────────────────


def route(x, wr, topk):
    """HF Qwen3-MoE routing: softmax of x Wr, the topk largest, normalized."""
    logits = x.astype(numpy.float64) @ wr
    p = numpy.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    idx = numpy.argsort(-p, axis=1, kind="stable")[:, :topk]
    w = numpy.take_along_axis(p, idx, axis=1)
    return idx, w / w.sum(axis=1, keepdims=True)


def ref_moe(x, idx, w, gate_up, down):
    """The experts with the kernel's arithmetic: x and the intermediate
    quantized per group, the weights int4."""
    inter = down.shape[2]
    xq = quantize_x(x).astype(numpy.float64)
    y = numpy.zeros(x.shape)
    for r in range(x.shape[0]):
        for j in range(idx.shape[1]):
            e = idx[r, j]
            g = xq[r] @ int4(gate_up[e, :inter].T)
            u = xq[r] @ int4(gate_up[e, inter:].T)
            h = quantize_x((silu(g) * u)[None].astype(numpy.float32))[0]
            y[r] += w[r, j] * (h.astype(numpy.float64) @ int4(down[e].T))
    return y


HID, INTER, EXPERTS, TOPK = 256, 384, 16, 3
gate_up = (rng.standard_normal((EXPERTS, 2 * INTER, HID)) * 0.1).astype(
    numpy.float32
)
down = (rng.standard_normal((EXPERTS, HID, INTER)) * 0.1).astype(numpy.float32)
wr = rng.standard_normal((HID, EXPERTS)).astype(numpy.float32)
gu_tiles, dn_tiles = k3_w4.pack_moe(gate_up, down)
# one decode row; prefill rows grouped by expert (_moe_prefill), with
# experts of no row, of fewer and of more rows than a block (MOE_MB)
for m in (1, 8, 32):
    spec = k3_w4.moe_spec(m, HID, INTER, EXPERTS, TOPK, THREADS)
    ee = jit([spec])
    x = rng.standard_normal((m, HID)).astype(numpy.float32)
    (y,) = call(ee, spec["name"], 1, [2], x, wr, gu_tiles, dn_tiles)
    idx, w = route(x, wr, TOPK)
    report(f"moe m{m}", y, ref_moe(x, idx, w, gate_up, down), 1e-5)
# CHECK: moe m1: (1, 256) error ok
# CHECK-NEXT: moe m8: (8, 256) error ok
# CHECK-NEXT: moe m32: (32, 256) error ok


def rmsnorm(x, nw, eps):
    x = x.astype(numpy.float64)
    rinv = 1 / numpy.sqrt((x * x).mean(axis=1, keepdims=True) + eps)
    return (x * rinv * nw).astype(numpy.float32)


# with the RMSNorm of the experts' input fused (fuse_rmsnorm): the router
# and the experts see the normalized x
EPS = 1e-6
nw = (1 + 0.1 * rng.standard_normal(HID)).astype(numpy.float32)
for m in (1, 32):
    spec = k3_w4.moe_spec(m, HID, INTER, EXPERTS, TOPK, THREADS)
    spec = dict(spec, name=spec["name"] + "_rms", norm_eps=EPS)
    ee = jit([spec])
    x = (3 * rng.standard_normal((m, HID))).astype(numpy.float32)
    (y,) = call(ee, spec["name"], 1, [2], x, wr, gu_tiles, dn_tiles, nw)
    xn = rmsnorm(x, nw, EPS)
    idx, w = route(xn, wr, TOPK)
    report(f"moe rms m{m}", y, ref_moe(xn, idx, w, gate_up, down), 1e-5)
# CHECK-NEXT: moe rms m1: (1, 256) error ok
# CHECK-NEXT: moe rms m32: (32, 256) error ok

# with the residual add fused too (fuse_residual, one row): h + y, the
# same floats as the graph's add
spec = k3_w4.moe_spec(1, HID, INTER, EXPERTS, TOPK, THREADS)
rms = dict(spec, name=spec["name"] + "_rms", norm_eps=EPS)
res = dict(rms, name=rms["name"] + "_res", residual=True)
ee = jit([rms, res])
x = (3 * rng.standard_normal((1, HID))).astype(numpy.float32)
h = rng.standard_normal((1, HID)).astype(numpy.float32)
(y,) = call(ee, rms["name"], 1, [2], x, wr, gu_tiles, dn_tiles, nw)
(yr,) = call(ee, res["name"], 1, [2], x, wr, gu_tiles, dn_tiles, nw, h)
print("moe residual: bit-identical", numpy.array_equal(yr, h + y))
# CHECK-NEXT: moe residual: bit-identical True


# ── attention with q_norm / k_norm ──────────────────────────────────────────


def rope(x, pos, inv_freq):
    """HF rotate_half RoPE of x [..., rows, d] at positions pos [rows]."""
    ang = pos[:, None].astype(numpy.float64) * inv_freq[None, :]
    cos = numpy.concatenate([numpy.cos(ang)] * 2, axis=1)
    sin = numpy.concatenate([numpy.sin(ang)] * 2, axis=1)
    half = x.shape[-1] // 2
    rot = numpy.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def head_norm(x, nw, d, eps):
    """RMSNorm over each head of x [m, heads * d]."""
    m = x.shape[0]
    h = x.reshape(m, -1, d).astype(numpy.float64)
    h = h / numpy.sqrt((h**2).mean(axis=2, keepdims=True) + eps) * nw
    return h.reshape(m, -1)


def f16(x):
    return x.astype(numpy.float16).astype(numpy.float64)


def ref_attention(q, k, v, kc, vc, start, inv_freq, scale, ime=False):
    d = kc.shape[3]
    m = q.shape[0]
    q = q.reshape(m, -1, d).transpose(1, 0, 2)
    k = k.reshape(m, -1, d).transpose(1, 0, 2)
    v = v.reshape(m, -1, d).transpose(1, 0, 2)
    group = q.shape[0] // k.shape[0]
    pos = numpy.arange(start, start + m)
    kc, vc = kc.astype(numpy.float64), vc.astype(numpy.float64)
    kc[0, :, start : start + m] = rope(k, pos, inv_freq)
    vc[0, :, start : start + m] = v
    qr = rope(q, pos, inv_freq)
    out = numpy.zeros(q.shape)
    for hh in range(q.shape[0]):
        for i in range(m):
            keys = kc[0, hh // group, : start + i + 1]
            vals = vc[0, hh // group, : start + i + 1]
            if ime:
                s = f16(keys) @ f16(qr[hh, i] * scale)
                p = numpy.exp(s - s.max())
                out[hh, i] = f16(p) @ f16(vals) / p.sum()
                continue
            s = keys @ qr[hh, i] * scale
            p = numpy.exp(s - s.max())
            out[hh, i] = p @ vals / p.sum()
    return out.transpose(1, 0, 2).reshape(m, -1), kc, vc


H, KVH, D, EPS = 4, 2, 128, 1e-6
inv_freq = (5e6 ** (-numpy.arange(0, D, 2) / D)).astype(numpy.float32)
for m, start, ctx, ime in (
    (1, 0, 96, False),
    (1, 70, 96, False),
    (32, 41, 96, False),
    (64, 41, 128, True),
):
    spec = {
        "name": f"k3_attn_m{m}_s{start}_qkn",
        "kind": "attn",
        "m": m,
        "heads": H,
        "kv_heads": KVH,
        "dim": D,
        "scale": D**-0.5,
        "ctx": ctx,
        "qk_norm_eps": EPS,
    }
    if ime:
        spec.update(name=spec["name"] + "_ime", ime=True)
    ee = jit([spec], emulate_ime=ime)
    q = rng.standard_normal((m, H * D)).astype(numpy.float32)
    k = rng.standard_normal((m, KVH * D)).astype(numpy.float32)
    v = rng.standard_normal((m, KVH * D)).astype(numpy.float32)
    qn = (1 + 0.1 * rng.standard_normal(D)).astype(numpy.float32)
    kn = (1 + 0.1 * rng.standard_normal(D)).astype(numpy.float32)
    kc = numpy.zeros((1, KVH, ctx, D), numpy.float32)
    vc = numpy.zeros((1, KVH, ctx, D), numpy.float32)
    kc[0, :, :start] = rng.standard_normal((KVH, start, D))
    vc[0, :, :start] = rng.standard_normal((KVH, start, D))
    want, want_k, want_v = ref_attention(
        head_norm(q, qn, D, EPS),
        head_norm(k, kn, D, EPS),
        v,
        kc,
        vc,
        start,
        inv_freq,
        D**-0.5,
        ime,
    )
    pos = numpy.array([start], numpy.int64)
    o, _, _, _ = call(
        ee,
        spec["name"],
        4,
        [2, 3, 4, 4],
        q,
        k,
        v,
        kc,
        vc,
        pos,
        inv_freq,
        qn,
        kn,
    )
    label = f"attention qk_norm m{m} start {start}" + (" ime" if ime else "")
    report(label, o, want, 5e-4 if ime else 1e-5)
    print(f"  k cache: {numpy.allclose(kc, want_k, atol=1e-5)}")
# CHECK: attention qk_norm m1 start 0: (1, 512) error ok
# CHECK-NEXT: k cache: True
# CHECK-NEXT: attention qk_norm m1 start 70: (1, 512) error ok
# CHECK-NEXT: k cache: True
# CHECK-NEXT: attention qk_norm m32 start 41: (32, 512) error ok
# CHECK-NEXT: k cache: True
# CHECK-NEXT: attention qk_norm m64 start 41 ime: (64, 512) error ok
# CHECK-NEXT: k cache: True
