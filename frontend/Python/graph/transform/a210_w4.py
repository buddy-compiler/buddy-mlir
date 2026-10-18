# ===- a210_w4.py --------------------------------------------------------------
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
# Int4 group-quantized LLM kernels for RVV with VLEN 128, variant "w4g32"
# with "kernels": "a210" (docs/A210DeepSeekR1.md): the Zhihe A210, whose
# 8 cores are 4 XuanTie C908 (CPUs 0-3) and 4 XuanTie C920 (CPUs 4-7).
#
# The graph rewrite is the one of k3_w4.py (A210Rewriter only changes the
# weight layout and the kernel names); the kernels are this module's.
#
# Weight layout. W[K, N] is int4 with one f16 scale per (32-row group,
# column), quantized like k3_w4.quantize_q4 (llama.cpp Q4_0). Columns are
# split into tiles of TW = 16 (one e8 register at VLEN 128); every tile is
# one contiguous block, group by group:
#
#   16 rows of 16 bytes   byte j of row p: low nibble  W[g*32 + 2p,   n0 + j]
#                                          high nibble W[g*32 + 2p+1, n0 + j]
#   16 f16 scales
#
# x is quantized on the fly to int8 per 32-element group, as in k3_w4.py; the
# 32 products of a group accumulate exactly in one int16 vector per row
# (vwmacc.vx), which vfwcvt turns into f32 once per group: the sums, and the
# results, are those of the K3 kernels.
#
# Work split. The A210's cores differ: in these kernels a C920 core
# multiplies 2.5 to 3 times as fast as a C908 one, and the C908 cluster
# reads DRAM faster than the C920 one (~8 against ~3 GB/s).
# The thread pool (runtime/threadpool/BuddyThreadPool.c) pins thread i to
# the i-th CPU the process may run on, and a kernel gives thread i a share
# of its work proportional to weights[i] (spec "thread_weights", one list for
# the prefill kernels and one for decode); without weights, equal shares.
# The prefill tiles of thread i are those of its core (spec "thread_cores",
# TILE_CONFIGS).
#
# ===---------------------------------------------------------------------------

import numpy
from buddy_mlir import ir
from buddy_mlir.dialects import arith, func, llvm, math, memref, scf, vector

from . import k3_w4
from .k3_w4 import (
    PREFILL_LOGITS_FLAG,
    G,
    _addf,
    _addi,
    _alloc,
    _bcast,
    _divf,
    _divui,
    _fma,
    _Fn,
    _llvm_ptr,
    _memref,
    _mulf,
    _muli,
    _parallel,
    _read,
    _reduce,
    _silu_mul,
    _subi,
    _Types,
    _vec,
)

# The version of the packed weight layout (import_model.w4g32_layout). Bump
# the first part whenever pack_tiles changes the bytes; the second is the
# K3 version, as the values come from k3_w4.quantize_q4: a change of the
# quantization bumps the K3 version, which re-exports the A210 weights too.
W4G32_LAYOUT_VERSION = f"1.k3-{k3_w4.W4G32_LAYOUT_VERSION}"
TW = 16  # columns per weight tile: one e8 register at VLEN 128
# Prefill tiles by core: the block shape (rows of a block, which share each
# unpacked weight vector; int16 accumulators per row) and the LLVM
# scheduling model of the tile function ("tune-cpu", None: generic).
#
# Four rows and two accumulators fill the 32 vector registers (8 int16 m2
# and 4 f32 m4 vectors). On one core (the 64 x 1536 x 1536 o projection):
# C920 18.2 GOPS, 19.8 scheduled for the in-order dual-issue SiFive U74
# (LLVM has no model of the C920 of the A210; xt-c920v2: 17.4); C908 7.7,
# whose best is the generic schedule. Two rows and one accumulator gave the
# C908 8.2 in C but 6.7 from these kernels: on the in-order C908 such
# differences come from the instruction order llc happens to pick, not from
# the shape.
TILE_CONFIGS = {
    "c920": {"shape": (4, 2), "tune": "sifive-u74"},
    "c908": {"shape": (4, 2), "tune": None},
}
DEFAULT_CORE = "c920"
# Rows of a work item of a prefill kernel: a multiple of every block.
ITEM_ROWS = 4
GROUP_BYTES = G // 2 * TW + TW * 2


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def tile_bytes(k: int) -> int:
    """Bytes of one tile of a [k, *] weight."""
    return (k // G) * GROUP_BYTES


def packed_bytes(k: int, n: int) -> int:
    """Bytes of a packed [k, n] weight: as many as the K3 layout's."""
    return tile_bytes(k) * (n // TW)


def pack_tiles(q, scale) -> numpy.ndarray:
    """The tile layout of the module comment, as int8."""
    groups, _, n = q.shape
    assert n % TW == 0, n
    rows = ((q[:, 0::2, :] & 0x0F) | ((q[:, 1::2, :] & 0x0F) << 4)).astype(
        numpy.uint8
    )
    rows = rows.reshape(groups, G // 2, n // TW, TW).transpose(2, 0, 1, 3)
    sc = (
        scale.view(numpy.uint8)
        .reshape(groups, n // TW, TW * 2)
        .transpose(1, 0, 2)
    )
    out = numpy.concatenate(
        [rows.reshape(n // TW, groups, G // 2 * TW), sc], axis=2
    )
    return out.reshape(-1).view(numpy.int8)


def pack_q4(w: numpy.ndarray) -> numpy.ndarray:
    """Quantize a [K, N] float matrix into the tile layout."""
    return pack_tiles(*k3_w4.quantize_q4(w))


def pack_glu(wg: numpy.ndarray, wu: numpy.ndarray) -> numpy.ndarray:
    """Gate and up tiles interleaved: g0, u0, g1, u1, ..."""
    k, n = wg.shape
    tb = tile_bytes(k)
    g, u = pack_q4(wg), pack_q4(wu)
    tiles = numpy.stack([g.reshape(n // TW, tb), u.reshape(n // TW, tb)], 1)
    return tiles.reshape(-1)


# ---------------------------------------------------------------------------
# Kernel specs and the graph rewrite
# ---------------------------------------------------------------------------


def kernel_spec(
    kind, m, k, ns, bias, threads, weights=None, cores=None
) -> dict:
    """A matmul kernel (kinds as in k3_w4.kernel_spec); `weights`: the work
    share of each thread (None: equal); `cores`: the core of each thread, a
    key of TILE_CONFIGS (None: DEFAULT_CORE for all)."""
    name = f"a210_q4_{kind}_m{m}_k{k}_n{'_'.join(map(str, ns))}"
    if bias:
        name += "_b"
    return {
        "name": name,
        "kind": kind,
        "m": m,
        "k": k,
        "ns": list(ns),
        "bias": bias,
        "threads": threads,
        "weights": None if weights is None else list(weights),
        "cores": None if cores is None else list(cores),
    }


class A210Rewriter(k3_w4._Rewriter):
    """k3_w4's graph rewrite with this module's weight layout and kernels.
    The thread weights and cores of a rewrite are set before it
    (A210Rewriter.weights: one list for the kernels of more than one row,
    one for those of one; A210Rewriter.cores)."""

    PREFIX = "a210"
    weights = {"prefill": None, "decode": None}
    cores = None

    def pack_q4(self, w):
        return pack_q4(w), None

    def pack_glu(self, wg, wu):
        return pack_glu(wg, wu), None

    def kernel_spec(self, kind, m, k, ns, bias, ime=False):
        assert not ime, "the A210 has no matrix engine"
        w = self.weights["decode" if m == 1 else "prefill"]
        return kernel_spec(kind, m, k, ns, bias, self.threads, w, self.cores)


def a210_w4_rewrite(
    graph, m, pos_input, threads, weights=None, cores=None
) -> list:
    """k3_w4.k3_w4_rewrite with the A210 kernels; `weights`: {"prefill":
    [...], "decode": [...]} thread weights (None: equal shares); `cores`:
    the core of each thread (see kernel_spec)."""

    class Rewriter(A210Rewriter):
        pass

    Rewriter.cores = cores
    if weights:
        Rewriter.weights = {
            "prefill": weights.get("prefill"),
            "decode": weights.get("decode"),
        }
    return k3_w4.k3_w4_rewrite(graph, m, pos_input, threads, rewriter=Rewriter)


# ---------------------------------------------------------------------------
# Work split
# ---------------------------------------------------------------------------


def split_bounds(items: int, threads: int, weights=None) -> list:
    """The first item of each thread, and `items` at the end: thread i gets
    [b[i], b[i + 1]), about items * weights[i] / sum(weights)."""
    if weights is None:
        weights = [1.0] * threads
    assert len(weights) == threads, (weights, threads)
    total = float(sum(weights))
    bounds, acc = [0], 0.0
    for w in weights:
        acc += w
        bounds.append(int(round(items * acc / total)))
    bounds[-1] = items
    return bounds


def _select_bound(fn, slot, bounds):
    """bounds[slot] for an index `slot` in [0, len(bounds) - 1)."""
    value = fn.idx(bounds[-2])
    for i in range(len(bounds) - 3, -1, -1):
        is_i = arith.CmpIOp(arith.CmpIPredicate.eq, slot, fn.idx(i)).result
        value = arith.SelectOp(is_i, fn.idx(bounds[i]), value).result
    return value


def _thread_range(fn, slot, items, threads, weights):
    """[lo, hi) of thread `slot` (split_bounds)."""
    bounds = split_bounds(items, threads, weights)
    lo = _select_bound(fn, slot, bounds)
    hi = _select_bound(fn, slot, bounds[1:] + [items])
    return lo, hi


# ---------------------------------------------------------------------------
# Matmul kernels
# ---------------------------------------------------------------------------


def _act_types(ty, spec):
    """The quantized activations: int8 values and one f32 scale per group."""
    m, k = spec["m"], spec["k"]
    return (_memref([m, k], ty.i8), _memref([m, k // G], ty.f32))


def _tile_types(ty, spec):
    # w, xq, xs, tile, out, out column, first row, last row (excl.)
    t = [ty.bytes, *_act_types(ty, spec), ty.index, ty.mat, ty.index]
    t += [ty.index, ty.index]
    if spec["kind"] == "glu":
        t.append(ty.mat)  # where the gate sums wait for the up ones
    if spec["bias"]:
        t += [ty.row, ty.index]
    return t


# Row pairs of a group per step of _group_sums: one 64-byte weight load.
PAIRS_PER_STEP = 4


def _group_sums(fn, ty, a, gbase, kg, rows, chains):
    """The int16 sums of one group of a tile for each row in `rows`, the
    unpacked weights shared by the rows, PAIRS_PER_STEP row pairs (one
    64-byte load) per step, `chains` accumulators per row (alternate row
    pairs: with one, a step is a chain of 8 dependent vwmacc per row). The
    steps are a loop: fully unrolled, llc unpacked all 16 row pairs first
    and spilled them to multiply row by row."""
    v8, v16 = _vec(TW, ty.i8), _vec(TW, ty.i16)
    four = fn.const(v8, 4)
    n = len(rows)
    steps = G // 2 // PAIRS_PER_STEP

    def step(s, accs):
        wide = vector.LoadOp(
            _vec(PAIRS_PER_STEP * TW, ty.i8),
            a["w"],
            [_addi(gbase, _muli(s, fn.idx(PAIRS_PER_STEP * TW)))],
        ).result
        k0 = _addi(kg, _muli(s, fn.idx(2 * PAIRS_PER_STEP)))
        for p in range(PAIRS_PER_STEP):
            w = vector.ExtractStridedSliceOp(v8, wide, [p * TW], [TW], [1])
            w = w.result
            lo = arith.ShRSIOp(arith.ShLIOp(w, four).result, four).result
            hi = arith.ShRSIOp(w, four).result
            halves = [arith.ExtSIOp(v16, h).result for h in (lo, hi)]
            for i, row in enumerate(rows):
                c = i + n * (p % chains)
                for j, half in enumerate(halves):
                    xb = memref.LoadOp(
                        a["xq"], [row, _addi(k0, fn.idx(2 * p + j))]
                    ).result
                    xv = _bcast(v16, arith.ExtSIOp(ty.i16, xb).result)
                    accs[c] = _addi(accs[c], _muli(half, xv))
        return accs

    zero = [fn.const(v16, 0)] * (chains * n)
    loop = scf.ForOp(fn.idx(0), fn.idx(steps), fn.idx(1), zero)
    with ir.InsertionPoint(loop.body):
        accs = step(loop.induction_variable, list(loop.inner_iter_args))
        scf.YieldOp(accs)
    res = list(loop.results)
    # at most 32 products of 127 * 8 in all: no int16 overflow
    sums = res[:n]
    for c in range(1, chains):
        sums = [_addi(x, y) for x, y in zip(sums, res[c * n : (c + 1) * n])]
    return sums


def _scales(fn, ty, a, gbase):
    """The f16 weight scales of a group, in f32."""
    raw = vector.LoadOp(
        _vec(2 * TW, ty.i8), a["w"], [_addi(gbase, fn.idx(G // 2 * TW))]
    ).result
    vf = _vec(TW, ty.f32)
    return arith.ExtFOp(
        vf, vector.BitCastOp(_vec(TW, ty.f16), raw).result
    ).result


def _add_group(fn, ty, a, scale, acc16, row, g, total):
    """total + the group's sum in f32 (int16 -> f32 with vfwcvt), times the
    weight and activation scales; the operations of k3_w4._accumulate."""
    vf = _vec(TW, ty.f32)
    xs = _bcast(vf, memref.LoadOp(a["xs"], [row, g]).result)
    sx = _mulf(scale, xs)
    return _fma(arith.SIToFPOp(vf, acc16).result, sx, total)


def _accumulate(fn, ty, spec, a, wtile, rows, chains):
    """The vector<TW x f32> sums of `rows` over all groups of weight tile
    `wtile`, in registers."""
    k = spec["k"]
    vf = _vec(TW, ty.f32)
    tbase = _muli(wtile, fn.idx(tile_bytes(k)))
    loop = scf.ForOp(
        fn.idx(0), fn.idx(k // G), fn.idx(1), [fn.const(vf, 0.0)] * len(rows)
    )
    with ir.InsertionPoint(loop.body):
        g = loop.induction_variable
        gbase = _addi(tbase, _muli(g, fn.idx(GROUP_BYTES)))
        kg = _muli(g, fn.idx(G))
        accs = _group_sums(fn, ty, a, gbase, kg, rows, chains)
        scale = _scales(fn, ty, a, gbase)
        outs = [
            _add_group(fn, ty, a, scale, acc, row, g, total)
            for acc, row, total in zip(accs, rows, loop.inner_iter_args)
        ]
        scf.YieldOp(outs)
    return list(loop.results)


def _tile_rows(fn, ty, spec, a, wtile, rows, chains, dst, dcol):
    """The sums of `rows` over tile `wtile`, stored at dst[row, dcol ..]."""
    vals = _accumulate(fn, ty, spec, a, wtile, rows, chains)
    for row, v in zip(rows, vals):
        vector.StoreOp(v, dst, [row, dcol])


def _tile_name(spec, config):
    """The tile function of `spec` for a tile config (decode: one)."""
    if spec["m"] == 1:
        return f"{spec['name']}__tile"
    rows, chains = config["shape"]
    tune = config["tune"] or "generic"
    return f"{spec['name']}__tile_r{rows}a{chains}_{tune.replace('-', '_')}"


def _tile_fn(ty, spec, config):
    """One weight tile (TW output columns) for rows [r0, r1), in blocks of
    the config's shape (rows, accumulators per row) and scheduled for its
    "tune" CPU; decode: one row, two accumulators, generic."""
    m = spec["m"]
    mb, chains = (1, 2) if m == 1 else config["shape"]
    glu, bias = spec["kind"] == "glu", spec["bias"]
    names = ["w", "xq", "xs", "wt", "out", "ocol", "r0", "r1"]
    if glu:
        names.append("gsum")
    if bias:
        names += ["bias", "bcol"]
    name = _tile_name(spec, config)
    fn = _Fn(name, _tile_types(ty, spec), [], public=False)
    if m > 1 and config["tune"]:
        # kept by -convert-func-to-llvm as the llvm.func's tune_cpu
        fn.op.attributes["llvm.tune_cpu"] = ir.StringAttr.get(config["tune"])
    a = dict(zip(names, fn.args))
    vf = _vec(TW, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    with ir.InsertionPoint(fn.entry):
        loop = scf.ForOp(a["r0"], a["r1"], fn.idx(mb))
        with ir.InsertionPoint(loop.body):
            r = loop.induction_variable
            rows = [_addi(r, fn.idx(i)) for i in range(mb)]
            out, col = a["out"], a["ocol"]
            if glu:
                # gate into gsum, up into out, then silu(gate) * up
                tg = _muli(a["wt"], fn.idx(2))
                up = _addi(tg, fn.idx(1))
                _tile_rows(fn, ty, spec, a, tg, rows, chains, a["gsum"], col)
                _tile_rows(fn, ty, spec, a, up, rows, chains, out, col)
                for row in rows:
                    g = vector.LoadOp(vf, a["gsum"], [row, col]).result
                    u = vector.LoadOp(vf, out, [row, col]).result
                    val = _silu_mul(fn, ty, g, u, TW)
                    vector.StoreOp(val, out, [row, col])
            else:
                _tile_rows(fn, ty, spec, a, a["wt"], rows, chains, out, col)
                if bias:
                    b = _read(vf, a["bias"], [a["bcol"]], f0)
                    for row in rows:
                        v = vector.LoadOp(vf, out, [row, col]).result
                        vector.StoreOp(_addf(v, b), out, [row, col])
            scf.YieldOp([])
        func.ReturnOp([])


def _quantize(fn, ty, spec, a, xq, xs):
    """x (with the fused RMSNorm) -> int8 per group and its scale; one row
    serially, the rows of a chunk in parallel."""
    m, k = spec["m"], spec["k"]
    groups = k // G
    norm = spec.get("norm_eps") is not None
    vg = _vec(G, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    if m == 1:
        rloop = scf.ForOp(fn.idx(0), fn.idx(1), fn.idx(1))
        body, r = rloop.body, rloop.induction_variable
    else:
        body, (r,) = _parallel(fn, [fn.idx(m)])
    with ir.InsertionPoint(body):
        if norm:
            ss = scf.ForOp(
                fn.idx(0), fn.idx(groups), fn.idx(1), [fn.const(vg, 0.0)]
            )
            with ir.InsertionPoint(ss.body):
                kk = _muli(ss.induction_variable, fn.idx(G))
                xv = _read(vg, a["x"], [r, kk], f0)
                scf.YieldOp([_fma(xv, xv, ss.inner_iter_args[0])])
            total = _reduce(ty.f32, "add", ss.results[0], reassoc=True)
            mean = _divf(total, fn.const(ty.f32, float(k)))
            rinv = math.RsqrtOp(
                _addf(mean, fn.const(ty.f32, spec["norm_eps"]))
            ).result
            rinv = _bcast(vg, rinv)
        gl = scf.ForOp(fn.idx(0), fn.idx(groups), fn.idx(1))
        with ir.InsertionPoint(gl.body):
            g = gl.induction_variable
            kk = _muli(g, fn.idx(G))
            xv = _read(vg, a["x"], [r, kk], f0)
            if norm:
                xv = _mulf(_mulf(xv, rinv), _read(vg, a["nw"], [kk], f0))
            mx = _reduce(ty.f32, "maximumf", math.AbsFOp(xv).result)
            c127 = fn.const(ty.f32, 127.0)
            nonzero = arith.CmpFOp(arith.CmpFPredicate.OGT, mx, f0).result
            inv = arith.SelectOp(nonzero, _divf(c127, mx), f0).result
            xr = math.RoundEvenOp(_mulf(xv, _bcast(vg, inv))).result
            xr = arith.MinimumFOp(xr, fn.const(vg, 127.0)).result
            xr = arith.MaximumFOp(xr, fn.const(vg, -127.0)).result
            xi = arith.FPToSIOp(_vec(G, ty.i8), xr).result
            vector.StoreOp(xi, xq, [r, kk])
            memref.StoreOp(_divf(mx, c127), xs, [r, g])
            scf.YieldOp([])
        if m == 1:
            scf.YieldOp([])
        else:
            scf.ReduceOp([], 0)


def _matmul_fn(ty, spec):
    """One matmul kernel: quantize x, then the tiles on all threads, thread
    i taking a range of (tile, block of ITEM_ROWS rows) items in tile order
    with the tile function of its core."""
    m, k, ns, kind = spec["m"], spec["k"], spec["ns"], spec["kind"]
    assert k % G == 0 and all(n % TW == 0 for n in ns), spec
    assert m == 1 or m % ITEM_ROWS == 0, spec
    threads = spec["threads"]
    norm = spec.get("norm_eps") is not None
    names = (
        ["x", "w"]
        + (["bias"] if spec["bias"] else [])
        + (["nw"] if norm else [])
    )
    types = {"x": ty.mat, "w": ty.bytes, "bias": ty.row, "nw": ty.row}
    outs = [_memref([m, n], ty.f32) for n in ns]
    fn = _Fn(spec["name"], [types[n] for n in names], outs)
    a = dict(zip(names, fn.args))
    with ir.InsertionPoint(fn.entry):
        ys = [_alloc(t) for t in outs]
    body_ip = ir.InsertionPoint(fn.entry)
    if spec.get("prefill_logits"):
        with ir.InsertionPoint(fn.entry):
            flag = llvm.LoadOp(
                ty.i32,
                llvm.AddressOfOp(_llvm_ptr(), PREFILL_LOGITS_FLAG).result,
            ).result
            needed = arith.CmpIOp(
                arith.CmpIPredicate.ne, flag, fn.const(ty.i32, 0)
            ).result
            cond = scf.IfOp(needed)
        body_ip = ir.InsertionPoint(cond.then_block)
    with body_ip:
        xq_t, xs_t = _act_types(ty, spec)
        xq, xs = _alloc(xq_t), _alloc(xs_t)
        _quantize(fn, ty, spec, a, xq, xs)
        glu = kind == "glu"
        gsum = _alloc(_memref([m, ns[0]], ty.f32)) if glu else None
        tiles = ns[0] // TW if glu else sum(ns) // TW
        blocks = 1 if m == 1 else m // ITEM_ROWS
        body, (slot,) = _parallel(fn, [fn.idx(threads)])
        with ir.InsertionPoint(body):
            lo, hi = _thread_range(
                fn, slot, tiles * blocks, threads, spec.get("weights")
            )
            nb = fn.idx(blocks)
            t0 = _divui(lo, nb)
            t1 = _divui(_addi(hi, fn.idx(blocks - 1)), nb)
            tl = scf.ForOp(t0, t1, fn.idx(1))
            with ir.InsertionPoint(tl.body):
                t = tl.induction_variable
                first = _muli(t, nb)
                b0 = _subi(arith.MaxUIOp(lo, first).result, first)
                b1 = _subi(arith.MinUIOp(hi, _addi(first, nb)).result, first)
                r0 = _muli(b0, fn.idx(ITEM_ROWS if m > 1 else 1))
                r1 = _muli(b1, fn.idx(ITEM_ROWS if m > 1 else 1))
                col = _muli(t, fn.idx(TW))
                dsts = [memref.CastOp(ty.mat, y).result for y in ys]
                # multi: the output the tile belongs to, and its column there
                dst = dsts[-1]
                dcol = _subi(col, fn.idx(sum(ns[:-1])))
                for i in range(len(ns) - 2, -1, -1):
                    inside = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        t,
                        fn.idx(sum(ns[: i + 1]) // TW),
                    ).result
                    dst = arith.SelectOp(inside, dsts[i], dst).result
                    dcol = arith.SelectOp(
                        inside, _subi(col, fn.idx(sum(ns[:i]))), dcol
                    ).result
                args = [a["w"], xq, xs, t, dst, dcol, r0, r1]
                if glu:
                    args.append(memref.CastOp(ty.mat, gsum).result)
                if spec["bias"]:
                    args += [a["bias"], col]
                _call_tile(fn, spec, slot, args)
                scf.YieldOp([])
            scf.ReduceOp([], 0)
        if gsum is not None:
            memref.DeallocOp(gsum)
        memref.DeallocOp(xq)
        memref.DeallocOp(xs)
        if spec.get("prefill_logits"):
            scf.YieldOp([])
    with ir.InsertionPoint(fn.entry):
        func.ReturnOp(ys)


def _thread_configs(spec) -> list:
    """The tile config of each thread of a prefill kernel."""
    cores = spec.get("cores") or [DEFAULT_CORE] * spec["threads"]
    assert len(cores) == spec["threads"], spec
    return [TILE_CONFIGS[c] for c in cores]


def _spec_configs(spec) -> list:
    """The tile configs a kernel uses, each once (decode: one)."""
    if spec["m"] == 1:
        return [TILE_CONFIGS[DEFAULT_CORE]]
    configs = []
    for config in _thread_configs(spec):
        assert ITEM_ROWS % config["shape"][0] == 0, config
        if config not in configs:
            configs.append(config)
    return configs


def _call_tile(fn, spec, slot, args, configs=None):
    """Calls the tile function of thread `slot`'s core: if the thread has
    configs[0], ..., else if configs[1], ..., else the last."""
    configs = _spec_configs(spec) if configs is None else configs
    if len(configs) == 1:
        func.CallOp([], _tile_name(spec, configs[0]), args)
        return
    hit = None
    for i, config in enumerate(_thread_configs(spec)):
        if config == configs[0]:
            eq = arith.CmpIOp(arith.CmpIPredicate.eq, slot, fn.idx(i)).result
            hit = eq if hit is None else arith.OrIOp(hit, eq).result
    cond = scf.IfOp(hit, has_else=True)
    with ir.InsertionPoint(cond.then_block):
        func.CallOp([], _tile_name(spec, configs[0]), args)
        scf.YieldOp([])
    with ir.InsertionPoint(cond.else_block):
        _call_tile(fn, spec, slot, args, configs[1:])
        scf.YieldOp([])


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def build_kernels(specs) -> ir.Module:
    """The module of the kernels `specs` (a210_w4_rewrite) describe, in the
    active context; verified. The attention kernels are still those of
    k3_w4.py."""
    ty = _Types()
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        if any(s.get("prefill_logits") for s in specs):
            k3_w4._prefill_logits_flag(ty)
        for s in specs:
            if s["kind"] == "attn":
                if s["m"] == 1:
                    k3_w4._attn_decode_fn(ty, s)
                else:
                    k3_w4._attn_prefill_fn(ty, s)
            else:
                _matmul_fn(ty, s)
                for config in _spec_configs(s):
                    _tile_fn(ty, s, config)
    if not module.operation.verify():
        raise RuntimeError("a210_w4: the generated kernels do not verify")
    return module


def gen_kernels(specs) -> str:
    """build_kernels() in a context of its own, as MLIR text."""
    with ir.Context(), ir.Location.unknown():
        return str(build_kernels(specs))
