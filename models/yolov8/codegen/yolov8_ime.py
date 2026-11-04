# ===- yolov8_ime.py - YOLOv8 on the SpacemiT K3 matrix engine -------------===//
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
# ===----------------------------------------------------------------------===//
#
# The convolutions of YOLOv8 on the IME of the SpacemiT K3 A100 cores
# (smt.vfwmadot: 8 x 8 fp16 blocks multiplied into f32, ime.intr.vfmadot).
#
# The graph rewrite (rewrite_graph) replaces every convolution of the
# backbone, the neck and the head branches (with its SiLU, and the residual
# add of a bottleneck) by a call of a kernel generated here (gen_kernels);
# the SPPF max pools, the nearest upsamplings, the channel concatenations
# and the head decode become kernels too. Between the kernels an activation
# of C channels and H x W pixels is fp16 in a channel-blocked, zero-padded
# layout:
#
#   [ceil(C / 8)][H + 2 + slack][W + 2][8]
#
# a block of 8 channels, a halo of one zero pixel on every side (the padding
# of the 3 x 3 convolutions) and `slack` zero rows at the end, which the
# tiles of the last row may read past their row (slack_rows). 8 pixels of a
# row times the 8 channels of a block are 128 contiguous bytes, one A
# operand of smt.vfwmadot. Channel splits are views of whole blocks.
#
# The input of a stride-2 convolution is first split by column parity
# (split2_kernel: [blocks][rows][2][(W + 3) / 2][8], the even then the odd
# padded columns of each row), so that the 8 pixels of every filter tap are
# contiguous too: on the A100 an indexed or strided vector load costs 10 to
# 40 times a contiguous one. The first convolution (3 input channels)
# reads the space-to-depth of the image (s2d_kernel, 12 of 16 channels) as
# a 2 x 2 convolution of stride 1. The last convolution of each head branch computes
# transposed tiles (row = channel), so that its NCHW f32 stores write 8
# contiguous pixels per channel.
#
# A kernel tile is Mb blocks of 8 pixels of an output row times Nb blocks of
# 8 output channels (Mb * Nb <= 8, 2 * Mb * Nb accumulator registers). Per K
# step (8 input channels at one filter tap) it loads the A operands of its
# Mb pixel blocks (one load: they are contiguous) and the B operands of its
# Nb channel blocks (one load: the weights are packed in the order the tile
# reads them), and runs Mb * Nb smt.vfwmadot. The epilogue adds the bias,
# applies SiLU and the residual in f32 and stores fp16, or for the last
# convolution of a head branch f32 in the NCHW layout the decode reads.
#
# The weights are packed to fp16 here (pack_conv) and stored in the f32
# parameter pack, two fp16 per f32 element; the kernels read their bytes.
#
# The kernels are written for the A100 VLEN of 1024 (vscale 16):
# vector<[4]xf16> is one register of 64 fp16 (an 8 x 8 operand) and
# vector<[4]xf32> a register pair of 64 f32 (an 8 x 8 accumulator).
#
# ===----------------------------------------------------------------------===//

import numpy
import torch
from buddy.compiler.graph import TensorDType
from buddy.compiler.graph.operation import (
    AddOp,
    CallExternalOp,
    CatOp,
    Conv2dOp,
    GetItemOp,
    LowMemoryMaxPoolWithOffsetsOp,
    MulOp,
    OutputOp,
    PlaceholderOp,
    SigmoidOp,
    SplitOp,
    UnsafeIndexOp,
)

# Pixels a tile may read past the end of its row: up to 8 pixel blocks plus
# the filter width (stride 2: in a column-parity half of twice the row).
SLACK_PX = 2 * 8 * 4 + 2

REG_MAX = 16  # the DFL bins of a box side

# The weights of the convolutions that re-read them most (their pixel
# tiles) are copied into the TCM of every core pair of the K3 A100 cores
# (768 KiB each, runtime/spacemit/BuddySpacemitTcm.c) at the first call: a
# core loads 1 KiB from its pair's TCM in ~9 ns whatever the others do,
# from the cache in ~40 ns when 3 or 4 cores of its cluster load.
TCM_BYTES = 704 * 1024
TCM_REGIONS = 4


def slack_rows(w: int) -> int:
    """Zero rows after the bottom halo of an activation of width w."""
    return -(-SLACK_PX // (w + 2))


def act_shape(c: int, h: int, w: int) -> list:
    """The blocked fp16 layout of an activation of c channels, h x w."""
    return [-(-c // 8), h + 2 + slack_rows(w), w + 2, 8]


def phase_shape(c: int, h: int, w: int) -> list:
    """act_shape with each padded row split into its even and odd columns."""
    cb, rows, _, _ = act_shape(c, h, w)
    return [cb, rows, 2, (w + 3) // 2, 8]


def _pow2_at_most(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


class Conv:
    """One convolution kernel: shapes, tile and epilogue."""

    def __init__(
        self, name, ic, oc, k, s, hin, win, silu, residual=False, nchw=False
    ):
        assert k in (1, 2, 3) and s in (1, 2) and ic % 8 == 0, (name, ic, k, s)
        assert s == 1 or k == 3, (name, k, s)
        self.name, self.ic, self.oc, self.k, self.s = name, ic, oc, k, s
        self.hin, self.win = hin, win
        # k == 2: the first convolution on the space-to-depth image, taps at
        # the padded pixels (y + ky, x + kx), i.e. input pixels y - 1 and y.
        pad = k // 2
        self.hout = (hin + 2 * pad - k) // s + 1 - (k == 2)
        self.wout = (win + 2 * pad - k) // s + 1 - (k == 2)
        self.silu, self.residual, self.nchw = silu, residual, nchw
        # Offset of the weights in the TCM (set by the rewrite), or None.
        self.tcm_off = None
        self.icb, self.ocb = ic // 8, -(-oc // 8)
        # The tile: Mb pixel blocks x Nb channel blocks, Mb * Nb = 8 where the
        # image is wide enough. Per K step it loads (Mb + Nb) * 128 bytes
        # for Mb * Nb smt.vfwmadot: 2 x 4 loads 768 bytes, 1 x 8 1152. With
        # the operands of the next K step loaded ahead (2 * (Mb + Nb)
        # registers) and 2 * Mb * Nb accumulators, 2 x 4 takes 28 of the 32
        # vector registers.
        mb_max = min(4, _pow2_at_most(-(-self.wout // 8)))
        self._tile(mb_max)
        # Flat tiles: on narrow images (10, 20 pixels: tiles of rows waste
        # 38% of their pixels), the stride-1 convolutions with blocked
        # outputs tile the padded image as one sequence of pixels across its
        # rows (the outputs on the halo columns stored as zeros). From 40
        # pixels on, the masks of the halo columns cost more than the waste
        # (measured).
        self.flat = s == 1 and not nchw and self.wout <= 20
        if self.flat:
            self._tile(4)
        # Offset of the filter window in the padded input.
        self.off = 1 - pad

    def _tile(self, mb_max):
        if self.ocb >= 4 and mb_max >= 2:
            self.nb, self.mb = 4, 2
        elif self.ocb >= 8:
            self.nb, self.mb = 8, 1
        else:
            self.nb = _pow2_at_most(self.ocb)
            self.mb = min(mb_max, 8 // self.nb)
        self.nbl = self.nb
        # The last tile may be partial (51 channels: 7 blocks): its weights
        # and bias are zero past ocb, its stores of those blocks skipped.
        self.nt = -(-self.ocb // self.nb)
        self.xt = -(-self.wout // (8 * self.mb))
        # Flat tiles over the h x (w + 2) output positions.
        self.qt = -(-self.hout * (self.wout + 2) // (8 * self.mb))

    @property
    def in_shape(self):
        if self.s == 2:
            return phase_shape(self.ic, self.hin, self.win)
        return act_shape(self.ic, self.hin, self.win)

    @property
    def out_shape(self):
        if self.nchw:
            return [1, self.oc, self.hout, self.wout]
        return act_shape(self.oc, self.hout, self.wout)

    @property
    def ksteps(self):
        return self.icb * self.k * self.k

    def w_elems(self):
        """f32 elements of the packed weights (two fp16 each)."""
        return self.nt * self.ksteps * self.nbl * 64 // 2

    def b_elems(self):
        return self.nt * self.nb * 64


def pack_conv(conv: Conv, w: numpy.ndarray, b: numpy.ndarray):
    """The packed fp16 weights (as f32 elements) and the expanded f32 bias.

    Weights: [nt][cb][ky][kx][nbl][c][k] for output channel
    (nt * Nb + nb) * 8 + c and input channel cb * 8 + k (the blocks past Nb
    zero): the B operands of one K step of a tile are contiguous, and a tile
    streams its K steps. Bias: [output block][r][c] = b[block * 8 + c], the
    layout of an accumulator (row r = pixel, column c = channel); for the
    NCHW outputs, whose tiles are transposed, b[block * 8 + r]."""
    oc, ic, kh, kw = w.shape
    assert kh == kw == conv.k and oc == conv.oc and ic <= conv.ic, w.shape
    wp = numpy.zeros((conv.nt, conv.nbl * 8, conv.ic, kh, kw), numpy.float32)
    wf = numpy.zeros((conv.nt * conv.nb * 8, conv.ic, kh, kw), numpy.float32)
    wf[:oc, :ic] = w
    wp[:, : conv.nb * 8] = wf.reshape(conv.nt, conv.nb * 8, conv.ic, kh, kw)
    # [nt, nbl, c, cb, k, ky, kx] -> [nt, cb, ky, kx, nbl, c, k]
    wp = wp.reshape(conv.nt, conv.nbl, 8, conv.icb, 8, kh, kw)
    wp = wp.transpose(0, 3, 5, 6, 1, 2, 4)
    packed = numpy.ascontiguousarray(wp).astype(numpy.float16)
    assert packed.size == 2 * conv.w_elems()
    nblk = conv.nt * conv.nb
    bp = numpy.zeros((nblk * 8,), numpy.float32)
    bp[:oc] = b
    bp = bp.reshape(nblk, 1, 8)
    if conv.nchw:
        bp = bp.transpose(0, 2, 1)  # transposed tiles: row r = channel
    bp = numpy.broadcast_to(bp, (nblk, 8, 8)).reshape(-1)
    return packed.view(numpy.float32).copy(), numpy.ascontiguousarray(bp)


# ---------------------------------------------------------------------------
# MLIR text of the kernels
# ---------------------------------------------------------------------------

F16 = "f16"
VH = "vector<[4]xf16>"  # one register: an 8 x 8 fp16 operand
VF = "vector<[4]xf32>"  # a register pair: an 8 x 8 f32 accumulator
VI = "vector<[4]xi64>"  # 64 byte offsets
VM = "vector<[4]xi1>"
PTR = "!llvm.ptr"


def _memref(shape, elt, dynamic_layout=True):
    dims = "x".join(str(d) for d in shape)
    if not dynamic_layout:
        return f"memref<{dims}x{elt}>"
    strides = ", ".join("?" for _ in shape)
    return f"memref<{dims}x{elt}, strided<[{strides}], offset: ?>>"


def _lit(value, ty):
    if ty in ("f16", "f32"):
        return value if isinstance(value, str) else repr(float(value))
    if ty == "i1":
        return "true" if value else "false"
    return str(int(value))


class _Body:
    """The lines of one function body, with fresh SSA names and constants
    hoisted to the entry block."""

    def __init__(self):
        self.lines = []
        self.consts = {}
        self.const_lines = []
        self.n = 0
        self.indent = 2

    def fresh(self, hint="v"):
        self.n += 1
        return f"%{hint}{self.n}"

    def emit(self, text):
        self.lines.append(" " * self.indent + text)

    def op(self, text, hint="v"):
        name = self.fresh(hint)
        self.emit(f"{name} = {text}")
        return name

    def const(self, value, ty):
        """A constant: a scalar, a splat vector, or (a list) a dense one."""
        key = (repr(value), ty)
        if key not in self.consts:
            name = f"%k{len(self.consts)}"
            if ty.startswith("vector"):
                elt = ty.rsplit("x", 1)[1].rstrip(">")
                if isinstance(value, list):
                    lit = "[" + ", ".join(_lit(v, elt) for v in value) + "]"
                else:
                    lit = _lit(value, elt)
                text = f"dense<{lit}> : {ty}"
            else:
                text = f"{_lit(value, ty)} : {ty}"
            self.const_lines.append(f"  {name} = arith.constant {text}")
            self.consts[key] = name
        return self.consts[key]

    def const_poison(self, ty):
        key = ("poison", ty)
        if key not in self.consts:
            name = f"%k{len(self.consts)}"
            self.const_lines.append(f"  {name} = llvm.mlir.poison : {ty}")
            self.consts[key] = name
        return self.consts[key]

    # i64 / index arithmetic -------------------------------------------------
    def i(self, v):
        return self.const(int(v), "i64") if isinstance(v, int) else v

    def x(self, v):
        return self.const(int(v), "index") if isinstance(v, int) else v

    def arith(self, opname, a, b, ty="i64"):
        if ty == "i64":
            a, b = self.i(a), self.i(b)
        elif ty == "index":
            a, b = self.x(a), self.x(b)
        return self.op(f"arith.{opname} {a}, {b} : {ty}")

    def add(self, a, b, ty="i64"):
        if isinstance(b, int) and b == 0:
            return self.i(a) if ty == "i64" else self.x(a)
        return self.arith("addi", a, b, ty)

    def sub(self, a, b):
        return self.arith("subi", a, b)

    def mul(self, a, b, ty="i64"):
        if isinstance(b, int) and b == 1:
            return self.i(a) if ty == "i64" else self.x(a)
        return self.arith("muli", a, b, ty)

    def cmp(self, pred, a, b):
        return self.op(f"arith.cmpi {pred}, {self.i(a)}, {self.i(b)} : i64")

    def to_i64(self, v):
        return self.op(f"arith.index_cast {v} : index to i64")

    def to_index(self, v):
        return self.op(f"arith.index_cast {v} : i64 to index")

    def gep(self, p, off):
        if isinstance(off, int) and off == 0:
            return p
        off = self.i(off)
        return self.op(
            f"llvm.getelementptr {p}[{off}] : ({PTR}, i64) -> {PTR}, i8", "p"
        )

    def load(self, p, ty, align=2):
        return self.op(f"llvm.load {p} <alignment = {align}> : {PTR} -> {ty}")

    def store(self, v, p, ty, align=2):
        self.emit(f"llvm.store {v}, {p} <alignment = {align}> : {ty}, {PTR}")

    def masked_load(self, p, mask, ty, mty, align=2):
        return self.op(
            f"llvm.intr.masked.load({p}, {mask}), alignment({align}) : "
            f"({PTR}, {mty}) -> {ty}"
        )

    def masked_store(self, v, p, mask, ty, mty, align=2):
        self.emit(
            f"llvm.intr.masked.store({v}, {p}, {mask}), alignment({align}) : "
            f"{ty}, {mty} into {PTR}"
        )

    def scatter(self, v, p, offs, mask, ty):
        """Unordered indexed store (vsuxei) of the lanes of v at p + offs[i]
        (bytes) where mask; llvm.masked.scatter would be ordered (vsoxei),
        several times slower on the A100."""
        self.emit(
            f'llvm.call_intrinsic "llvm.riscv.vsuxei.mask"({v}, {p}, {offs}, '
            f"{mask}, {self.i(-1)}) : ({ty}, {PTR}, {VI}, {VM}, i64) -> ()"
        )

    def prefix_mask(self, n, mty=VM):
        """The mask of the first n (i64) lanes."""
        return self.op(f"vector.create_mask {self.to_index(n)} : {mty}")

    def ptr_of(self, mem, mtype, rank, elt_bytes):
        """The address of element 0 of a memref (aligned pointer + offset)."""
        base = self.op(
            f"memref.extract_aligned_pointer_as_index {mem} : {mtype} -> index"
        )
        res = ", ".join(["index"] * (1 + 2 * rank))
        elt = mtype.split("x")[-1].split(",")[0].rstrip(">")
        name = self.fresh("md")
        self.emit(
            f"%mb{self.n}, {name}, %ms{self.n}:{rank}, %mt{self.n}:{rank} = "
            f"memref.extract_strided_metadata {mem} : {mtype} -> "
            f"memref<{elt}>, {res}"
        )
        off = self.mul(name, elt_bytes, "index")
        addr = self.to_i64(self.add(base, off, "index"))
        return self.op(f"llvm.inttoptr {addr} : i64 to {PTR}", "p")

    # structure ------------------------------------------------------------
    def open_if(self, cond):
        self.emit(f"scf.if {cond} {{")
        self.indent += 2

    def open_for(self, lo, hi, step=1):
        """An scf.for over [lo, hi); returns the i64 induction variable."""
        iv = self.fresh("i")
        self.emit(
            f"scf.for {iv} = {self.x(lo)} to {self.x(hi)} step {self.x(step)} {{"
        )
        self.indent += 2
        return self.to_i64(iv)

    def open_parallel(self, n):
        """An scf.parallel over [0, n) (the threads); returns the i64 iv."""
        iv = self.fresh("it")
        self.emit(
            f"scf.parallel ({iv}) = ({self.x(0)}) to ({self.x(n)}) step "
            f"({self.x(1)}) {{"
        )
        self.indent += 2
        return self.to_i64(iv)

    def close(self, parallel=False):
        if parallel:
            self.emit("scf.reduce")
        self.indent -= 2
        self.emit("}")

    def divmod(self, v, d):
        q = self.op(f"arith.divui {v}, {self.i(d)} : i64")
        r = self.op(f"arith.remui {v}, {self.i(d)} : i64")
        return q, r

    def text(self):
        return "\n".join(self.const_lines + self.lines)


def _func(name, args, result, body: _Body):
    sig = ", ".join(f"{n}: {t}" for n, t in args)
    return (
        f"func.func @{name}({sig}) -> {result} "
        f"attributes {{llvm.emit_c_interface}} {{\n{body.text()}\n}}\n"
    )


def _vec_types(vf):
    """(float, int32, mask) vector types of the f32 vector type vf."""
    lanes = vf[len("vector<") :].rsplit("x", 1)[0]
    return vf, f"vector<{lanes}xi32>", f"vector<{lanes}xi1>"


def _exp(f: _Body, z, vf=VF):
    """exp(z) on f32 vectors (z clamped to [-87, 87]): 2^n * p(r), p a
    degree 5 Taylor polynomial on |r| <= ln(2) / 2 (relative error 3e-6,
    the results go to fp16 or into normalized sums)."""
    _, vi, _ = _vec_types(vf)
    z = f.op(f"arith.maxnumf {z}, {f.const(-87.0, vf)} : {vf}")
    z = f.op(f"arith.minnumf {z}, {f.const(87.0, vf)} : {vf}")
    t = f.op(f"arith.mulf {z}, {f.const(1.4426950408889634, vf)} : {vf}")
    # Round to nearest even through the f32 mantissa (|t| < 2^22).
    magic = f.const(12582912.0, vf)
    nf = f.op(f"arith.addf {t}, {magic} : {vf}")
    nf = f.op(f"arith.subf {nf}, {magic} : {vf}")
    r = f.op(f"math.fma {nf}, {f.const(-0.6931471805599453, vf)}, {z} : {vf}")
    coeffs = [1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0]
    p = f.const(coeffs[0], vf)
    for c in coeffs[1:]:
        p = f.op(f"math.fma {p}, {r}, {f.const(c, vf)} : {vf}")
    n = f.op(f"arith.fptosi {nf} : {vf} to {vi}")
    n = f.op(f"arith.addi {n}, {f.const(127, vi)} : {vi}")
    n = f.op(f"arith.shli {n}, {f.const(23, vi)} : {vi}")
    sc = f.op(f"arith.bitcast {n} : {vi} to {vf}")
    return f.op(f"arith.mulf {p}, {sc} : {vf}")


def _recip(f: _Body, d, vf=VF):
    """1 / d (d >= 1 here): vfrec7 (7 bits) and two Newton steps (vfdiv
    costs 27 ns at LMUL 2 on the A100, 10 multiplies)."""
    r = f.op(
        f'llvm.call_intrinsic "llvm.riscv.vfrec7"({f.const_poison(vf)}, {d}, '
        f"{f.i(7)}, {f.i(-1)}) : ({vf}, {vf}, i64, i64) -> {vf}"
    )
    two = f.const(2.0, vf)
    for _ in range(2):
        e = f.op(
            f"math.fma {d}, {f.op(f'arith.negf {r} : {vf}')}, {two} : {vf}"
        )
        r = f.op(f"arith.mulf {r}, {e} : {vf}")
    return r


def _sigmoid(f: _Body, v, vf=VF):
    e = _exp(f, f.op(f"arith.negf {v} : {vf}"), vf)
    return _recip(f, f.op(f"arith.addf {e}, {f.const(1.0, vf)} : {vf}"), vf)


def _silu(f: _Body, v, vf=VF):
    """v * sigmoid(v)."""
    return f.op(f"arith.mulf {v}, {_sigmoid(f, v, vf)} : {vf}")


def _interleaved(f: _Body, gens):
    """Emit the independent instruction sequences of the callables gens one
    instruction of each in turn; returns their results."""
    parts, results = [], []
    for gen in gens:
        start = len(f.lines)
        results.append(gen())
        parts.append(f.lines[start:])
        del f.lines[start:]
    for i in range(max(len(p) for p in parts)):
        for p in parts:
            if i < len(p):
                f.lines.append(p[i])
    return results


def _zero_rows(f: _Body, p, first_px, last_px):
    """Zero the pixels [first_px, last_px) at p (16 bytes each), 64 at a
    time."""
    zero = f.const(0.0, "vector<[32]xf16>")
    n = last_px - first_px
    for off in range(0, n * 16, 1024):
        q = f.gep(p, first_px * 16 + off)
        if n * 16 - off >= 1024:
            f.store(zero, q, "vector<[32]xf16>")
        else:
            mty = "vector<[32]xi1>"
            mask = f.op(
                f"vector.create_mask {f.x((n * 16 - off) // 2)} : {mty}"
            )
            f.masked_store(zero, q, mask, "vector<[32]xf16>", mty)


def _border(f: _Body, out_p, blk, y, h, w, rows):
    """Zero the halo of output row y of block blk (blocked layout): its two
    halo pixels; with row 0 the top halo row, with row h - 1 the bottom halo
    and slack rows."""
    wp = w + 2
    bp = f.gep(out_p, f.mul(blk, rows * wp * 16))
    rowp = f.gep(bp, f.mul(f.add(y, 1), wp * 16))
    z8 = f.const(0.0, "vector<8xf16>")
    f.store(z8, rowp, "vector<8xf16>")
    f.store(z8, f.gep(rowp, (w + 1) * 16), "vector<8xf16>")
    f.open_if(f.cmp("eq", y, 0))
    _zero_rows(f, bp, 0, wp)
    f.close()
    f.open_if(f.cmp("eq", y, h - 1))
    _zero_rows(f, bp, (h + 1) * wp, rows * wp)
    f.close()


def _flat_epilogue(f: _Body, c: Conv, loop_res, nt, qt, q0, op, rp):
    """The epilogue of a flat tile (Conv.flat): the activation chains of two
    accumulators interleaved; the lanes of the pixels on the halo columns or
    past the last row stored as zeros (the halos of the layout); the first
    and the last tile zero the rest of the border."""
    mb, nb = c.mb, c.nb
    rows, wp = c.out_shape[1], c.out_shape[2]
    w, h = c.wout, c.hout
    vi = "vector<[4]xi32>"
    zero = f.const(0.0, VF)
    jv = f.op(f"llvm.intr.stepvector : {vi}")
    jv = f.op(f"arith.shrui {jv}, {f.const(3, vi)} : {vi}")
    valid = []
    for m in range(mb):
        # Pixels q .. q + 7 of the tile: row y0 from column x0, at most one
        # row boundary (w + 2 > 8).
        q = f.add(q0, 8 * m)
        y0, x0 = f.divmod(q, wp)
        x0v = f.op(
            f"vector.broadcast {f.op(f'arith.trunci {x0} : i64 to i32')} : "
            f"i32 to {vi}"
        )
        xv = f.op(f"arith.addi {jv}, {x0v} : {vi}")
        in1 = f.op(f"arith.cmpi ult, {xv}, {f.const(w, vi)} : {vi}")
        wr1 = f.op(f"arith.cmpi uge, {xv}, {f.const(wp, vi)} : {vi}")
        wr2 = f.op(f"arith.cmpi ult, {xv}, {f.const(wp + w, vi)} : {vi}")
        in2 = f.op(f"arith.andi {wr1}, {wr2} : {VM}")
        row1 = f.op(f"vector.broadcast {f.cmp('ult', y0, h)} : i1 to {VM}")
        row2 = f.op(f"vector.broadcast {f.cmp('ult', y0, h - 1)} : i1 to {VM}")
        in1 = f.op(f"arith.andi {in1}, {row1} : {VM}")
        in2 = f.op(f"arith.andi {in2}, {row2} : {VM}")
        valid.append(f.op(f"arith.ori {in1}, {in2} : {VM}"))
    flat = [(m, n) for m in range(mb) for n in range(nb)]
    for i in range(0, len(flat), 2):
        dsts = []
        for m, n in flat[i : i + 2]:
            blk = f.add(f.mul(nt, nb), n)
            pos = f.add(f.mul(blk, rows * wp), f.add(q0, 8 * m + wp + 1))
            dsts.append((m, n, blk, f.mul(pos, 16)))

        def chain(m, n, off):
            v = f"{loop_res}#{m * nb + n}"
            if c.silu:
                v = _silu(f, v)
            if c.residual:
                rv = f.load(f.gep(rp, off), VH)
                rv = f.op(f"arith.extf {rv} : {VH} to {VF}")
                v = f.op(f"arith.addf {v}, {rv} : {VF}")
            v = f.op(f"arith.select {valid[m]}, {v}, {zero} : {VM}, {VF}")
            return f.op(f"arith.truncf {v} : {VF} to {VH}")

        hs = _interleaved(
            f, [lambda d=d: chain(d[0], d[1], d[3]) for d in dsts]
        )
        for hv, (m, n, blk, off) in zip(hs, dsts):
            partial = c.ocb % nb and n >= c.ocb % nb
            if partial:
                # A block past ocb of the last tile: no output block.
                f.open_if(f.cmp("ult", blk, c.ocb))
            f.store(hv, f.gep(op, off), VH)
            if partial:
                f.close()
    # The top halo row and the first left halo; after the last tile, the
    # rest of the bottom halo and slack rows.
    end = c.qt * 8 * mb + wp + 1
    for first, lo, hi in ((True, 0, wp + 1), (False, end, rows * wp)):
        f.open_if(f.cmp("eq", qt, 0 if first else c.qt - 1))
        for n in range(nb):
            blk = f.add(f.mul(nt, nb), n)
            partial = c.ocb % nb and n >= c.ocb % nb
            if partial:
                f.open_if(f.cmp("ult", blk, c.ocb))
            _zero_rows(f, f.gep(op, f.mul(blk, rows * wp * 16)), lo, hi)
            if partial:
                f.close()
        f.close()


def _tcm_copy(f: _Body, c: Conv, wp):
    """Copy the weights of c into the TCM of every core pair, once per
    weight address (the global of the kernel keeps the last one)."""
    g = f.op(f"llvm.mlir.addressof @{c.name}_tcm : {PTR}", "p")
    prev = f.load(g, "i64", 8)
    src = f.op(f"llvm.ptrtoint {wp} : {PTR} to i64")
    f.open_if(f.cmp("ne", prev, src))
    for r in range(TCM_REGIONS):
        d = f.op(
            f"func.call @buddy_spacemit_tcm_pair({f.i(r)}, {f.i(TCM_REGIONS)},"
            f" {f.i(TCM_BYTES)}) : (i64, i64, i64) -> i64"
        )
        f.open_if(f.cmp("ne", d, 0))
        dst = f.op(f"llvm.inttoptr {f.add(d, c.tcm_off)} : i64 to {PTR}", "p")
        f.emit(
            f'"llvm.intr.memcpy"({dst}, {wp}, {f.i(c.w_elems() * 4)}) '
            f"<{{isVolatile = false}}> : ({PTR}, {PTR}, i64) -> ()"
        )
        f.close()
    f.store(src, g, "i64", 8)
    f.close()


def conv_kernel(c: Conv) -> str:
    f = _Body()
    xt_in = _memref(c.in_shape, F16)
    wt = _memref([c.w_elems()], "f32")
    bt = _memref([c.b_elems()], "f32")
    ot = _memref(c.out_shape, "f32" if c.nchw else F16, dynamic_layout=False)
    args = [("%x", xt_in), ("%w", wt), ("%b", bt)]
    if c.residual:
        args.append(("%r", _memref(c.out_shape, F16)))
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    xp = f.ptr_of("%x", xt_in, len(c.in_shape), 2)
    wp = f.ptr_of("%w", wt, 1, 4)
    bp = f.ptr_of("%b", bt, 1, 4)
    op = f.ptr_of("%out", ot, 4, 4 if c.nchw else 2)
    rp = f.ptr_of("%r", args[3][1], 4, 2) if c.residual else None

    mb, nb, nbl, kk = c.mb, c.nb, c.nbl, c.k * c.k
    va = f"vector<[{4 * mb}]xf16>"
    vb = f"vector<[{4 * nbl}]xf16>"
    if c.s == 1:
        rows_in, row_bytes = c.in_shape[1], c.in_shape[2] * 16
    else:
        rows_in, row_bytes = c.in_shape[1], 2 * c.in_shape[3] * 16
    blk_in = rows_in * row_bytes  # bytes per input channel block

    if c.nchw:
        # The tiles of the NCHW outputs are transposed (the weights as A,
        # the pixels as B): row r of an accumulator is a channel, its 8
        # pixels contiguous in the output. Byte offsets of the 64 lanes,
        # and their r and c. (Scattering the lanes of an untransposed tile,
        # every one into another channel plane, took 7.5 times as long.)
        step = f.op(f"llvm.intr.stepvector : {VI}")
        rr = f.op(f"arith.shrui {step}, {f.const(3, VI)} : {VI}")
        cc = f.op(f"arith.andi {step}, {f.const(7, VI)} : {VI}")
        hw = c.hout * c.wout
        so = f.op(f"arith.muli {rr}, {f.const(hw, VI)} : {VI}")
        so = f.op(f"arith.addi {so}, {cc} : {VI}")
        soff = f.op(f"arith.muli {so}, {f.const(4, VI)} : {VI}")

    if c.tcm_off is not None:
        _tcm_copy(f, c, wp)
    if c.flat:
        # Output position q = y * (w + 2) + x (x past w: the halo columns)
        # reads the padded input at q + (ky + off) * (w + 2) + kx + off.
        it = f.open_parallel(c.nt * c.qt)
        nt, qt = f.divmod(it, c.qt)
        q0 = f.mul(qt, 8 * mb)
        abase = f.gep(xp, f.mul(f.add(q0, c.off * (c.win + 3)), 16))
    else:
        it = f.open_parallel(c.nt * c.hout * c.xt)
        t, xt = f.divmod(it, c.xt)
        nt, y = f.divmod(t, c.hout)
        x0 = f.mul(xt, 8 * mb)
        # Input row and column of the tile's first output pixel at tap
        # (0, 0).
        arow = f.add(f.mul(y, c.s), c.off)
        if c.s == 1:
            abase = f.gep(
                xp,
                f.add(f.mul(arow, row_bytes), f.mul(f.add(x0, c.off), 16)),
            )
        else:
            abase = f.gep(xp, f.add(f.mul(arow, row_bytes), f.mul(x0, 16)))
    wstep = nbl * 128  # bytes of the B operands of one K step
    wsrc = wp
    if c.tcm_off is not None:
        # The copy of the weights in the TCM of this core's pair, if any.
        t = f.op(
            f"func.call @buddy_spacemit_tcm_here({f.i(TCM_REGIONS)}, "
            f"{f.i(TCM_BYTES)}) : (i64, i64) -> i64"
        )
        use = f.cmp("ne", t, 0)
        wpi = f.op(f"llvm.ptrtoint {wp} : {PTR} to i64")
        wt_ = f.op(f"arith.select {use}, {f.add(t, c.tcm_off)}, {wpi} : i64")
        wsrc = f.op(f"llvm.inttoptr {wt_} : i64 to {PTR}", "p")
    bbase = f.gep(wsrc, f.mul(nt, c.icb * kk * wstep))

    def tap_off(kp):
        ky, kx = divmod(kp, c.k)
        if c.s == 1:
            return ky * row_bytes + kx * 16
        # Tap kx of output x is padded column 2x + kx: parity kx & 1, entry
        # x + kx // 2 of that half of the row.
        return ky * row_bytes + (kx & 1) * (row_bytes // 2) + (kx >> 1) * 16

    def load_step(cbi, kp):
        """The A and B operands of K step (cbi, kp)."""
        ab = f.gep(abase, f.add(f.mul(cbi, blk_in), tap_off(kp)))
        bb = f.gep(bbase, f.add(f.mul(cbi, kk * wstep), kp * wstep))
        return f.load(ab, va), f.load(bb, vb)

    # Software pipelined: the operands of the next K step are loaded before
    # the smt.vfwmadot of the current one (the in-order A100 would wait for
    # each load), carried across iterations; the last step loads a valid
    # but unused one.
    # The accumulators start from the bias (the loads of the epilogue would
    # stall the in-order core).
    biases = [
        f.load(f.gep(bp, f.mul(f.add(f.mul(nt, nb), n), 256)), VF, 4)
        for n in range(nb)
    ]
    a0, b0 = load_step(f.i(0), 0)
    accs = [f"%acc{m}_{n}" for m in range(mb) for n in range(nb)]
    loop_res = f.fresh("acc")
    cb = f.fresh("cb")
    iters = ", ".join(
        [f"%acc{m}_{n} = {biases[n]}" for m in range(mb) for n in range(nb)]
        + [f"%an = {a0}", f"%bn = {b0}"]
    )
    types = ", ".join([VF] * len(accs) + [va, vb])
    f.emit(
        f"{loop_res}:{len(accs) + 2} = scf.for {cb} = {f.x(0)} to "
        f"{f.x(c.icb)} step {f.x(1)} iter_args({iters}) -> ({types}) {{"
    )
    f.indent += 2
    cbi = f.to_i64(cb)
    cur = list(accs)
    av, bv = "%an", "%bn"
    for kp in range(kk):
        if kp + 1 < kk:
            nxt = load_step(cbi, kp + 1)
        else:
            cbn = f.op(f"arith.addi {cbi}, {f.i(1)} : i64")
            cbn = f.op(f"arith.minui {cbn}, {f.i(c.icb - 1)} : i64")
            nxt = load_step(cbn, 0)
        a_ops = [
            av
            if mb == 1
            else f.op(f"vector.scalable.extract {av}[{4 * m}] : {VH} from {va}")
            for m in range(mb)
        ]
        b_ops = [
            bv
            if nbl == 1
            else f.op(f"vector.scalable.extract {bv}[{4 * n}] : {VH} from {vb}")
            for n in range(nb)
        ]
        cur = [
            f.op(
                f'"ime.intr.vfmadot"({cur[m * nb + n]}, '
                + (
                    f"{b_ops[n]}, {a_ops[m]}"
                    if c.nchw
                    else f"{a_ops[m]}, {b_ops[n]}"
                )
                + f") : ({VF}, {VH}, {VH}) -> {VF}"
            )
            for m in range(mb)
            for n in range(nb)
        ]
        av, bv = nxt
    f.emit(f"scf.yield {', '.join(cur + [av, bv])} : {types}")
    f.indent -= 2
    f.emit("}")

    if c.flat:
        _flat_epilogue(f, c, loop_res, nt, qt, q0, op, rp)
        f.close(parallel=True)
        f.emit(f"return %out : {ot}")
        return _func(c.name, args, ot, f)
    full = c.wout % (8 * mb) == 0
    lefts = []
    for m in range(mb):
        px0 = f.add(x0, 8 * m)
        left = None
        if not full or c.nchw:
            left = f.op(f"arith.subi {f.i(c.wout)}, {px0} : i64")
            left = f.op(f"arith.maxsi {left}, {f.i(0)} : i64")
            left = f.op(f"arith.minsi {left}, {f.i(8)} : i64")
        lefts.append((px0, left))
    if c.nchw:
        # No activation: scatter each accumulator.
        for m in range(mb):
            px0, left = lefts[m]
            for n in range(nb):
                blk = f.add(f.mul(nt, nb), n)
                ocl = f.op(f"arith.subi {f.i(c.oc)}, {f.mul(blk, 8)} : i64")
                lv = f.op(f"vector.broadcast {left} : i64 to {VI}")
                ov = f.op(f"vector.broadcast {ocl} : i64 to {VI}")
                m1 = f.op(f"arith.cmpi slt, {cc}, {lv} : {VI}")
                m2 = f.op(f"arith.cmpi slt, {rr}, {ov} : {VI}")
                mask = f.op(f"arith.andi {m1}, {m2} : {VM}")
                base = f.add(f.mul(blk, 8 * hw), f.add(f.mul(y, c.wout), px0))
                f.scatter(
                    f"{loop_res}#{m * nb + n}",
                    f.gep(op, f.mul(base, 4)),
                    soff,
                    mask,
                    VF,
                )
    else:
        rout, wpout = c.out_shape[1], c.out_shape[2]
        # The activation chains of two accumulators interleaved, one
        # instruction of each in turn: the in-order core issues one while the
        # other waits for a result (at LMUL 4 as one chain, the pairs spilled
        # the other accumulators).
        flat = [(m, n) for m in range(mb) for n in range(nb)]
        for i in range(0, len(flat), 2):
            grp = flat[i : i + 2]
            dsts = []
            for m, n in grp:
                blk = f.add(f.mul(nt, nb), n)
                pix = f.add(
                    f.mul(f.add(f.mul(blk, rout), f.add(y, 1)), wpout),
                    f.add(lefts[m][0], 1),
                )
                dsts.append((m, n, blk, f.mul(pix, 16)))

            def chain(m, n, off):
                v = f"{loop_res}#{m * nb + n}"
                if c.silu:
                    v = _silu(f, v)
                if c.residual:
                    rv = f.load(f.gep(rp, off), VH)
                    rv = f.op(f"arith.extf {rv} : {VH} to {VF}")
                    v = f.op(f"arith.addf {v}, {rv} : {VF}")
                return f.op(f"arith.truncf {v} : {VF} to {VH}")

            hs = _interleaved(
                f, [lambda d=d: chain(d[0], d[1], d[3]) for d in dsts]
            )
            for h, (m, n, blk, off) in zip(hs, dsts):
                partial = c.ocb % nb and n >= c.ocb % nb
                if partial:
                    # A block past ocb of the last tile: no output block.
                    f.open_if(f.cmp("ult", blk, c.ocb))
                dst = f.gep(op, off)
                if full:
                    f.store(h, dst, VH)
                else:
                    mask = f.prefix_mask(f.mul(lefts[m][1], 8))
                    f.masked_store(h, dst, mask, VH, VM)
                if partial:
                    f.close()
    if not c.nchw:
        rout = c.out_shape[1]
        f.open_if(f.cmp("eq", xt, 0))
        for n in range(nb):
            blk = f.add(f.mul(nt, nb), n)
            partial = c.ocb % nb and n >= c.ocb % nb
            if partial:
                f.open_if(f.cmp("ult", blk, c.ocb))
            _border(f, op, blk, y, c.hout, c.wout, rout)
            if partial:
                f.close()
        f.close()
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(c.name, args, ot, f)


def s2d_kernel(name, c, h, w) -> str:
    """The f32 NCHW image [1, c, h, w] to its space-to-depth by 2 in the
    blocked layout: pixel (y, x) of h / 2 x w / 2, channel ch * 4 + dy * 2 +
    dx = image[ch][2y + dy][2x + dx]. Per row, channel and image row of 64
    floats: one load, its even and odd lanes (dx); per block of 8 channels
    one segment store (vsseg8e16) interleaves them into the layout."""
    assert 4 * c <= 16 and h % 2 == 0 and w % 64 == 0, (c, h, w)
    oh, ow = h // 2, w // 2
    f = _Body()
    it_t = _memref([1, c, h, w], "f32")
    shape = act_shape(4 * c, oh, ow)
    ot = _memref(shape, F16, dynamic_layout=False)
    nblk, rows, wp = shape[0], shape[1], shape[2]
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ip = f.ptr_of("%img", it_t, 4, 4)
    op = f.ptr_of("%out", ot, 4, 2)
    v32h, v32i, v32f = "vector<[2]xf16>", "vector<[2]xi32>", "vector<[2]xf32>"
    i64v = "vector<[2]xi64>"
    zero32 = f.const(0.0, v32h)
    y = f.open_parallel(oh)
    for b in range(nblk):
        _border(f, op, f.i(b), y, oh, ow, rows)
    for x0 in range(0, ow, 32):
        chans = {}
        for ch in range(c):
            for dy in range(2):
                src = f.gep(
                    ip,
                    f.mul(
                        f.add(
                            f.mul(f.add(f.mul(y, 2), dy), w),
                            ch * h * w + 2 * x0,
                        ),
                        4,
                    ),
                )
                pairs = f.load(src, i64v, 4)
                for dx in range(2):
                    q = pairs
                    if dx:
                        q = f.op(
                            f"arith.shrui {q}, {f.const(32, i64v)} : {i64v}"
                        )
                    q = f.op(f"arith.trunci {q} : {i64v} to {v32i}")
                    q = f.op(f"arith.bitcast {q} : {v32i} to {v32f}")
                    chans[ch * 4 + dy * 2 + dx] = f.op(
                        f"arith.truncf {q} : {v32f} to {v32h}"
                    )
        for b in range(nblk):
            fields = [chans.get(8 * b + k, zero32) for k in range(8)]
            v = f.op(
                f'llvm.call_intrinsic "llvm.vector.interleave8"('
                f"{', '.join(fields)}) : ({', '.join([v32h] * 8)}) -> "
                f"vector<[16]xf16>"
            )
            dst = f.gep(
                op,
                f.add(f.mul(f.add(y, b * rows + 1), wp * 16), (x0 + 1) * 16),
            )
            f.store(v, dst, "vector<[16]xf16>")
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, [("%img", it_t)], ot, f)


def split2_kernel(name, c, h, w) -> str:
    """The blocked layout to its column-parity split (phase_shape), the
    input of a stride-2 convolution; 16-byte pixels moved as two i64 (a
    vector load costs ~10 ns on the A100, a gather ~10 ns per element)."""
    assert w % 2 == 0, w  # (w + 2) / 2 entries in each half of a row
    f = _Body()
    si, so = act_shape(c, h, w), phase_shape(c, h, w)
    it_t = _memref(si, F16)
    ot = _memref(so, F16, dynamic_layout=False)
    wp, half = si[2], so[3]
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ip = f.ptr_of("%x", it_t, 4, 2)
    op = f.ptr_of("%out", ot, 5, 2)
    row = f.open_parallel(si[0] * si[1])
    src = f.gep(ip, f.mul(row, wp * 16))
    dst = f.gep(op, f.mul(row, 2 * half * 16))
    j = f.open_for(0, half)
    for parity in (0, 1):
        s_ = f.gep(src, f.mul(f.add(f.mul(j, 2), parity), 16))
        d_ = f.gep(dst, f.mul(f.add(j, parity * half), 16))
        for word in (0, 8):
            v = f.load(f.gep(s_, word), "i64", 8)
            f.store(v, f.gep(d_, word), "i64", 8)
    f.close()
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, [("%x", it_t)], ot, f)


def upsample_kernel(name, c, h, w) -> str:
    """Nearest-neighbour upsampling by 2, blocked layout in and out."""
    f = _Body()
    si, so = act_shape(c, h, w), act_shape(c, 2 * h, 2 * w)
    it_t = _memref(si, F16)
    ot = _memref(so, F16, dynamic_layout=False)
    v8 = "vector<8xf16>"
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ip = f.ptr_of("%x", it_t, 4, 2)
    op = f.ptr_of("%out", ot, 4, 2)
    it = f.open_parallel(so[0] * 2 * h)
    blk, y = f.divmod(it, 2 * h)
    iy = f.op(f"arith.shrui {y}, {f.i(1)} : i64")
    x = f.open_for(0, 2 * w)
    ix = f.op(f"arith.shrui {x}, {f.i(1)} : i64")
    src = f.add(
        f.mul(f.add(f.mul(blk, si[1]), f.add(iy, 1)), si[2]), f.add(ix, 1)
    )
    v = f.load(f.gep(ip, f.mul(src, 16)), v8)
    dst = f.add(
        f.mul(f.add(f.mul(blk, so[1]), f.add(y, 1)), so[2]), f.add(x, 1)
    )
    f.store(v, f.gep(op, f.mul(dst, 16)), v8)
    f.close()
    _border(f, op, blk, y, 2 * h, 2 * w, so[1])
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, [("%x", it_t)], ot, f)


def cat_kernel(name, cs, h, w) -> str:
    """Concatenation along channels (whole blocks) in the blocked layout:
    in parallel over the padded rows of the output blocks."""
    f = _Body()
    shapes = [act_shape(c, h, w) for c in cs]
    in_ts = [_memref(sh, F16) for sh in shapes]
    nblk = sum(sh[0] for sh in shapes)
    rows, wp = shapes[0][1], shapes[0][2]
    ot = _memref([nblk, rows, wp, 8], F16, dynamic_layout=False)
    args = [(f"%x{i}", t) for i, t in enumerate(in_ts)]
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ips = [f.ptr_of(n, t, 4, 2) for n, t in args]
    op = f.ptr_of("%out", ot, 4, 2)
    row = wp * 16  # bytes of a padded row
    it = f.open_parallel(nblk * rows)
    blk, r = f.divmod(it, rows)
    dst = f.gep(op, f.mul(it, row))
    first = 0
    vc, mc = "vector<[32]xf16>", "vector<[32]xi1>"  # 512 fp16: 8 registers
    for i, sh in enumerate(shapes):
        lo, hi = f.cmp("uge", blk, first), f.cmp("ult", blk, first + sh[0])
        f.open_if(f.op(f"arith.andi {lo}, {hi} : i1"))
        src_row = f.add(f.mul(f.sub(blk, first), rows), r)
        src = f.gep(ips[i], f.mul(src_row, row))
        for off in range(0, row, 1024):
            n = min(1024, row - off) // 2
            if n == 512:
                f.store(f.load(f.gep(src, off), vc), f.gep(dst, off), vc)
            else:
                mask = f.op(f"vector.create_mask {f.x(n)} : {mc}")
                v = f.masked_load(f.gep(src, off), mask, vc, mc)
                f.masked_store(v, f.gep(dst, off), mask, vc, mc)
        f.close()
        first += sh[0]
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, args, ot, f)


def sppf_kernel(name, c, h, w) -> str:
    """The SPPF block of YOLOv8 without its convolutions: [x, p(x), p(p(x)),
    p(p(p(x)))] concatenated, p the 5 x 5 max pool of stride 1 and padding
    2 (-inf), i.e. the max over the windows of 5, 9 and 13 pixels clipped to
    the image. A padded row is one vector; its windows are maxima of shifted
    copies (whole pixels), then of rows."""
    f = _Body()
    si, so = act_shape(c, h, w), act_shape(4 * c, h, w)
    cb, rows, wp = si[0], si[1], si[2]
    it_t = _memref(si, F16)
    ot = _memref(so, F16, dynamic_layout=False)
    n = wp * 8
    vr, mr = f"vector<{n}xf16>", f"vector<{n}xi1>"
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ip = f.ptr_of("%x", it_t, 4, 2)
    op = f.ptr_of("%out", ot, 4, 2)
    halo = [i < 8 or i >= 8 * (w + 1) for i in range(n)]
    hmask = f.const(halo, mr)
    ninf = f.const("0xFC00", vr)
    zero = f.const(0.0, vr)

    def shifted(v, d):
        """v shifted by d pixels (lane i gets lane i + 8d), -inf past it."""
        idx = [i + 8 * d if 0 <= i + 8 * d < n else n for i in range(n)]
        return f.op(
            f"vector.shuffle {v}, {ninf} [{', '.join(map(str, idx))}] : "
            f"{vr}, {vr}"
        )

    def hmax(v, r, step):
        out = v
        for d in range(step, r + 1, step):
            for s in (d, -d):
                out = f.op(f"arith.maxnumf {out}, {shifted(v, s)} : {vr}")
        return out

    it = f.open_parallel(cb * h)
    blk, y = f.divmod(it, h)
    pools = [None, None, None]
    for d in range(-6, 7):
        yy = f.add(y, d)
        ok = f.op(
            f"arith.andi {f.cmp('sge', yy, 0)}, {f.cmp('slt', yy, h)} : i1"
        )
        yc = f.op(f"arith.select {ok}, {yy}, {y} : i64")
        row = f.load(
            f.gep(ip, f.mul(f.add(f.mul(blk, rows), f.add(yc, 1)), wp * 16)),
            vr,
        )
        if d == 0:
            x_row = row
        row = f.op(f"arith.select {hmask}, {ninf}, {row} : {mr}, {vr}")
        h2 = hmax(row, 2, 1)
        h4 = hmax(h2, 2, 2)
        h6 = hmax(h4, 2, 2)
        for i, (hr, r) in enumerate(((h2, 2), (h4, 4), (h6, 6))):
            if abs(d) > r:
                continue
            if pools[i] is None:
                pools[i] = f.op(f"arith.select {ok}, {hr}, {ninf} : {vr}")
            else:
                mx = f.op(f"arith.maxnumf {pools[i]}, {hr} : {vr}")
                pools[i] = f.op(f"arith.select {ok}, {mx}, {pools[i]} : {vr}")
    for i, v in enumerate([x_row] + pools):
        v = f.op(f"arith.select {hmask}, {zero}, {v} : {mr}, {vr}")
        ob = f.add(blk, i * cb)
        dst = f.gep(op, f.mul(f.add(f.mul(ob, rows), f.add(y, 1)), wp * 16))
        f.store(v, dst, vr)
        _border(f, op, ob, y, h, w, rows)
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, [("%x", it_t)], ot, f)


def decode_kernel(name, levels, nc, nk) -> str:
    """The Detect / Pose head decode of Ultralytics (export mode) from the
    f32 [1, C, h, w] outputs of the head convolutions, per level (h, w,
    stride): the boxes of the 4 x 16 DFL bins (softmax, expectation,
    dist2bbox to xywh, times the stride), the class scores (sigmoid) and the
    keypoints ((k * 2 + anchor - 0.5) * stride, the visibility a sigmoid),
    into the [1, 4 + nc + 3 * nk, anchors] head output. The anchors are the
    pixel centres (x + 0.5, y + 0.5) of each level. In parallel over chunks
    of 64 anchors of a level times 4 parts (x and width, y and height, the
    classes, the keypoints); the bins, classes and keypoints are loops, so
    that the loads of all of them are not scheduled at once (spilled)."""
    f = _Body()
    na = sum(h * w for h, w, _ in levels)
    nch = 4 + nc + 3 * nk
    args = []
    for kind, c in (("box", 4 * REG_MAX), ("cls", nc), ("kpt", 3 * nk)):
        if c == 0:
            continue
        for i, (h, w, _) in enumerate(levels):
            args.append((f"%{kind}{i}", _memref([1, c, h, w], "f32")))
    ot = _memref([1, nch, na], "f32", dynamic_layout=False)
    f.emit(f"%out = memref.alloc() alignment = 128 : {ot}")
    ptr = {n: f.ptr_of(n, t, 4, 4) for n, t in args}
    op = f.ptr_of("%out", ot, 3, 4)
    vi = "vector<[4]xi32>"
    lane = f.op(f"llvm.intr.stepvector : {vi}")
    chunks = [-(-h * w // 64) for h, w, _ in levels]
    it = f.open_parallel(sum(chunks) * 4)
    chunk, part = f.divmod(it, 4)
    first, base = 0, 0
    for li, (h, w, stride) in enumerate(levels):
        n = h * w
        lo = f.cmp("uge", chunk, first)
        hi = f.cmp("ult", chunk, first + chunks[li])
        f.open_if(f.op(f"arith.andi {lo}, {hi} : i1"))
        p0 = f.mul(f.sub(chunk, first), 64)
        left = f.op(f"arith.subi {f.i(n)}, {p0} : i64")
        mask = f.prefix_mask(f.op(f"arith.minsi {left}, {f.i(64)} : i64"))
        # Pixel coordinates of the lanes: the anchors minus 0.5.
        p0i = f.op(f"arith.trunci {p0} : i64 to i32")
        pv = f.op(f"vector.broadcast {p0i} : i32 to {vi}")
        pv = f.op(f"arith.addi {pv}, {lane} : {vi}")
        gy = f.op(f"arith.divui {pv}, {f.const(w, vi)} : {vi}")
        gx = f.op(f"arith.remui {pv}, {f.const(w, vi)} : {vi}")
        gx = f.op(f"arith.uitofp {gx} : {vi} to {VF}")
        gy = f.op(f"arith.uitofp {gy} : {vi} to {VF}")
        half = f.const(0.5, VF)
        sv = f.const(float(stride), VF)

        def load(kind, ch, li=li, p0=p0, n=n, mask=mask):
            src = f.gep(ptr[f"%{kind}{li}"], f.mul(f.add(p0, f.mul(ch, n)), 4))
            return f.masked_load(src, mask, VF, VM, 4)

        def store(ch, v, p0=p0, base=base, mask=mask):
            dst = f.gep(op, f.mul(f.add(p0, f.add(f.mul(ch, na), base)), 4))
            f.masked_store(v, dst, mask, VF, VM, 4)

        def dfl(side):
            """The expected distance of one box side (softmax of its bins)."""
            res = f.fresh("mx")
            bv = f.fresh("b")
            f.emit(
                f"{res} = scf.for {bv} = {f.x(0)} to {f.x(REG_MAX)} step "
                f"{f.x(1)} iter_args(%m = {f.const('0xFF800000', VF)}) -> "
                f"({VF}) {{"
            )
            f.indent += 2
            x = load("box", f.add(f.to_i64(bv), side * REG_MAX))
            f.emit(f"scf.yield {f.op(f'arith.maxnumf %m, {x} : {VF}')} : {VF}")
            f.close()
            sums = f.fresh("sm")
            bv = f.fresh("b")
            zero = f.const(0.0, VF)
            f.emit(
                f"{sums}:2 = scf.for {bv} = {f.x(0)} to {f.x(REG_MAX)} step "
                f"{f.x(1)} iter_args(%s = {zero}, %t = {zero}) -> "
                f"({VF}, {VF}) {{"
            )
            f.indent += 2
            bi = f.to_i64(bv)
            x = load("box", f.add(bi, side * REG_MAX))
            e = _exp(f, f.op(f"arith.subf {x}, {res} : {VF}"))
            bf = f.op(f"arith.uitofp {bi} : i64 to f32")
            bf = f.op(f"vector.broadcast {bf} : f32 to {VF}")
            ws = f.op(f"math.fma {e}, {bf}, %t : {VF}")
            f.emit(
                f"scf.yield {f.op(f'arith.addf %s, {e} : {VF}')}, {ws} : "
                f"{VF}, {VF}"
            )
            f.close()
            return f.op(f"arith.divf {sums}#1, {sums}#0 : {VF}")

        # dist2bbox (xywh): (x1 + x2) / 2 and x2 - x1, as Ultralytics.
        for pi, (g, lo_side, hi_side) in enumerate(((gx, 0, 2), (gy, 1, 3))):
            f.open_if(f.cmp("eq", part, pi))
            a = f.op(f"arith.addf {g}, {half} : {VF}")
            v1 = f.op(f"arith.subf {a}, {dfl(lo_side)} : {VF}")
            v2 = f.op(f"arith.addf {a}, {dfl(hi_side)} : {VF}")
            ctr = f.op(f"arith.addf {v1}, {v2} : {VF}")
            ctr = f.op(f"arith.mulf {ctr}, {half} : {VF}")
            size = f.op(f"arith.subf {v2}, {v1} : {VF}")
            store(f.i(pi), f.op(f"arith.mulf {ctr}, {sv} : {VF}"))
            store(f.i(pi + 2), f.op(f"arith.mulf {size}, {sv} : {VF}"))
            f.close()
        f.open_if(f.cmp("eq", part, 2))
        k = f.open_for(0, nc)
        store(f.add(k, 4), _sigmoid(f, load("cls", k)))
        f.close()
        f.close()
        if nk:
            f.open_if(f.cmp("eq", part, 3))
            k = f.open_for(0, nk)
            two = f.const(2.0, VF)
            k3 = f.mul(k, 3)
            for j, g in ((0, gx), (1, gy)):
                v = f.op(
                    f"math.fma {load('kpt', f.add(k3, j))}, {two}, {g} : {VF}"
                )
                store(
                    f.add(k3, 4 + nc + j), f.op(f"arith.mulf {v}, {sv} : {VF}")
                )
            store(f.add(k3, 4 + nc + 2), _sigmoid(f, load("kpt", f.add(k3, 2))))
            f.close()
            f.close()
        f.close()
        first += chunks[li]
        base += n
    f.close(parallel=True)
    f.emit(f"return %out : {ot}")
    return _func(name, args, ot, f)


_KERNELS = {
    "conv": lambda spec: conv_kernel(spec),
    "s2d": lambda spec: s2d_kernel(*spec),
    "split2": lambda spec: split2_kernel(*spec),
    "upsample": lambda spec: upsample_kernel(*spec),
    "cat": lambda spec: cat_kernel(*spec),
    "sppf": lambda spec: sppf_kernel(*spec),
    "decode": lambda spec: decode_kernel(*spec),
}


def gen_kernels(kernels) -> str:
    """The MLIR module of the kernels rewrite_graph returned."""
    parts = [_KERNELS[kind](spec) for kind, spec in kernels]
    tcm = [s for k, s in kernels if k == "conv" and s.tcm_off is not None]
    if tcm:
        parts = (
            [
                "func.func private @buddy_spacemit_tcm_pair(i64, i64, i64) -> "
                "i64 attributes {llvm.emit_c_interface}\n",
                "func.func private @buddy_spacemit_tcm_here(i64, i64) -> i64 "
                "attributes {llvm.emit_c_interface}\n",
            ]
            + [
                f"llvm.mlir.global internal @{c.name}_tcm(0 : i64) : i64\n"
                for c in tcm
            ]
            + parts
        )
    return "module {\n" + "\n".join(parts) + "}\n"


# ---------------------------------------------------------------------------
# Graph rewrite
# ---------------------------------------------------------------------------


class _Rewriter:
    def __init__(self, graph, params):
        self.g = graph
        self.params = params  # torch tensors, in graph.params order
        self.kernels = []
        self.names = set()
        # name -> (channels, h, w) of the values in the blocked layout
        self.blocked = {}
        # name of a blocked value -> its column-parity split (split2)
        self.phase = {}
        # (Conv, call name) of the convolutions with f32 NCHW outputs: the
        # last of each head branch
        self.head = []
        self.image_size = None

    def node(self, name):
        return self.g.node_table[name]

    def param(self, name):
        for i, p in enumerate(self.g.params):
            if p.name == name:
                return i
        raise KeyError(name)

    def set_param(self, name, array):
        i = self.param(name)
        node = self.node(name)
        node.tensor_meta["shape"] = torch.Size(list(array.shape))
        node.tensor_meta["dtype"] = TensorDType.Float32
        self.params[i] = torch.from_numpy(numpy.ascontiguousarray(array))

    def unlink(self, node):
        for p in node._parents:
            pn = self.g.node_table.get(p)
            if pn is not None:
                while node.name in pn._children:
                    pn._children.remove(node.name)
        node._parents = []

    def drop(self, node):
        """Remove a node whose uses are all rewritten."""
        self.unlink(node)
        node._children = []
        self.g.delete_node(node, [])

    def new_call(self, name, fn, args, shape, dtype):
        call = CallExternalOp(
            call_func_name=fn,
            args=list(args),
            args_index=[0] * len(args),
            tensor_meta={"shape": list(shape), "dtype": dtype},
            name=name,
            written_args=[],  # the kernels only read their arguments
        )
        call._parents = list(dict.fromkeys(args))
        for a in call._parents:
            self.node(a)._children.append(call.name)
        return call

    def put_call(self, target, fn, args, shape, dtype):
        """A call of fn in place of `target` (same name, same uses)."""
        self.unlink(target)
        call = self.new_call(target.name, fn, args, shape, dtype)
        idx = self.g.body.index(target)
        self.g.body[idx] = call
        self.g.node_table[call.name] = call
        call._children = target._children
        return call

    def insert_call(self, before, name, fn, args, shape, dtype):
        """A new call named `name` right before node `before`."""
        call = self.new_call(name, fn, args, shape, dtype)
        idx = self.g.body.index(before)
        self.g.body.insert(idx, call)
        self.g.node_table[name] = call
        self.g._inputs = [i + 1 if i >= idx else i for i in self.g._inputs]
        self.g._fake_params = [
            i + 1 if i >= idx else i for i in self.g._fake_params
        ]
        return call

    def fn_name(self, base):
        name, i = base, 1
        while name in self.names:
            i += 1
            name = f"{base}_{i}"
        self.names.add(name)
        return name

    # -----------------------------------------------------------------------

    def silu_of(self, conv):
        """(sigmoid, mul) of conv * sigmoid(conv), or None."""
        kids = [self.node(k) for k in conv._children]
        sig = [k for k in kids if isinstance(k, SigmoidOp)]
        if len(kids) != 2 or len(sig) != 1:
            return None
        mul = [k for k in kids if isinstance(k, MulOp)]
        if (
            len(mul) == 1
            and sorted(mul[0].args) == sorted([conv.name, sig[0].name])
            and sig[0]._children == [mul[0].name]
        ):
            return sig[0], mul[0]
        return None

    def first_conv_input(self, conv, x, wt):
        """The space-to-depth of the image for the first convolution (3 x 3,
        stride 2): returns its name and the weights of the equivalent
        2 x 2, stride 1 convolution on it."""
        _, c, h, w = self.node(x).tensor_meta["shape"]
        oc = wt.shape[0]
        assert wt.shape[1:] == (c, 3, 3) and list(conv.args[3]) == [2, 2]
        self.image_size = h
        name = self.fn_name("yolo_s2d")
        self.kernels.append(("s2d", (name, c, h, w)))
        node = self.insert_call(
            conv,
            f"{x}_s2d",
            name,
            [x],
            act_shape(4 * c, h // 2, w // 2),
            TensorDType.Float16,
        )
        self.blocked[node.name] = (-(-4 * c // 8) * 8, h // 2, w // 2)
        # Tap ky of the 3 x 3 kernel reads image row 2y + ky - 1: row
        # y + ky2 - 1 of the space-to-depth image, dy = ky + 1 - 2 * ky2.
        w2 = numpy.zeros((oc, -(-4 * c // 8) * 8, 2, 2), numpy.float32)
        for ky in range(3):
            ky2, dy = divmod(ky + 1, 2)
            for kx in range(3):
                kx2, dx = divmod(kx + 1, 2)
                for ch in range(c):
                    w2[:, ch * 4 + dy * 2 + dx, ky2, kx2] = wt[:, ch, ky, kx]
        return node.name, w2

    def phase_of(self, x, before):
        """The column-parity split of blocked value x (split2)."""
        if x not in self.phase:
            c, h, w = self.blocked[x]
            name = self.fn_name(f"yolo_split2_{c}_{h}x{w}")
            self.kernels.append(("split2", (name, c, h, w)))
            node = self.insert_call(
                before,
                f"{x}_phase",
                name,
                [x],
                phase_shape(c, h, w),
                TensorDType.Float16,
            )
            self.phase[x] = node.name
        return self.phase[x]

    def rewrite_conv(self, conv):
        x, wname, bname = conv.args[0], conv.args[1], conv.args[2]
        stride, padding, dilation = conv.args[3], conv.args[4], conv.args[5]
        transposed, groups = conv.args[6], conv.args[8]
        image = x in [n.name for n in self.g.inputs]
        if not image and x not in self.blocked:
            return
        wt = self.params[self.param(wname)].detach().numpy()
        bt = self.params[self.param(bname)].detach().numpy()
        oc, ic, kh, kw = wt.shape
        assert kh == kw and groups == 1 and not transposed
        assert list(stride) in ([1, 1], [2, 2]) and list(dilation) == [1, 1]
        assert list(padding) == [kh // 2] * 2, (conv.name, padding)
        s = stride[0]
        if image:
            x, wt = self.first_conv_input(conv, x, wt)
            ic, kh, s = wt.shape[1], 2, 1
        c, h, w = self.blocked[x]
        assert ic == c, (conv.name, ic, c)
        silu = self.silu_of(conv)
        target, residual = conv, None
        if silu is not None:
            target = silu[1]
            if len(target._children) == 1:
                add = self.node(target._children[0])
                if isinstance(add, AddOp):
                    other = [a for a in add.args if a != target.name]
                    if len(other) == 1 and other[0] in self.blocked:
                        residual = other[0]
                        target = add
        spec = Conv(
            self.fn_name(f"yolo_{conv.name}"),
            -(-c // 8) * 8,
            oc,
            kh,
            s,
            h,
            w,
            silu=silu is not None,
            residual=residual is not None,
            nchw=silu is None,
        )
        wp, bp = pack_conv(spec, wt, bt)
        self.set_param(wname, wp)
        self.set_param(bname, bp)
        self.kernels.append(("conv", spec))
        xin = self.phase_of(x, conv) if s == 2 else x
        args = [xin, wname, bname] + ([residual] if residual else [])
        dtype = TensorDType.Float32 if spec.nchw else TensorDType.Float16
        rest = [conv] + (list(silu) if silu else [])
        call = self.put_call(target, spec.name, args, spec.out_shape, dtype)
        for n in rest:
            if self.g.node_table.get(n.name) is n:
                self.drop(n)
        if spec.nchw:
            self.head.append((spec, call.name))
        else:
            self.blocked[call.name] = (oc, spec.hout, spec.wout)

    def rewrite_split(self, node):
        x, size, dim = node.args
        if x not in self.blocked:
            return
        c, h, w = self.blocked[x]
        assert dim == 1 and size % 8 == 0 and c % size == 0
        node._arguments = [x, size // 8, 0]
        n = c // size
        node.tensor_meta["shape"] = tuple(
            torch.Size(act_shape(size, h, w)) for _ in range(n)
        )
        node.tensor_meta["dtype"] = tuple(TensorDType.Float16 for _ in range(n))
        for k in node._children:
            gi = self.node(k)
            assert isinstance(gi, GetItemOp)
            gi.tensor_meta["shape"] = torch.Size(act_shape(size, h, w))
            gi.tensor_meta["dtype"] = TensorDType.Float16
            self.blocked[gi.name] = (size, h, w)

    def pool_of(self, name):
        """(input, kernel size) of max_pool(input)[0] named name, or None."""
        gi = self.g.node_table.get(name)
        if not isinstance(gi, GetItemOp) or gi.args[1] != 0:
            return None
        mp = self.node(gi.args[0])
        if not isinstance(mp, LowMemoryMaxPoolWithOffsetsOp):
            return None
        x, ks, st, pad, dil, ceil = mp.args
        k = ks[0]
        if (
            list(ks) != [k, k]
            or list(st) != [1, 1]
            or list(pad) != [k // 2] * 2
            or list(dil) != [1, 1]
            or ceil
        ):
            return None
        return x, k

    def rewrite_cat(self, node):
        xs, dim = node.args[0], node.args[1]
        if dim != 1 or not xs or xs[0] not in self.blocked:
            return
        # SPPF: [x, p(x), p(p(x)), p(p(p(x)))], p the 5 x 5 max pool.
        if len(xs) == 4 and all(
            self.pool_of(xs[i]) == (xs[i - 1], 5) for i in (1, 2, 3)
        ):
            c, h, w = self.blocked[xs[0]]
            name = self.fn_name(f"yolo_sppf_{c}_{h}x{w}")
            self.kernels.append(("sppf", (name, c, h, w)))
            call = self.put_call(
                node, name, [xs[0]], act_shape(4 * c, h, w), TensorDType.Float16
            )
            self.blocked[call.name] = (4 * c, h, w)
            return
        assert all(x in self.blocked for x in xs), node.name
        shapes = [self.blocked[x] for x in xs]
        assert all(c % 8 == 0 and (h, w) == shapes[0][1:] for c, h, w in shapes)
        cs = [s[0] for s in shapes]
        h, w = shapes[0][1:]
        name = self.fn_name(f"yolo_cat{len(cs)}_{sum(cs)}_{h}x{w}")
        self.kernels.append(("cat", (name, cs, h, w)))
        call = self.put_call(
            node, name, xs, act_shape(sum(cs), h, w), TensorDType.Float16
        )
        self.blocked[call.name] = (sum(cs), h, w)

    def rewrite_upsample(self, node):
        x, idx = node.args
        if x not in self.blocked:
            return
        c, h, w = self.blocked[x]
        oshape = list(node.tensor_meta["shape"])
        assert oshape == [1, c, 2 * h, 2 * w], (node.name, oshape)
        assert idx[0] is None and idx[1] is None
        name = self.fn_name(f"yolo_upsample2_{c}_{h}x{w}")
        self.kernels.append(("upsample", (name, c, h, w)))
        call = self.put_call(
            node, name, [x], act_shape(c, 2 * h, 2 * w), TensorDType.Float16
        )
        self.blocked[call.name] = (c, 2 * h, 2 * w)

    def rewrite_head(self, nc, nk):
        """The head decode (everything after the head convolutions) to one
        kernel, in place of the graph output."""
        out = [n for n in self.g.body if isinstance(n, OutputOp)]
        assert len(out) == 1 and len(out[0].args) == 1
        target = self.node(out[0].args[0])
        groups = {}
        for kind, c in (("box", 4 * REG_MAX), ("cls", nc), ("kpt", 3 * nk)):
            if c:
                groups[kind] = sorted(
                    [(sp, n) for sp, n in self.head if sp.oc == c],
                    key=lambda t: -t[0].hout,
                )
        sizes = [(sp.hout, sp.wout) for sp, _ in groups["box"]]
        assert len(self.head) == sum(len(g) for g in groups.values())
        assert all(
            [(sp.hout, sp.wout) for sp, _ in g] == sizes
            for g in groups.values()
        ), "the head convolutions do not form levels"
        levels = [(h, w, self.image_size // h) for h, w in sizes]
        na = sum(h * w for h, w, _ in levels)
        assert list(target.tensor_meta["shape"]) == [1, 4 + nc + 3 * nk, na]
        self.check_head_params(levels)
        name = self.fn_name("yolo_head_decode")
        self.kernels.append(("decode", (name, levels, nc, nk)))
        args = [
            n
            for kind in ("box", "cls", "kpt")
            if kind in groups
            for _, n in groups[kind]
        ]
        self.put_call(
            target, name, args, [1, 4 + nc + 3 * nk, na], TensorDType.Float32
        )

    def check_head_params(self, levels):
        """The anchors, strides and DFL weights of the graph are those the
        decode kernel computes."""
        ax, ay, st = [], [], []
        for h, w, stride in levels:
            gy, gx = numpy.meshgrid(
                numpy.arange(h), numpy.arange(w), indexing="ij"
            )
            ax.append(gx.reshape(-1) + 0.5)
            ay.append(gy.reshape(-1) + 0.5)
            st.append(numpy.full(h * w, stride))
        anchors = numpy.stack([numpy.concatenate(ax), numpy.concatenate(ay)])
        strides = numpy.concatenate(st)[None]
        found = set()
        for i, p in enumerate(self.g.params):
            t = self.params[i].detach().numpy()
            if t.shape == anchors.shape:
                assert numpy.array_equal(t, anchors), "unexpected anchors"
                found.add("anchors")
            elif t.shape == strides.shape:
                assert numpy.array_equal(t, strides), "unexpected strides"
                found.add("strides")
            elif t.shape == (1, REG_MAX, 1, 1):
                assert numpy.array_equal(
                    t.reshape(-1), numpy.arange(REG_MAX)
                ), "unexpected DFL weights"
                found.add("dfl")
        assert found == {"anchors", "strides", "dfl"}, found

    def place_tcm(self):
        """Weights into the TCM: those of the convolutions with the most
        pixel tiles (each tile reads its weights once) first, while they
        fit."""
        convs = [s for k, s in self.kernels if k == "conv"]
        used = 0
        for c in sorted(
            convs,
            key=lambda c: -(c.hout * (c.xt if not c.flat else c.qt / c.hout)),
        ):
            size = -(-c.w_elems() * 4 // 128) * 128
            if used + size <= TCM_BYTES:
                c.tcm_off = used
                used += size

    def remove_dead(self):
        changed = True
        while changed:
            changed = False
            for node in list(self.g.body):
                if isinstance(node, (PlaceholderOp, OutputOp)):
                    continue
                if not node._children:
                    self.drop(node)
                    changed = True

    def run(self, nc, nk):
        for node in list(self.g.body):
            if self.g.node_table.get(node.name) is not node:
                continue
            if isinstance(node, Conv2dOp):
                self.rewrite_conv(node)
            elif isinstance(node, SplitOp):
                self.rewrite_split(node)
            elif isinstance(node, CatOp):
                self.rewrite_cat(node)
            elif isinstance(node, UnsafeIndexOp):
                self.rewrite_upsample(node)
            elif isinstance(node, AddOp):
                assert not any(a in self.blocked for a in node.args), (
                    f"unfused add of blocked values: {node.name}"
                )
        self.rewrite_head(nc, nk)
        self.remove_dead()
        self.place_tcm()
        # The kernels still called (the max pools of the SPPF are not).
        called = {
            n.call_func_name
            for n in self.g.body
            if isinstance(n, CallExternalOp)
        }
        self.kernels = [
            (kind, spec)
            for kind, spec in self.kernels
            if (spec.name if kind == "conv" else spec[0]) in called
        ]
        self.g._enable_external_calls = True


def rewrite_graph(graph, params, nc, nk) -> list:
    """Rewrite the YOLOv8 graph (nc classes, nk keypoints) to the IME
    kernels (see the header); params (the torch tensors of graph.params, in
    order) get the packed weights. Returns the kernels to generate
    (gen_kernels)."""
    rw = _Rewriter(graph, params)
    rw.run(nc, nk)
    return rw.kernels
