# ===- k3_w4.py ----------------------------------------------------------------
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
# Int4 group-quantized LLM layers for RVV with VLEN 1024 (the A100 cores of
# the SpacemiT K3), variant "w4g32" (docs/K3DeepSeekR1.md).
#
# 1. k3_w4_rewrite(graph, m, pos_input, threads) replaces every Linear
#    (MatmulOp / AddMMOp with a parameter weight) by a CallExternalOp to a
#    kernel and repacks the weight parameter in place:
#    - q / k / v projections that share an input become one 3-output kernel;
#    - gate / up projections followed by silu(gate) * up become one kernel
#      that returns the product;
#    - an RMSNorm feeding a kernel is computed by that kernel;
#    - RoPE, the KV cache update and attention over the cache become one
#      kernel that only visits the positions up to the current one; it reads
#      q / k / v and writes its output as the [rows, heads * dim] matrices of
#      the projections, so the head views in between disappear.
# 2. build_kernels(specs) builds those kernels with the MLIR Python bindings,
#    one func.func per shape in one module (scf / vector / memref,
#    scf.parallel for the threads); LLVM lowers the inner loop to vle8 / vsll /
#    vsra / vwmacc.vx.
#
# Weight layout. W[K, N] is int4 with one f16 scale per (32-row group,
# column), rounded like llama.cpp Q4_0 (d = signed max / -8, q in [-8, 7]).
# Columns are split into tiles of NB = 128 (one e8 register at VLEN 1024);
# every tile is one contiguous block, group by group:
#
#   16 rows of 128 bytes   byte j of row p: low nibble  W[g*32 + 2p,   n0 + j]
#                                           high nibble W[g*32 + 2p+1, n0 + j]
#   128 f16 scales
#
# A thread streams its tiles linearly. x is quantized on the fly to int8 per
# 32-element group (absmax / 127); the products of a group accumulate exactly
# in int16 (32 * 127 * 8 < 2^15) and are scaled into f32 once per group.
#
# ===---------------------------------------------------------------------------

import numpy
import torch
from buddy_mlir import ir
from buddy_mlir.dialects import arith, func, math, memref, scf, vector

from .. import Graph
from ..operation import (
    AddMMOp,
    CallExternalOp,
    GetItemOp,
    IndexPutOp,
    MatmulOp,
    MeanOp,
    MulOp,
    OutputOp,
    PermuteOp,
    PlaceholderOp,
    PowOp,
    ReshapeOp,
    RsqrtOp,
    ScaledDotProductFlashAttentionForCpuOp,
    SigmoidOp,
    ViewOp,
)
from ..operation import AddOp as _AddOp
from ..type import TensorDType

G = 32  # rows per quantization group
NB = 128  # columns per weight tile
# Rows of a prefill chunk that share each unpacked weight vector.
PREFILL_MB = 4
# Prefill attention: query rows per work item, and head dimensions per pass.
# gen_config.py (w4g32_param_counts) requires prefill_chunk % ATTN_ROWS == 0
# (which also makes it a multiple of PREFILL_MB) and head_dim % ATTN_DIMS == 0.
ATTN_ROWS = 32
ATTN_DIMS = 16


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def tile_bytes(k: int) -> int:
    """Bytes of one tile of a [k, *] weight."""
    return (k // G) * (G // 2 * NB + NB * 2)


def packed_bytes(k: int, n: int) -> int:
    """Bytes of a packed [k, n] weight."""
    return tile_bytes(k) * (n // NB)


def quantize_q4(w: numpy.ndarray):
    """[K, N] floats -> int4 values q [K/G, G, N] and f16 scales [K/G, N],
    rounded like llama.cpp Q4_0 (scale = signed max / -8, q in [-8, 7])."""
    k, n = w.shape
    assert k % G == 0 and n % NB == 0, (k, n)
    w = w.astype(numpy.float32).reshape(k // G, G, n)
    idx = numpy.abs(w).argmax(axis=1)[:, None, :]
    signed_max = numpy.take_along_axis(w, idx, axis=1)[:, 0, :]
    scale = numpy.ascontiguousarray((signed_max / -8.0).astype(numpy.float16))
    s32 = scale.astype(numpy.float32)
    inv = numpy.where(s32 != 0, 1.0 / numpy.where(s32 != 0, s32, 1), 0)
    q = numpy.clip(numpy.rint(w * inv[:, None, :]), -8, 7).astype(numpy.int8)
    return q, scale


def dequantize_q4(q, scale) -> numpy.ndarray:
    """The [K, N] float matrix a (q, scale) pair stands for."""
    groups, _, n = q.shape
    w = q.astype(numpy.float32) * scale.astype(numpy.float32)[:, None, :]
    return w.reshape(groups * G, n)


def pack_tiles(q, scale) -> numpy.ndarray:
    """The tile layout of the module comment, as int8."""
    groups, _, n = q.shape
    rows = ((q[:, 0::2, :] & 0x0F) | ((q[:, 1::2, :] & 0x0F) << 4)).astype(
        numpy.uint8
    )
    rows = rows.reshape(groups, G // 2, n // NB, NB).transpose(2, 0, 1, 3)
    sc = (
        scale.view(numpy.uint8)
        .reshape(groups, n // NB, NB * 2)
        .transpose(1, 0, 2)
    )
    out = numpy.concatenate(
        [rows.reshape(n // NB, groups, G // 2 * NB), sc], axis=2
    )
    return out.reshape(-1).view(numpy.int8)


def pack_q4(w: numpy.ndarray) -> numpy.ndarray:
    """Quantize a [K, N] float matrix into the tile layout."""
    return pack_tiles(*quantize_q4(w))


def pack_glu(wg: numpy.ndarray, wu: numpy.ndarray) -> numpy.ndarray:
    """Gate and up tiles interleaved: g0, u0, g1, u1, ..."""
    k, n = wg.shape
    tb = tile_bytes(k)
    g = pack_q4(wg).reshape(n // NB, tb)
    u = pack_q4(wu).reshape(n // NB, tb)
    return numpy.stack([g, u], axis=1).reshape(-1)


# ---------------------------------------------------------------------------
# Kernel specs
# ---------------------------------------------------------------------------


def kernel_spec(
    kind: str, m: int, k: int, ns: list, bias: bool, threads: int
) -> dict:
    """A matmul kernel: kind "plain" (x W), "multi" (x [W0 | W1 | ...] with
    one result per part, q / k / v) or "glu" (silu(x Wg) * (x Wu))."""
    name = f"k3_q4_{kind}_m{m}_k{k}_n{'_'.join(map(str, ns))}"
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
    }


# ---------------------------------------------------------------------------
# Graph rewrite
# ---------------------------------------------------------------------------


class _Rewriter:
    def __init__(self, graph: Graph, m: int, threads: int):
        self.g = graph
        self.m = m
        self.threads = threads
        self.refs = graph._params_ref
        self.kernels = {}
        self.uid = 0

    # -- bookkeeping helpers ------------------------------------------------

    def param_index(self, name):
        for i, p in enumerate(self.g.params):
            if p.name == name:
                return i
        return None

    def weight(self, name) -> numpy.ndarray:
        return self.refs[self.param_index(name)].detach().float().numpy()

    def set_param(self, name, array: numpy.ndarray, dtype: TensorDType):
        i = self.param_index(name)
        node = self.g.node_table[name]
        node.tensor_meta["shape"] = torch.Size(list(array.shape))
        node.tensor_meta["dtype"] = dtype
        self.refs[i] = torch.from_numpy(numpy.ascontiguousarray(array))

    def drop_param(self, name):
        i = self.param_index(name)
        node = self.g.node_table[name]
        assert not node._children, (name, node._children)
        self.g.delete_node(node, [])
        del self.refs[i]

    def unlink(self, node):
        for p in node._parents:
            pn = self.g.node_table.get(p)
            if pn is not None and node.name in pn._children:
                pn._children.remove(node.name)

    def delete(self, node):
        """Delete a node whose children are all gone."""
        assert not node._children, (node.name, node._children)
        self.unlink(node)
        self.g.delete_node(node, [])

    def link(self, node, parents):
        node._parents = list(parents)
        for p in parents:
            self.g.node_table[p]._children.append(node.name)

    def insert_before(self, anchor, node):
        idx = self.g.body.index(anchor)
        self.g.body.insert(idx, node)
        self.g.node_table[node.name] = node
        # body indices of inputs / params after idx shift by one
        self.g._inputs = [i + 1 if i >= idx else i for i in self.g._inputs]
        self.g._fake_params = [
            i + 1 if i >= idx else i for i in self.g._fake_params
        ]

    def replace(self, old, new):
        """Put `new` in place of `old` (same name): keeps old's children."""
        assert new.name == old.name
        self.unlink(old)
        idx = self.g.body.index(old)
        self.g.body[idx] = new
        self.g.node_table[new.name] = new
        new._children = old._children

    def call(self, spec, args, out_shapes, anchor):
        self.kernels[spec["name"]] = spec
        self.uid += 1
        multi = len(out_shapes) > 1
        node = CallExternalOp(
            call_func_name=spec["name"],
            args=args,
            args_index=[0] * len(args),
            tensor_meta={
                "shape": (
                    [list(s) for s in out_shapes]
                    if multi
                    else list(out_shapes[0])
                ),
                "dtype": (
                    [TensorDType.Float32] * len(out_shapes)
                    if multi
                    else TensorDType.Float32
                ),
            },
            name=f"k3call_{self.uid}",
        )
        self.insert_before(anchor, node)
        self.link(node, args)
        return node

    def getitem(self, old, call, index, shape):
        """Put call[index] in place of `old`."""
        self.unlink(old)
        old._parents = []
        gi = GetItemOp()
        gi._name = old.name
        gi._arguments = [call.name, index]
        gi._tensor_meta = {
            "shape": torch.Size(shape),
            "dtype": TensorDType.Float32,
        }
        self.replace(old, gi)
        gi._parents = [call.name]
        call._children.append(gi.name)

    @staticmethod
    def _weight_arg(node):
        if isinstance(node, MatmulOp):
            return 0, 1, None
        if isinstance(node, AddMMOp):
            return 1, 2, 0
        return None

    def is_linear(self, node):
        pos = self._weight_arg(node)
        if pos is None:
            return False
        w = str(node.args[pos[1]])
        return (
            isinstance(self.g.node_table.get(w), PlaceholderOp)
            and self.param_index(w) is not None
        )

    # -- rewrites -----------------------------------------------------------

    def rewrite_qkv(self):
        """Three biased projections of one input -> one "multi" kernel."""
        groups = {}
        for node in list(self.g.body):
            if isinstance(node, AddMMOp) and self.is_linear(node):
                lhs = self.g.node_table[str(node.args[1])]
                src = str(lhs.args[0]) if isinstance(lhs, ViewOp) else lhs.name
                groups.setdefault(src, []).append(node)
        for nodes in groups.values():
            if len(nodes) != 3:
                continue
            ws = [str(n.args[2]) for n in nodes]
            bs = [str(n.args[0]) for n in nodes]
            mats = [self.weight(w) for w in ws]
            kdim = mats[0].shape[0]
            ns = [mm.shape[1] for mm in mats]
            # one quantization of [Wq | Wk | Wv]; tiles never straddle two
            # parts, as each N is a multiple of NB
            packed = pack_q4(numpy.concatenate(mats, axis=1))
            bias = numpy.concatenate([self.weight(b) for b in bs])
            lhs = str(nodes[0].args[1])
            spec = kernel_spec("multi", self.m, kdim, ns, True, self.threads)
            for n in nodes:
                self.unlink(n)
                n._parents = []
            self.set_param(ws[0], packed, TensorDType.Int8)
            call = self.call(
                spec, [lhs, ws[0], bs[0]], [(self.m, n) for n in ns], nodes[0]
            )
            for i, n in enumerate(nodes):
                self.getitem(n, call, i, [self.m, ns[i]])
            self.set_param(
                bs[0], bias.astype(numpy.float32), TensorDType.Float32
            )
            for w in ws[1:] + bs[1:]:
                self.drop_param(w)

    def rewrite_glu(self):
        """mul(mul(vg, sigmoid(vg)), vu) with vg = view(mm(x, Wg)),
        vu = view(mm(x', Wu)) -> one "glu" kernel."""
        for node in list(self.g.body):
            if (
                not isinstance(node, MulOp)
                or node.name not in self.g.node_table
            ):
                continue
            a = self.g.node_table.get(str(node.args[0]))
            vu = self.g.node_table.get(str(node.args[1]))
            if not isinstance(a, MulOp) or not isinstance(vu, ViewOp):
                continue
            vg = self.g.node_table.get(str(a.args[0]))
            sg = self.g.node_table.get(str(a.args[1]))
            if not isinstance(vg, ViewOp) or not isinstance(sg, SigmoidOp):
                continue
            mg = self.g.node_table.get(str(vg.args[0]))
            mu = self.g.node_table.get(str(vu.args[0]))
            if not (
                isinstance(mg, MatmulOp)
                and isinstance(mu, MatmulOp)
                and self.is_linear(mg)
                and self.is_linear(mu)
            ):
                continue
            wg, wu = str(mg.args[1]), str(mu.args[1])
            Wg, Wu = self.weight(wg), self.weight(wu)
            kdim, n = Wg.shape
            spec = kernel_spec("glu", self.m, kdim, [n], False, self.threads)
            dead = [a, sg, vg, mg, vu, mu]
            lhs_u = self.g.node_table[str(mu.args[0])]
            self.unlink(node)
            node._parents = []
            for d in (mg, mu):
                self.unlink(d)
                d._parents = []
            self.set_param(wg, pack_glu(Wg, Wu), TensorDType.Int8)
            call = self.call(spec, [str(mg.args[0]), wg], [(self.m, n)], node)
            view = ViewOp()
            view._name = node.name
            shape = list(node.tensor_meta["shape"])
            view._arguments = [call.name, shape]
            view._tensor_meta = {
                "shape": torch.Size(shape),
                "dtype": TensorDType.Float32,
            }
            self.replace(node, view)
            view._parents = [call.name]
            call._children.append(view.name)
            for d in dead:
                d._children = []
                self.delete(d)
            if not lhs_u._children:
                self.delete(lhs_u)
            self.drop_param(wu)

    def rewrite_plain(self):
        """Every other Linear -> one "plain" kernel."""
        for node in list(self.g.body):
            if not isinstance(node, MatmulOp) or not self.is_linear(node):
                continue
            w = str(node.args[1])
            W = self.weight(w)
            kdim, n = W.shape
            spec = kernel_spec("plain", self.m, kdim, [n], False, self.threads)
            lhs = str(node.args[0])
            self.unlink(node)
            node._parents = []
            self.set_param(w, pack_q4(W), TensorDType.Int8)
            call = CallExternalOp(
                call_func_name=spec["name"],
                args=[lhs, w],
                args_index=[0, 0],
                tensor_meta={
                    "shape": [self.m, n],
                    "dtype": TensorDType.Float32,
                },
                name=node.name,
            )
            self.kernels[spec["name"]] = spec
            self.replace(node, call)
            self.link(call, [lhs, w])

    def _rope_input(self, name):
        """x of rope(x) = x * cos + rotate_half(x) * sin (HF Qwen2)."""
        add = self.g.node_table[name]
        if not isinstance(add, _AddOp):
            return None
        mul = self.g.node_table[str(add.args[0])]
        if not isinstance(mul, MulOp):
            return None
        return str(mul.args[0])

    def _inv_freq(self, dim):
        for p in self.g.params:
            if tuple(p.tensor_meta["shape"]) == (dim // 2,) and p._children:
                return p.name
        return None

    _HEAD_VIEWS = (ViewOp, ReshapeOp, PermuteOp)

    def _shape(self, name):
        return list(self.g.node_table[name].tensor_meta["shape"])

    def _matrix_above(self, name, cols):
        """The [m, cols] matrix that the head view `name` ([1, heads, m, d])
        is a view of: up through view / reshape / permute nodes."""
        node = self.g.node_table[name]
        while isinstance(node, self._HEAD_VIEWS):
            node = self.g.node_table[str(node.args[0])]
            if self._shape(node.name) == [self.m, cols]:
                return node.name
        raise RuntimeError(f"k3_w4: no [{self.m}, {cols}] matrix above {name}")

    def _matrix_below(self, node, cols):
        """The [m, cols] view of attention output `node` ([1, heads, m, d])
        that the o projection reads: down through getitem / view / reshape /
        permute nodes, each with a single user."""
        while self._shape(node.name) != [self.m, cols]:
            users = [self.g.node_table[c] for c in node._children]
            if len(users) != 1 or not isinstance(
                users[0], (GetItemOp,) + self._HEAD_VIEWS
            ):
                raise RuntimeError(
                    f"k3_w4: no [{self.m}, {cols}] view below {node.name}"
                )
            node = users[0]
        return node

    def rewrite_attention(self, pos_input: str):
        """RoPE + KV cache update + SDPA -> one attention kernel.

        The kernel gets the un-rotated q / k / v as the [m, heads * dim]
        outputs of their projections, the incoming KV caches, the start
        position and inv_freq; it rotates q and k, writes k and v into the
        caches in place, attends over the positions up to the current one
        only, and returns (out [m, heads * dim], lse, k_cache, v_cache). The
        view the o projection reads and the index_put nodes become getitems
        of the call; RoPE, cos / sin and the head views become dead."""
        for node in list(self.g.body):
            if not isinstance(node, ScaledDotProductFlashAttentionForCpuOp):
                continue
            puts = []
            for view in node.args[1:3]:
                n = self.g.node_table[str(view)]
                while not isinstance(n, IndexPutOp):
                    n = self.g.node_table[n._parents[0]]
                puts.append(n)
            kput, vput = puts
            q_pre = self._rope_input(str(node.args[0]))
            k_pre = self._rope_input(str(kput.args[2]))
            if q_pre is None or k_pre is None:
                raise RuntimeError("k3_w4: unexpected RoPE / attention pattern")
            _, heads, m, dim = self._shape(q_pre)
            cshape = self._shape(kput.name)
            kv_heads, ctx = cshape[1], cshape[2]
            inv_freq = self._inv_freq(dim)
            if inv_freq is None:
                raise RuntimeError("k3_w4: RoPE inv_freq parameter not found")
            q2 = self._matrix_above(q_pre, heads * dim)
            k2 = self._matrix_above(k_pre, kv_heads * dim)
            v2 = self._matrix_above(str(vput.args[2]), kv_heads * dim)
            out = self._matrix_below(node, heads * dim)
            scale = float(node.kwargs.get("scale", dim**-0.5))
            spec = {
                "name": f"k3_attn_m{m}_h{heads}_kv{kv_heads}_d{dim}_c{ctx}",
                "kind": "attn",
                "m": m,
                "heads": heads,
                "kv_heads": kv_heads,
                "dim": dim,
                "scale": scale,
                "ctx": ctx,
            }
            self.kernels[spec["name"]] = spec
            args = [
                q2,
                k2,
                v2,
                str(kput.args[0]),
                str(vput.args[0]),
                pos_input,
                inv_freq,
            ]
            self.uid += 1
            call = CallExternalOp(
                call_func_name=spec["name"],
                args=args,
                args_index=[0] * len(args),
                tensor_meta={
                    "shape": [[m, heads * dim], [1, heads, m], cshape, cshape],
                    "dtype": [TensorDType.Float32] * 4,
                },
                name=f"k3attn_{self.uid}",
            )
            first = min(self.g.body.index(kput), self.g.body.index(vput))
            self.insert_before(self.g.body[first], call)
            self.link(call, args)
            self.getitem(kput, call, 2, cshape)
            self.getitem(vput, call, 3, cshape)
            self.getitem(out, call, 0, [m, heads * dim])

    def _rmsnorm_of(self, name):
        """If `name` is view(w * (x * rsqrt(mean(x^2) + eps))) with a
        parameter w, return (x, w, eps), else None."""
        v = self.g.node_table.get(name)
        if isinstance(v, ViewOp):
            v = self.g.node_table.get(str(v.args[0]))
        while v is not None and type(v).__name__ == "AliasOp":
            v = self.g.node_table.get(str(v.args[0]))
        if not isinstance(v, MulOp):
            return None
        a, b = (self.g.node_table.get(str(x)) for x in v.args[:2])
        w, inner = (a, b) if isinstance(a, PlaceholderOp) else (b, a)
        if not isinstance(w, PlaceholderOp) or not isinstance(inner, MulOp):
            return None
        x = self.g.node_table.get(str(inner.args[0]))
        r = self.g.node_table.get(str(inner.args[1]))
        if not isinstance(r, RsqrtOp):
            return None
        ad = self.g.node_table.get(str(r.args[0]))
        if not isinstance(ad, _AddOp):
            return None
        mean = self.g.node_table.get(str(ad.args[0]))
        if not isinstance(mean, MeanOp):
            return None
        pw = self.g.node_table.get(str(mean.args[0]))
        if not (isinstance(pw, PowOp) and str(pw.args[0]) == x.name):
            return None
        return x.name, w.name, float(ad.args[1])

    def fuse_rmsnorm(self):
        """call(view(rmsnorm(x)), W, ...) -> call(view(x), W, ..., norm_w)."""
        for node in list(self.g.body):
            if (
                not isinstance(node, CallExternalOp)
                or "_q4_" not in node.call_func_name
            ):
                continue
            found = self._rmsnorm_of(str(node.args[0]))
            if found is None:
                continue
            x, w, eps = found
            spec = dict(self.kernels[node.call_func_name])
            m, k = spec["m"], spec["k"]
            spec["norm_eps"] = eps
            spec["name"] = spec["name"] + "_rms"
            self.kernels[spec["name"]] = spec
            self.uid += 1
            view = ViewOp()
            view._name = f"k3view_{self.uid}"
            view._arguments = [x, [m, k]]
            view._tensor_meta = {
                "shape": torch.Size([m, k]),
                "dtype": TensorDType.Float32,
            }
            self.insert_before(node, view)
            self.link(view, [x])
            old_lhs = str(node.args[0])
            self.g.node_table[old_lhs]._children.remove(node.name)
            node._arguments = [view.name] + list(node.args[1:]) + [w]
            node._args_index = [0] * len(node._arguments)
            node._parents = (
                [view.name] + [p for p in node._parents if p != old_lhs] + [w]
            )
            self.g.node_table[w]._children.append(node.name)
            view._children.append(node.name)
            node.call_func_name = spec["name"]
        # drop the specs no call uses any more
        used = {
            n.call_func_name
            for n in self.g.body
            if isinstance(n, CallExternalOp)
        }
        self.kernels = {k: v for k, v in self.kernels.items() if k in used}

    def remove_dead(self):
        """Delete computed nodes nothing uses (outputs are children)."""
        changed = True
        while changed:
            changed = False
            for node in list(self.g.body):
                if isinstance(node, (PlaceholderOp, OutputOp)):
                    continue
                if not node._children:
                    self.delete(node)
                    changed = True


def k3_w4_rewrite(graph: Graph, m: int, pos_input: str, threads: int) -> list:
    """Rewrite the Linear layers and attention of `graph` (m rows per call,
    start position in the graph input `pos_input`) to kernel calls; the
    kernels split their work over `threads` threads.

    Returns the kernel specs used. Must run after eliminate_transpose (the
    weights are [K, N]) and identically on graphs that share one weight
    layout (prefill and decode).
    """
    rw = _Rewriter(graph, m, threads)
    rw.rewrite_qkv()
    rw.rewrite_glu()
    rw.rewrite_plain()
    rw.rewrite_attention(pos_input)
    rw.fuse_rmsnorm()
    rw.remove_dead()
    graph._enable_external_calls = True
    return list(rw.kernels.values())


# ---------------------------------------------------------------------------
# Kernel generation, with the MLIR Python bindings
# ---------------------------------------------------------------------------
#
# Each kernel is a func.func of one module. The helpers below create
# operations at the current insertion point and return their result values.


def _vec(n, t):
    return ir.VectorType.get([n], t)


def _memref(shape, t, strides=None):
    """A memref type; `strides` (None: identity layout) and the offset are
    dynamic where they are None."""
    if strides is None:
        return ir.MemRefType.get(shape, t)
    dyn = ir.ShapedType.get_dynamic_stride_or_offset()
    layout = ir.StridedLayoutAttr.get(
        dyn, [dyn if s is None else s for s in strides]
    )
    return ir.MemRefType.get(shape, t, layout=layout)


class _Types:
    """The types of the kernels (created in the active context)."""

    def __init__(self):
        d = ir.ShapedType.get_dynamic_size()
        self.index = ir.IndexType.get()
        self.i1 = ir.IntegerType.get_signless(1)
        self.i8 = ir.IntegerType.get_signless(8)
        self.i16 = ir.IntegerType.get_signless(16)
        self.i32 = ir.IntegerType.get_signless(32)
        self.i64 = ir.IntegerType.get_signless(64)
        self.f16 = ir.F16Type.get()
        self.f32 = ir.F32Type.get()
        # memref<?x?xf32, strided<[?, 1], offset: ?>> and friends: the
        # layouts the graph passes its tensors with
        self.mat = _memref([d, d], self.f32, [None, 1])
        self.bytes = _memref([d], self.i8, [1])
        self.row = _memref([d], self.f32, [1])
        self.cache = _memref([d, d, d, d], self.f32, [None, None, None, 1])
        self.pos = _memref([d], self.i64, [None])


class _Fn:
    """One func.func under construction: its arguments, and constants
    created once each at the start of its body, in creation order."""

    def __init__(self, name, args, results, public=True):
        self.op = func.FuncOp(name, ir.FunctionType.get(args, results))
        if public:
            self.op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        else:
            self.op.attributes["sym_visibility"] = ir.StringAttr.get("private")
        self.entry = self.op.add_entry_block()
        self.args = list(self.entry.arguments)
        self._consts = {}

    def const(self, t, value):
        """A scalar constant, or a splat if `t` is a vector type."""
        key = (str(t), value)
        if key not in self._consts:
            is_vector = isinstance(t, ir.VectorType)
            elt = t.element_type if is_vector else t
            if isinstance(elt, ir.FloatType):
                attr = ir.FloatAttr.get(elt, value)
            else:
                attr = ir.IntegerAttr.get(elt, int(value))
            if is_vector:
                attr = ir.DenseElementsAttr.get_splat(t, attr)
            # after the constants created before, in creation order
            ops = self.entry.operations
            n = len(self._consts)
            ip = (
                ir.InsertionPoint(ops[n])
                if n < len(ops)
                else ir.InsertionPoint(self.entry)
            )
            with ip:
                self._consts[key] = arith.ConstantOp(t, attr).result
        return self._consts[key]

    def idx(self, value):
        return self.const(ir.IndexType.get(), value)


def _addi(a, b):
    return arith.AddIOp(a, b).result


def _subi(a, b):
    return arith.SubIOp(a, b).result


def _muli(a, b):
    return arith.MulIOp(a, b).result


def _divui(a, b):
    return arith.DivUIOp(a, b).result


def _addf(a, b):
    return arith.AddFOp(a, b).result


def _subf(a, b):
    return arith.SubFOp(a, b).result


def _mulf(a, b):
    return arith.MulFOp(a, b).result


def _divf(a, b):
    return arith.DivFOp(a, b).result


def _fma(a, b, c):
    return math.FmaOp(a, b, c).result


def _bcast(t, v):
    return vector.BroadcastOp(t, v).result


def _reduce(t, kind, v, reassoc=False):
    fastmath = (
        ir.Attribute.parse("#arith.fastmath<reassoc>") if reassoc else None
    )
    return vector.ReductionOp(
        t, ir.Attribute.parse(f"#vector.kind<{kind}>"), v, fastmath=fastmath
    ).result


def _read(t, mem, indices, pad, column=False):
    """vector.transfer_read of a 1-D vector along the last dimension of
    `mem`, or along the first one of a 2-D `mem` (column)."""
    rank = ir.MemRefType(mem.type).rank
    perm = (
        ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(0)])
        if column
        else ir.AffineMap.get_minor_identity(rank, 1)
    )
    return vector.TransferReadOp(t, mem, indices, perm, pad, [True]).result


def _write(v, mem, indices, column=False):
    rank = ir.MemRefType(mem.type).rank
    perm = (
        ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(0)])
        if column
        else ir.AffineMap.get_minor_identity(rank, 1)
    )
    vector.TransferWriteOp(None, v, mem, indices, perm, [True])


def _alloc(t):
    return memref.AllocOp(t, [], [], alignment=128).result


def _parallel(fn, his):
    """scf.parallel over [0, hi) for each hi; returns (op, induction vars).
    The body must end with scf.ReduceOp([], 0)."""
    n = len(his)
    op = scf.ParallelOp([], [fn.idx(0)] * n, his, [fn.idx(1)] * n, [])
    block = op.region.blocks.append(*[ir.IndexType.get()] * n)
    return block, list(block.arguments)


def _exp(fn, ty, x, lanes):
    """exp(x) for vector<lanes x f32> (|x| <= 87): 2^n * p(r), p degree 6."""
    vf, vi = _vec(lanes, ty.f32), _vec(lanes, ty.i32)
    xd = arith.MinimumFOp(
        arith.MaximumFOp(x, fn.const(vf, -87.0)).result, fn.const(vf, 87.0)
    ).result
    nf = math.RoundEvenOp(_mulf(xd, fn.const(vf, 1.4426950408889634))).result
    r = _fma(nf, fn.const(vf, -0.6931471805599453), xd)
    coeffs = [1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0]
    p = fn.const(vf, coeffs[0])
    for c in coeffs[1:]:
        p = _fma(p, r, fn.const(vf, c))
    n = arith.FPToSIOp(vi, nf).result
    bits = arith.ShLIOp(_addi(n, fn.const(vi, 127)), fn.const(vi, 23)).result
    return _mulf(p, arith.BitcastOp(vf, bits).result)


def _silu_mul(fn, ty, g, u, lanes):
    """silu(g) * u = g / (1 + exp(-g)) * u."""
    vf = _vec(lanes, ty.f32)
    e = _exp(fn, ty, arith.NegFOp(g).result, lanes)
    return _mulf(_divf(g, _addf(e, fn.const(vf, 1.0))), u)


def _act_types(ty, spec):
    """The quantized activations: int8 values and one f32 scale per group."""
    m, k = spec["m"], spec["k"]
    return (
        _memref([m, k], ty.i8),
        _memref([m, k // G], ty.f32),
    )


def _tile_types(ty, spec):
    t = [ty.bytes, *_act_types(ty, spec), ty.index, ty.mat, ty.index]
    if spec["kind"] == "glu":
        t.append(ty.mat)
    if spec["bias"]:
        t += [ty.row, ty.index]
    return t


def _accumulate(fn, ty, spec, a, wtile, rows):
    """f32 accumulators (vector<NB x f32>) of `rows` over all groups of
    weight tile `wtile`."""
    k = spec["k"]
    vf, v8, v16, v32 = (_vec(NB, t) for t in (ty.f32, ty.i8, ty.i16, ty.i32))
    four = fn.const(v8, 4)
    tbase = _muli(wtile, fn.idx(tile_bytes(k)))
    zero = fn.const(vf, 0.0)
    loop = scf.ForOp(fn.idx(0), fn.idx(k // G), fn.idx(1), [zero] * len(rows))
    with ir.InsertionPoint(loop.body):
        g = loop.induction_variable
        gbase = _addi(tbase, _muli(g, fn.idx(G // 2 * NB + NB * 2)))
        kg = _muli(g, fn.idx(G))
        acc16 = [fn.const(v16, 0)] * len(rows)
        for p in range(G // 2):
            # 4 row pairs (512 bytes, contiguous) per load: a vector load
            # costs about the same whatever its size on the A100, and with
            # the exact VLEN given to llc the 128-byte slices are register
            # halves of one vl4r.
            if p % 4 == 0:
                wide = vector.LoadOp(
                    _vec(4 * NB, ty.i8), a["w"], [_addi(gbase, fn.idx(p * NB))]
                ).result
            w = vector.ExtractStridedSliceOp(
                v8, wide, [(p % 4) * NB], [NB], [1]
            ).result
            lo = arith.ShRSIOp(arith.ShLIOp(w, four).result, four).result
            hi = arith.ShRSIOp(w, four).result
            halves = [arith.ExtSIOp(v16, h).result for h in (lo, hi)]
            for i, row in enumerate(rows):
                for j, half in enumerate(halves):
                    xb = memref.LoadOp(
                        a["xq"], [row, _addi(kg, fn.idx(2 * p + j))]
                    ).result
                    xv = _bcast(v16, arith.ExtSIOp(ty.i16, xb).result)
                    acc16[i] = _addi(acc16[i], _muli(half, xv))
        raw = vector.LoadOp(
            _vec(2 * NB, ty.i8), a["w"], [_addi(gbase, fn.idx(G // 2 * NB))]
        ).result
        scale = arith.ExtFOp(
            vf, vector.BitCastOp(_vec(NB, ty.f16), raw).result
        ).result
        outs = []
        for i, row in enumerate(rows):
            xs = _bcast(vf, memref.LoadOp(a["xs"], [row, g]).result)
            # the scale before the conversion: with the conversion first, llc
            # allocates the registers of the 4-row prefill tiles worse
            # (prefill ~5% slower on the K3)
            sx = _mulf(scale, xs)
            prod = arith.SIToFPOp(
                vf, arith.ExtSIOp(v32, acc16[i]).result
            ).result
            outs.append(_fma(prod, sx, loop.inner_iter_args[i]))
        scf.YieldOp(outs)
    return list(loop.results)


def _tile_fn(ty, spec):
    """One weight tile (NB output columns) for all the rows."""
    m = spec["m"]
    mb = 1 if m == 1 else PREFILL_MB
    assert m % mb == 0, (m, mb)
    names = ["w", "xq", "xs", "wt", "out", "ocol"]
    if spec["kind"] == "glu":
        names.append("out2")
    if spec["bias"]:
        names += ["bias", "bcol"]
    fn = _Fn(f"{spec['name']}__tile", _tile_types(ty, spec), [], public=False)
    a = dict(zip(names, fn.args))
    vf = _vec(NB, ty.f32)
    with ir.InsertionPoint(fn.entry):
        loop = scf.ForOp(fn.idx(0), fn.idx(m), fn.idx(mb))
        with ir.InsertionPoint(loop.body):
            rows = [
                _addi(loop.induction_variable, fn.idx(i)) for i in range(mb)
            ]
            if spec["kind"] == "glu":
                tg = _muli(a["wt"], fn.idx(2))
                gate = _accumulate(fn, ty, spec, a, tg, rows)
                up = _accumulate(fn, ty, spec, a, _addi(tg, fn.idx(1)), rows)
                vals = [_silu_mul(fn, ty, g, u, NB) for g, u in zip(gate, up)]
            else:
                vals = _accumulate(fn, ty, spec, a, a["wt"], rows)
                if spec["bias"]:
                    bias = _read(
                        vf, a["bias"], [a["bcol"]], fn.const(ty.f32, 0.0)
                    )
                    vals = [_addf(v, bias) for v in vals]
            for row, v in zip(rows, vals):
                vector.StoreOp(v, a["out"], [row, a["ocol"]])
            scf.YieldOp([])
        func.ReturnOp([])


def _matmul_fn(ty, spec):
    """One matmul kernel: quantize x (with the fused RMSNorm), then the
    tiles on all threads."""
    m, k, ns, kind = spec["m"], spec["k"], spec["ns"], spec["kind"]
    assert k % G == 0 and all(n % NB == 0 for n in ns), spec
    groups, threads = k // G, spec["threads"]
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
    vg = _vec(G, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    with ir.InsertionPoint(fn.entry):
        ys = [_alloc(t) for t in outs]
        xq_t, xs_t = _act_types(ty, spec)
        xq, xs = _alloc(xq_t), _alloc(xs_t)

        # quantize x: one row serially (a parallel region costs more)
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

        # the tiles on all threads: thread i gets tiles
        # [tiles * i / threads, tiles * (i + 1) / threads)
        tiles = ns[0] // NB if kind == "glu" else sum(ns) // NB
        body, (slot,) = _parallel(fn, [fn.idx(threads)])
        with ir.InsertionPoint(body):
            ntiles, nthreads = fn.idx(tiles), fn.idx(threads)
            lo = _divui(_muli(slot, ntiles), nthreads)
            hi = _divui(_muli(_addi(slot, fn.idx(1)), ntiles), nthreads)
            tl = scf.ForOp(lo, hi, fn.idx(1))
            with ir.InsertionPoint(tl.body):
                t = tl.induction_variable
                col = _muli(t, fn.idx(NB))
                dsts = [memref.CastOp(ty.mat, y).result for y in ys]
                # multi: the output the tile belongs to, and its column there
                dst = dsts[-1]
                dcol = _subi(col, fn.idx(sum(ns[:-1])))
                for i in range(len(ns) - 2, -1, -1):
                    inside = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        t,
                        fn.idx(sum(ns[: i + 1]) // NB),
                    ).result
                    dst = arith.SelectOp(inside, dsts[i], dst).result
                    dcol = arith.SelectOp(
                        inside, _subi(col, fn.idx(sum(ns[:i]))), dcol
                    ).result
                args = [a["w"], xq, xs, t, dst, dcol]
                if kind == "glu":
                    args.append(dst)
                if spec["bias"]:
                    args += [a["bias"], col]
                func.CallOp([], f"{spec['name']}__tile", args)
                scf.YieldOp([])
            scf.ReduceOp([], 0)
        memref.DeallocOp(xq)
        memref.DeallocOp(xs)
        func.ReturnOp(ys)


def _attn_fn(ty, spec):
    """The attention kernel's function, its arguments, and the prologue
    shared by decode and prefill: the outputs, the start position, cos / sin
    of (start + i) * inv_freq in %cs / %sn [m][d/2], and the rotated k and
    the v of the m new rows (kp / vp [m, kv_heads * d]) written into the KV
    caches kc / vc at positions start + i (in place)."""
    m, h, kvh, d = spec["m"], spec["heads"], spec["kv_heads"], spec["dim"]
    hd = d // 2
    cache = _memref([1, kvh, spec["ctx"], d], ty.f32)
    outs = [
        _memref([m, h * d], ty.f32),
        _memref([1, h, m], ty.f32),
        cache,
        cache,
    ]
    names = ["q", "kp", "vp", "kc", "vc", "pos", "if"]
    types = [ty.mat, ty.mat, ty.mat, ty.cache, ty.cache, ty.pos, ty.row]
    fn = _Fn(spec["name"], types, outs)
    a = dict(zip(names, fn.args))
    vh = _vec(hd, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    with ir.InsertionPoint(fn.entry):
        a["o"] = _alloc(outs[0])
        a["lse"] = _alloc(outs[1])
        a["start"] = arith.IndexCastOp(
            ty.index, memref.LoadOp(a["pos"], [fn.idx(0)]).result
        ).result
        cs_t = _memref([m, hd], ty.f32)
        a["cs"], a["sn"] = _alloc(cs_t), _alloc(cs_t)
        invf = _read(vh, a["if"], [fn.idx(0)], f0)
        if m == 1:
            rl = scf.ForOp(fn.idx(0), fn.idx(1), fn.idx(1))
            body, ri = rl.body, rl.induction_variable
        else:
            body, (ri,) = _parallel(fn, [fn.idx(m)])
        with ir.InsertionPoint(body):
            pp = _addi(a["start"], ri)
            pf = arith.SIToFPOp(
                ty.f32, arith.IndexCastOp(ty.i64, pp).result
            ).result
            ang = _mulf(invf, _bcast(vh, pf))
            cv, sv = math.CosOp(ang).result, math.SinOp(ang).result
            vector.StoreOp(cv, a["cs"], [ri, fn.idx(0)])
            vector.StoreOp(sv, a["sn"], [ri, fn.idx(0)])
            hl = scf.ForOp(fn.idx(0), fn.idx(kvh), fn.idx(1))
            with ir.InsertionPoint(hl.body):
                c_lo = _muli(hl.induction_variable, fn.idx(d))
                c_hi = _addi(c_lo, fn.idx(hd))
                klo = _read(vh, a["kp"], [ri, c_lo], f0)
                khi = _read(vh, a["kp"], [ri, c_hi], f0)
                rlo = _fma(klo, cv, arith.NegFOp(_mulf(khi, sv)).result)
                rhi = _fma(khi, cv, _mulf(klo, sv))
                at = [fn.idx(0), hl.induction_variable, pp]
                _write(rlo, a["kc"], at + [fn.idx(0)])
                _write(rhi, a["kc"], at + [fn.idx(hd)])
                _write(
                    _read(vh, a["vp"], [ri, c_lo], f0),
                    a["vc"],
                    at + [fn.idx(0)],
                )
                _write(
                    _read(vh, a["vp"], [ri, c_hi], f0),
                    a["vc"],
                    at + [fn.idx(hd)],
                )
                scf.YieldOp([])
            if m == 1:
                scf.YieldOp([])
            else:
                scf.ReduceOp([], 0)
    return fn, a, outs


def _attn_epilogue(fn, a, outs):
    memref.DeallocOp(a["cs"])
    memref.DeallocOp(a["sn"])
    kco = memref.CastOp(outs[2], a["kc"]).result
    vco = memref.CastOp(outs[3], a["vc"]).result
    func.ReturnOp([a["o"], a["lse"], kco, vco])


def _attn_decode_fn(ty, spec, block=16):
    """Causal attention of m query rows (positions start .. start + m - 1)
    against the updated KV cache, one (head, row) per work item: keys
    0 .. start + i only, online softmax over blocks of `block` keys (one
    rescale per block, vectorized exp), then a per-key tail. Used for decode
    (m == 1)."""
    m, h, kvh, d = spec["m"], spec["heads"], spec["kv_heads"], spec["dim"]
    hd, B = d // 2, block
    fn, a, outs = _attn_fn(ty, spec)
    vd, vb, vh = _vec(d, ty.f32), _vec(B, ty.f32), _vec(hd, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    c0 = fn.idx(0)
    with ir.InsertionPoint(fn.entry):
        body, (hh, i) = _parallel(fn, [fn.idx(h), fn.idx(m)])
        with ir.InsertionPoint(body):
            kh = _divui(hh, fn.idx(h // kvh))
            q_lo = _muli(hh, fn.idx(d))
            qa = _read(vh, a["q"], [i, q_lo], f0)
            qb = _read(vh, a["q"], [i, _addi(q_lo, fn.idx(hd))], f0)
            ca = vector.LoadOp(vh, a["cs"], [i, c0]).result
            sa = vector.LoadOp(vh, a["sn"], [i, c0]).result
            rl = _fma(qa, ca, arith.NegFOp(_mulf(qb, sa)).result)
            rh = _fma(qb, ca, _mulf(qa, sa))
            qv = vector.InsertStridedSliceOp(
                rl, fn.const(vd, 0.0), [0], [1]
            ).result
            qv = vector.InsertStridedSliceOp(rh, qv, [hd], [1]).result
            qs = _mulf(qv, fn.const(vd, spec["scale"]))
            length = arith.MinUIOp(
                _addi(_addi(a["start"], i), fn.idx(1)), fn.idx(spec["ctx"])
            ).result
            nblk = _divui(length, fn.idx(B))
            ninf = fn.const(ty.f32, float("-inf"))

            def key(j):
                return _read(vd, a["kc"], [c0, kh, j, c0], f0)

            def value(j):
                return _read(vd, a["vc"], [c0, kh, j, c0], f0)

            def score(j):
                return _reduce(ty.f32, "add", _mulf(qs, key(j)), reassoc=True)

            # blocks of B keys
            bl = scf.ForOp(c0, nblk, fn.idx(1), [ninf, f0, fn.const(vd, 0.0)])
            with ir.InsertionPoint(bl.body):
                mx, l, acc = bl.inner_iter_args
                j0 = _muli(bl.induction_variable, fn.idx(B))
                s = fn.const(vb, 0.0)
                for u in range(B):
                    s = vector.InsertOp(
                        score(_addi(j0, fn.idx(u))), s, [], [u]
                    ).result
                mn = arith.MaximumFOp(mx, _reduce(ty.f32, "maximumf", s)).result
                alpha = math.ExpOp(_subf(mx, mn)).result
                p = _exp(fn, ty, _subf(s, _bcast(vb, mn)), B)
                acc = _mulf(acc, _bcast(vd, alpha))
                for u in range(B):
                    pu = vector.ExtractOp(p, [], [u]).result
                    acc = _fma(value(_addi(j0, fn.idx(u))), _bcast(vd, pu), acc)
                psum = _reduce(ty.f32, "add", p, reassoc=True)
                scf.YieldOp([mn, _addf(_mulf(l, alpha), psum), acc])
            # the remaining keys, one at a time
            tl = scf.ForOp(
                _muli(nblk, fn.idx(B)), length, fn.idx(1), list(bl.results)
            )
            with ir.InsertionPoint(tl.body):
                mx, l, acc = tl.inner_iter_args
                j = tl.induction_variable
                s = score(j)
                mn = arith.MaximumFOp(mx, s).result
                alpha = math.ExpOp(_subf(mx, mn)).result
                beta = math.ExpOp(_subf(s, mn)).result
                acc = _fma(
                    value(j), _bcast(vd, beta), _mulf(acc, _bcast(vd, alpha))
                )
                scf.YieldOp([mn, _addf(_mulf(l, alpha), beta), acc])
            mx, l, acc = tl.results
            vector.StoreOp(_divf(acc, _bcast(vd, l)), a["o"], [i, q_lo])
            lse = _addf(math.LogOp(l).result, mx)
            memref.StoreOp(lse, a["lse"], [c0, hh, i])
            scf.ReduceOp([], 0)
        _attn_epilogue(fn, a, outs)


def _attn_prefill_fn(ty, spec):
    """Causal attention for a prefill chunk (m > 1 rows at positions
    start .. start + m - 1) against the updated KV cache, vectorized across
    R query rows; per work item (head, R rows):
      S[j][r]   = scale * sum_d Q[r][d] K[j][d]   (Q^T 16 dims at a time)
      P[j][r]   = exp(S[j][r] - max_j), masked to j <= start + r
      O^T[d][r] = sum_j V[j][d] P[j][r] / sum_j P[j][r]
    """
    m, h, kvh, d, ctx = (
        spec["m"],
        spec["heads"],
        spec["kv_heads"],
        spec["dim"],
        spec["ctx"],
    )
    R, DB = ATTN_ROWS, ATTN_DIMS
    assert m % R == 0 and d % DB == 0, spec  # checked by gen_config.py
    hd = d // 2
    fn, a, outs = _attn_fn(ty, spec)
    vr, vri = _vec(R, ty.f32), _vec(R, ty.index)
    f0 = fn.const(ty.f32, 0.0)
    c0, c1 = fn.idx(0), fn.idx(1)
    zero = fn.const(vr, 0.0)
    ninf = fn.const(vr, float("-inf"))
    with ir.InsertionPoint(fn.entry):
        iota = vector.StepOp(vri).result
        body, (hh, rb) = _parallel(fn, [fn.idx(h), fn.idx(m // R)])
        with ir.InsertionPoint(body):
            kh = _divui(hh, fn.idx(h // kvh))
            i0 = _muli(rb, fn.idx(R))
            p0 = _addi(a["start"], i0)
            length = _addi(p0, fn.idx(R))  # keys 0 .. start + i0 + R - 1
            qt = _alloc(_memref([d, R], ty.f32))
            sp = _alloc(_memref([ctx, R], ty.f32))
            scale = fn.const(vr, spec["scale"])
            q_lo = _muli(hh, fn.idx(d))
            # rotated, scaled Q^T: rows dd and dd + d / 2
            rl = scf.ForOp(c0, fn.idx(hd), c1)
            with ir.InsertionPoint(rl.body):
                dd = rl.induction_variable
                dh = _addi(dd, fn.idx(hd))
                qa = _read(vr, a["q"], [i0, _addi(q_lo, dd)], f0, column=True)
                qb = _read(vr, a["q"], [i0, _addi(q_lo, dh)], f0, column=True)
                ca = _read(vr, a["cs"], [i0, dd], f0, column=True)
                sa = _read(vr, a["sn"], [i0, dd], f0, column=True)
                lo = _fma(qa, ca, arith.NegFOp(_mulf(qb, sa)).result)
                hi = _fma(qb, ca, _mulf(qa, sa))
                vector.StoreOp(_mulf(lo, scale), qt, [dd, c0])
                vector.StoreOp(_mulf(hi, scale), qt, [dh, c0])
                scf.YieldOp([])
            # scores, 16 dims of Q^T in registers per pass
            sl = scf.ForOp(c0, fn.idx(d), fn.idx(DB))
            with ir.InsertionPoint(sl.body):
                db = sl.induction_variable
                dims = [_addi(db, fn.idx(u)) for u in range(DB)]
                qs = [vector.LoadOp(vr, qt, [di, c0]).result for di in dims]
                first = arith.CmpIOp(arith.CmpIPredicate.eq, db, c0).result
                jl = scf.ForOp(c0, length, c1)
                with ir.InsertionPoint(jl.body):
                    j = jl.induction_variable
                    old = vector.LoadOp(vr, sp, [j, c0]).result
                    acc = arith.SelectOp(first, zero, old).result
                    for di, qv in zip(dims, qs):
                        kd = memref.LoadOp(a["kc"], [c0, kh, j, di]).result
                        acc = _fma(qv, _bcast(vr, kd), acc)
                    vector.StoreOp(acc, sp, [j, c0])
                    scf.YieldOp([])
                scf.YieldOp([])
            # softmax over j <= start + i0 + r
            posv = _addi(_bcast(vri, p0), iota)

            def visible(j):
                return arith.CmpIOp(
                    arith.CmpIPredicate.ule, _bcast(vri, j), posv
                ).result

            ml = scf.ForOp(c0, length, c1, [ninf])
            with ir.InsertionPoint(ml.body):
                j = ml.induction_variable
                s = arith.SelectOp(
                    visible(j), vector.LoadOp(vr, sp, [j, c0]).result, ninf
                ).result
                scf.YieldOp([arith.MaximumFOp(ml.inner_iter_args[0], s).result])
            mx = ml.results[0]
            ll = scf.ForOp(c0, length, c1, [zero])
            with ir.InsertionPoint(ll.body):
                j = ll.induction_variable
                e = _exp(
                    fn, ty, _subf(vector.LoadOp(vr, sp, [j, c0]).result, mx), R
                )
                pj = arith.SelectOp(visible(j), e, zero).result
                vector.StoreOp(pj, sp, [j, c0])
                scf.YieldOp([_addf(ll.inner_iter_args[0], pj)])
            lsum = ll.results[0]
            # O^T = V^T P, 16 dims at a time
            ol = scf.ForOp(c0, fn.idx(d), fn.idx(DB))
            with ir.InsertionPoint(ol.body):
                db = ol.induction_variable
                dims = [_addi(db, fn.idx(u)) for u in range(DB)]
                vl = scf.ForOp(c0, length, c1, [zero] * DB)
                with ir.InsertionPoint(vl.body):
                    j = vl.induction_variable
                    pj = vector.LoadOp(vr, sp, [j, c0]).result
                    accs = []
                    for di, acc in zip(dims, vl.inner_iter_args):
                        vd = memref.LoadOp(a["vc"], [c0, kh, j, di]).result
                        accs.append(_fma(pj, _bcast(vr, vd), acc))
                    scf.YieldOp(accs)
                for di, acc in zip(dims, vl.results):
                    _write(
                        _divf(acc, lsum),
                        a["o"],
                        [i0, _addi(q_lo, di)],
                        column=True,
                    )
                scf.YieldOp([])
            # lse (not used by the model, written for completeness)
            lse = _addf(math.LogOp(lsum).result, mx)
            vector.StoreOp(lse, a["lse"], [c0, hh, i0])
            memref.DeallocOp(qt)
            memref.DeallocOp(sp)
            scf.ReduceOp([], 0)
        _attn_epilogue(fn, a, outs)


def build_kernels(specs) -> ir.Module:
    """The module of the kernels `specs` (k3_w4_rewrite) describe, in the
    active context; verified."""
    ty = _Types()
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        for s in specs:
            if s["kind"] == "attn":
                if s["m"] == 1:
                    _attn_decode_fn(ty, s)
                else:
                    _attn_prefill_fn(ty, s)
            else:
                _matmul_fn(ty, s)
                _tile_fn(ty, s)
    if not module.operation.verify():
        raise RuntimeError("k3_w4: the generated kernels do not verify")
    return module


def gen_kernels(specs) -> str:
    """build_kernels() in a context of its own, as MLIR text."""
    with ir.Context(), ir.Location.unknown():
        return str(build_kernels(specs))
