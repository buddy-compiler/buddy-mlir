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
from buddy_mlir.dialects import arith, func, llvm, math, memref, scf, vector

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
# "prefill_ime": the matrix-engine tiles cover this many rows (two halves of
# 32, 4 row blocks of 8 each); gen_config.py requires prefill_chunk == 64.
IME_ROWS = 64


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


# The IME layout of the prefill tiles ("prefill_ime"): per block of 8
# columns, per group: 128 bytes, byte c * 16 + j holding W[32g + j][8b + c]
# (low nibble) and W[32g + 16 + j][8b + c] (high nibble), i.e. the two 8 x 16
# int8 B operands of smt.vmadot, then the 8 f16 scales of the block.
IME_BLOCK = 128 + 16


def pack_ime(q, scale) -> numpy.ndarray:
    """The IME layout of an int4 matrix (quantize_q4)."""
    groups, _, n = q.shape
    b = ((q[:, :16, :] & 0x0F) | ((q[:, 16:, :] & 0x0F) << 4)).astype(
        numpy.uint8
    )
    b = b.reshape(groups, 16, n // 8, 8).transpose(2, 0, 3, 1)  # nb, g, c, j
    sc = scale.reshape(groups, n // 8, 8).transpose(1, 0, 2)  # nb, g, c
    sc = numpy.ascontiguousarray(sc).view(numpy.uint8)  # nb, g, 16
    out = numpy.concatenate([b.reshape(n // 8, groups, 128), sc], axis=2)
    return out.reshape(-1).view(numpy.int8)


def pack_q4_both(w: numpy.ndarray):
    """(tile layout, IME layout) of one quantization of w."""
    q, scale = quantize_q4(w)
    return pack_tiles(q, scale), pack_ime(q, scale)


def pack_glu_both(wg: numpy.ndarray, wu: numpy.ndarray):
    """(tile layout, IME layout) of gate and up: the tiles interleaved (g0,
    u0, g1, u1, ...); the IME blocks of gate, then those of up."""
    k, n = wg.shape
    tb = tile_bytes(k)
    g, gi = pack_q4_both(wg)
    u, ui = pack_q4_both(wu)
    tiles = numpy.stack(
        [g.reshape(n // NB, tb), u.reshape(n // NB, tb)], axis=1
    )
    return tiles.reshape(-1), numpy.concatenate([gi, ui])


def pack_glu(wg: numpy.ndarray, wu: numpy.ndarray) -> numpy.ndarray:
    """Gate and up tiles interleaved: g0, u0, g1, u1, ..."""
    return pack_glu_both(wg, wu)[0]


# ---------------------------------------------------------------------------
# Kernel specs
# ---------------------------------------------------------------------------


def kernel_spec(
    kind: str, m: int, k: int, ns: list, bias: bool, threads: int, ime=False
) -> dict:
    """A matmul kernel: kind "plain" (x W), "multi" (x [W0 | W1 | ...] with
    one result per part, q / k / v) or "glu" (silu(x Wg) * (x Wu)); with
    `ime`, its tiles run on the matrix engine (IME layout, m == IME_ROWS)."""
    name = f"k3_q4_{kind}_m{m}_k{k}_n{'_'.join(map(str, ns))}"
    if bias:
        name += "_b"
    if ime:
        assert m == IME_ROWS, m
        name += "_ime"
    return {
        "ime": ime,
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
    def __init__(self, graph, m, threads, use_ime=False, ime_weights=None):
        self.g = graph
        self.m = m
        self.threads = threads
        self.refs = graph._params_ref
        self.kernels = {}
        self.uid = 0
        # The kernels of m rows run on the matrix engine and record their
        # weights in ime_weights; or (the other graph) the weights in
        # ime_weights get the same second, IME-layout copy, unused.
        self.use_ime = use_ime
        self.ime_weights = set() if ime_weights is None else ime_weights

    def add_param(self, name, array: numpy.ndarray, dtype: TensorDType):
        """A new parameter, after the existing ones (the importer binds
        placeholders to arguments in body order)."""
        node = PlaceholderOp()
        node._name = name
        node._tensor_meta = {
            "shape": torch.Size(list(array.shape)),
            "dtype": dtype,
        }
        idx = max(self.g._fake_params) + 1
        self.g.body.insert(idx, node)
        self.g.node_table[name] = node
        self.g._inputs = [i + 1 if i >= idx else i for i in self.g._inputs]
        self.g._fake_params = [
            i + 1 if i >= idx else i for i in self.g._fake_params
        ]
        self.g._fake_params.append(idx)
        self.refs.append(torch.from_numpy(numpy.ascontiguousarray(array)))
        return name

    def set_weights(self, name, packed, rows):
        """Store the packed weight `name` (tile layout, IME layout); return
        the parameter a kernel of `rows` rows reads and whether it runs on
        the matrix engine."""
        tiles, ime = packed
        self.set_param(name, tiles, TensorDType.Int8)
        on_ime = self.use_ime and rows == self.m == IME_ROWS
        if on_ime:
            self.ime_weights.add(name)
        if name in self.ime_weights:
            self.add_param(name + "_ime", ime, TensorDType.Int8)
        return (name + "_ime" if on_ime else name), on_ime

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
            written_args=[],  # the kernels only read their arguments
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
            packed = pack_q4_both(numpy.concatenate(mats, axis=1))
            bias = numpy.concatenate([self.weight(b) for b in bs])
            lhs = str(nodes[0].args[1])
            for n in nodes:
                self.unlink(n)
                n._parents = []
            w, ime = self.set_weights(ws[0], packed, self.m)
            spec = kernel_spec(
                "multi", self.m, kdim, ns, True, self.threads, ime
            )
            call = self.call(
                spec, [lhs, w, bs[0]], [(self.m, n) for n in ns], nodes[0]
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
            dead = [a, sg, vg, mg, vu, mu]
            lhs_u = self.g.node_table[str(mu.args[0])]
            self.unlink(node)
            node._parents = []
            for d in (mg, mu):
                self.unlink(d)
                d._parents = []
            w, ime = self.set_weights(wg, pack_glu_both(Wg, Wu), self.m)
            spec = kernel_spec(
                "glu", self.m, kdim, [n], False, self.threads, ime
            )
            call = self.call(spec, [str(mg.args[0]), w], [(self.m, n)], node)
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
        """Every other Linear -> one "plain" kernel, with the rows of its
        input (the lm_head of a prefill chunk sees its last row only)."""
        for node in list(self.g.body):
            if not isinstance(node, MatmulOp) or not self.is_linear(node):
                continue
            w = str(node.args[1])
            W = self.weight(w)
            kdim, n = W.shape
            rows = list(node.tensor_meta["shape"])[0]
            lhs = str(node.args[0])
            self.unlink(node)
            node._parents = []
            w, ime = self.set_weights(w, pack_q4_both(W), rows)
            spec = kernel_spec(
                "plain", rows, kdim, [n], False, self.threads, ime
            )
            if rows == 1 and self.m > 1:
                # the lm_head of a prefill chunk: computed only when the
                # session needs the logits (buddy_set_prefill_logits)
                spec["name"] += "_prefill_logits"
                spec["prefill_logits"] = True
            call = CallExternalOp(
                call_func_name=spec["name"],
                args=[lhs, w],
                args_index=[0, 0],
                tensor_meta={
                    "shape": [rows, n],
                    "dtype": TensorDType.Float32,
                },
                name=node.name,
                written_args=[],
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
                "threads": self.threads,
            }
            if (
                self.use_ime
                and m == IME_ROWS
                and dim % ATTN_IME_DIMS == 0
                and ctx % ATTN_IME_KEYS == 0
            ):
                spec["name"] += "_ime"
                spec["ime"] = True
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
                # the KV caches, updated in place
                written_args=[3, 4],
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


def k3_w4_rewrite(
    graph: Graph,
    m: int,
    pos_input: str,
    threads: int,
    use_ime: bool = False,
    ime_weights: set = None,
) -> list:
    """Rewrite the Linear layers and attention of `graph` (m rows per call,
    start position in the graph input `pos_input`) to kernel calls; the
    kernels split their work over `threads` threads.

    "prefill_ime": with `use_ime`, the kernels of m (IME_ROWS) rows run on
    the matrix engine, on a second copy of their weights in the IME layout,
    and their weights are added to `ime_weights`. The other graph is then
    rewritten with that set, so that its weights get the same copies.

    Returns the kernel specs used. Must run after eliminate_transpose (the
    weights are [K, N]) and identically on graphs that share one weight
    layout (prefill and decode).
    """
    rw = _Rewriter(graph, m, threads, use_ime, ime_weights)
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
    """The quantized activations: int8 values and one f32 scale per group;
    for IME tiles, the A operands of smt.vmadot ([group][row block of 8]
    [k 0-15 | k 16-31][row % 8][16]) and the scale of each row 8 times (f16,
    [group][row block][row % 8][8])."""
    m, k = spec["m"], spec["k"]
    if spec.get("ime"):
        return (
            _memref([k // G, m // 8, 256], ty.i8),
            _memref([k // G, m // 8, 64], ty.f16),
        )
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
    if spec.get("ime"):
        t.append(ty.index)  # the first group of the pass (_ime_pass_groups)
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
    # "prefill_logits": nothing while the session does not need the logits
    # (the outputs are then left unwritten)
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
                scale = _divf(mx, c127)
                if spec.get("ime"):
                    rb = _divui(r, fn.idx(8))
                    at = _muli(arith.RemUIOp(r, fn.idx(8)).result, fn.idx(16))
                    for half in (0, 1):
                        part = vector.ExtractStridedSliceOp(
                            _vec(16, ty.i8), xi, [16 * half], [16], [1]
                        ).result
                        vector.StoreOp(
                            part, xq, [g, rb, _addi(at, fn.idx(128 * half))]
                        )
                    sh = arith.TruncFOp(ty.f16, scale).result
                    vector.StoreOp(
                        _bcast(_vec(8, ty.f16), sh),
                        xs,
                        [
                            g,
                            rb,
                            _muli(
                                arith.RemUIOp(r, fn.idx(8)).result, fn.idx(8)
                            ),
                        ],
                    )
                else:
                    vector.StoreOp(xi, xq, [r, kk])
                    memref.StoreOp(scale, xs, [r, g])
                scf.YieldOp([])
            if m == 1:
                scf.YieldOp([])
            else:
                scf.ReduceOp([], 0)

        # the tiles on all threads: thread i gets tiles
        # [tiles * i / threads, tiles * (i + 1) / threads); IME tiles in
        # passes over K, each after a copy of its activations into the TCM
        tw = _tile_width(spec)
        tiles = ns[0] // tw if kind == "glu" else sum(ns) // tw
        if spec.get("ime"):
            gp = _ime_pass_groups(groups)
            passes = [p * gp for p in range(groups // gp)]
        else:
            passes = [None]
        for g0 in passes:
            if g0 is not None:
                _ime_stage(fn, ty, xq, xs, g0, gp)
            _tiles(fn, ty, spec, a, xq, xs, ys, tw, tiles, threads, g0)
        memref.DeallocOp(xq)
        memref.DeallocOp(xs)
        if spec.get("prefill_logits"):
            scf.YieldOp([])
    with ir.InsertionPoint(fn.entry):
        func.ReturnOp(ys)


# The lm_head of a prefill chunk (spec["prefill_logits"]) computes the
# chunk's logits only while this flag is non-zero (the default). A chunked
# prefill session clears it with buddy_set_prefill_logits for the calls
# whose logits it does not use: all but the last (gen_session.py).
PREFILL_LOGITS_FLAG = "buddy_prefill_logits"
PREFILL_LOGITS_SET_FN = "buddy_set_prefill_logits"


def _prefill_logits_flag(ty):
    """The flag of the prefill lm_head kernels and its setter,
    buddy_set_prefill_logits(int32_t)."""
    llvm.GlobalOp(
        ty.i32,
        PREFILL_LOGITS_FLAG,
        ir.Attribute.parse("#llvm.linkage<internal>"),
        value=ir.IntegerAttr.get(ty.i32, 1),
    )
    fn = _Fn(PREFILL_LOGITS_SET_FN, [ty.i32], [])
    with ir.InsertionPoint(fn.entry):
        addr = llvm.AddressOfOp(_llvm_ptr(), PREFILL_LOGITS_FLAG).result
        llvm.StoreOp(fn.args[0], addr)
        func.ReturnOp([])


def _tiles(fn, ty, spec, a, xq, xs, ys, tw, tiles, threads, g0):
    """The parallel loop of the tiles of _matmul_fn (IME: of the pass from
    group g0 on)."""
    ns, kind = spec["ns"], spec["kind"]
    body, (slot,) = _parallel(fn, [fn.idx(threads)])
    with ir.InsertionPoint(body):
        ntiles, nthreads = fn.idx(tiles), fn.idx(threads)
        lo = _divui(_muli(slot, ntiles), nthreads)
        hi = _divui(_muli(_addi(slot, fn.idx(1)), ntiles), nthreads)
        tl = scf.ForOp(lo, hi, fn.idx(1))
        with ir.InsertionPoint(tl.body):
            t = tl.induction_variable
            col = _muli(t, fn.idx(tw))
            dsts = [memref.CastOp(ty.mat, y).result for y in ys]
            # multi: the output the tile belongs to, and its column there
            dst = dsts[-1]
            dcol = _subi(col, fn.idx(sum(ns[:-1])))
            for i in range(len(ns) - 2, -1, -1):
                inside = arith.CmpIOp(
                    arith.CmpIPredicate.ult,
                    t,
                    fn.idx(sum(ns[: i + 1]) // tw),
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
            if g0 is not None:
                args.append(fn.idx(g0))
            func.CallOp([], f"{spec['name']}__tile", args)
            scf.YieldOp([])
        scf.ReduceOp([], 0)


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


def _decode_heads_per_item(spec) -> int:
    """Heads per work item of _attn_decode_fn: the fewest heads of one KV
    group (a divisor of heads / kv_heads) for which there are no more work
    items than threads, so that no thread runs two items (DeepSeek R1, 12
    heads in 2 groups on 8 threads: 2 heads, 6 items). Without
    spec["threads"], one head per item."""
    h, kvh, m = spec["heads"], spec["kv_heads"], spec["m"]
    group = h // kvh
    threads = spec.get("threads")
    if not threads:
        return 1
    for g in range(1, group + 1):
        if group % g == 0 and (h // g) * m <= threads:
            return g
    return group


def _attn_decode_fn(ty, spec, block=16):
    """Causal attention of m query rows (positions start .. start + m - 1)
    against the updated KV cache, G heads of one KV group and one row per
    work item (G = _decode_heads_per_item): keys 0 .. start + i only,
    online softmax over blocks of `block` keys (one rescale per block,
    vectorized exp), then a per-key tail. The G heads share the loads of
    each key and value row; each head computes exactly what it computes
    alone. Used for decode (m == 1)."""
    m, h, kvh, d = spec["m"], spec["heads"], spec["kv_heads"], spec["dim"]
    hd, B = d // 2, block
    G = _decode_heads_per_item(spec)
    fn, a, outs = _attn_fn(ty, spec)
    vd, vb, vh = _vec(d, ty.f32), _vec(B, ty.f32), _vec(hd, ty.f32)
    f0 = fn.const(ty.f32, 0.0)
    c0 = fn.idx(0)
    with ir.InsertionPoint(fn.entry):
        body, (hg, i) = _parallel(fn, [fn.idx(h // G), fn.idx(m)])
        with ir.InsertionPoint(body):
            # heads hg * G .. hg * G + G - 1 (one KV group: G divides it)
            kh = _divui(_muli(hg, fn.idx(G)), fn.idx(h // kvh))
            ca = vector.LoadOp(vh, a["cs"], [i, c0]).result
            sa = vector.LoadOp(vh, a["sn"], [i, c0]).result
            heads, qss = [], []
            for g in range(G):
                hh = _addi(_muli(hg, fn.idx(G)), fn.idx(g))
                q_lo = _muli(hh, fn.idx(d))
                qa = _read(vh, a["q"], [i, q_lo], f0)
                qb = _read(vh, a["q"], [i, _addi(q_lo, fn.idx(hd))], f0)
                rl = _fma(qa, ca, arith.NegFOp(_mulf(qb, sa)).result)
                rh = _fma(qb, ca, _mulf(qa, sa))
                qv = vector.InsertStridedSliceOp(
                    rl, fn.const(vd, 0.0), [0], [1]
                ).result
                qv = vector.InsertStridedSliceOp(rh, qv, [hd], [1]).result
                qss.append(_mulf(qv, fn.const(vd, spec["scale"])))
                heads.append((hh, q_lo))
            length = arith.MinUIOp(
                _addi(_addi(a["start"], i), fn.idx(1)), fn.idx(spec["ctx"])
            ).result
            nblk = _divui(length, fn.idx(B))
            ninf = fn.const(ty.f32, float("-inf"))

            def key(j):
                return _read(vd, a["kc"], [c0, kh, j, c0], f0)

            def value(j):
                return _read(vd, a["vc"], [c0, kh, j, c0], f0)

            def score(qs, kv):
                return _reduce(ty.f32, "add", _mulf(qs, kv), reassoc=True)

            # blocks of B keys; per head (max, sum, acc)
            init = [ninf, f0, fn.const(vd, 0.0)] * G
            bl = scf.ForOp(c0, nblk, fn.idx(1), init)
            with ir.InsertionPoint(bl.body):
                state = list(bl.inner_iter_args)
                j0 = _muli(bl.induction_variable, fn.idx(B))
                ss = [fn.const(vb, 0.0)] * G
                for u in range(B):
                    kv = key(_addi(j0, fn.idx(u)))
                    ss = [
                        vector.InsertOp(
                            score(qss[g], kv), ss[g], [], [u]
                        ).result
                        for g in range(G)
                    ]
                news, ps, accs = [], [], []
                for g in range(G):
                    mx, l, acc = state[3 * g : 3 * g + 3]
                    mn = arith.MaximumFOp(
                        mx, _reduce(ty.f32, "maximumf", ss[g])
                    ).result
                    alpha = math.ExpOp(_subf(mx, mn)).result
                    p = _exp(fn, ty, _subf(ss[g], _bcast(vb, mn)), B)
                    accs.append(_mulf(acc, _bcast(vd, alpha)))
                    ps.append(p)
                    news.append((mn, alpha, l))
                for u in range(B):
                    vv = value(_addi(j0, fn.idx(u)))
                    for g in range(G):
                        pu = vector.ExtractOp(ps[g], [], [u]).result
                        accs[g] = _fma(vv, _bcast(vd, pu), accs[g])
                out = []
                for g in range(G):
                    mn, alpha, l = news[g]
                    psum = _reduce(ty.f32, "add", ps[g], reassoc=True)
                    out += [mn, _addf(_mulf(l, alpha), psum), accs[g]]
                scf.YieldOp(out)
            # the remaining keys, one at a time
            tl = scf.ForOp(
                _muli(nblk, fn.idx(B)), length, fn.idx(1), list(bl.results)
            )
            with ir.InsertionPoint(tl.body):
                state = list(tl.inner_iter_args)
                j = tl.induction_variable
                kv, vv = key(j), value(j)
                out = []
                for g in range(G):
                    mx, l, acc = state[3 * g : 3 * g + 3]
                    s = score(qss[g], kv)
                    mn = arith.MaximumFOp(mx, s).result
                    alpha = math.ExpOp(_subf(mx, mn)).result
                    beta = math.ExpOp(_subf(s, mn)).result
                    acc = _fma(
                        vv, _bcast(vd, beta), _mulf(acc, _bcast(vd, alpha))
                    )
                    out += [mn, _addf(_mulf(l, alpha), beta), acc]
                scf.YieldOp(out)
            results = list(tl.results)
            for g, (hh, q_lo) in enumerate(heads):
                mx, l, acc = results[3 * g : 3 * g + 3]
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


# ---------------------------------------------------------------------------
# Prefill tiles on the matrix engine ("prefill_ime")
# ---------------------------------------------------------------------------
#
# The A100 cores of the SpacemiT K3 have a matrix engine (IME). A prefill
# tile of IME_ROWS rows multiplies int8 activations by the IME layout of the
# int4 weights with smt.vmadot.hp (ime.intr.vmadot.hp of buddy-mlir's IME
# dialect, -lower-ime target=k3). These functions are built into a module of
# their own (gen_kernels(..., "ime")), which compile_pipeline.py compiles
# with the A100 options (pipeline "kernels_ime").

IME_STEP_FN = "k3_ime_hp_step"
IME_A_GROUP = 2048  # bytes of int8 activations per group (64 rows x 32)
IME_S_GROUP = 1024  # bytes of f16 activation scales per group


def _tile_width(spec) -> int:
    """Output columns per work item. IME tiles use 64: the 1536-column
    projections then make 24 items (3 per thread) instead of 12. GLU uses 32:
    the gate and up accumulators take 16 KiB instead of 64 KiB (all of L1),
    and the 280 items of k 1536 x n 8960 split evenly over 8 threads."""
    if spec.get("ime"):
        return 32 if spec["kind"] == "glu" else 64
    return NB


def _ime_kchunk(groups: int) -> int:
    """Groups per K chunk of an IME tile. The activations of a group and
    their scales (3 KiB for the 64 rows) are read again for every 8-column
    block: from L2 when they fit there (k 1536); else (k 8960: 840 KiB, while
    the L2 of 4 cores is 1 MiB and the weights stream through it too) K is
    chunked so that a chunk's activations stay in L1. Measured on k 8960
    (280 groups): 10 groups per chunk 1.80 ms, 7: 1.83, 14: 1.85, 20: 1.90,
    28 and more: 2.03 (a chunk reloads the accumulators)."""
    if groups * 3 <= 512:
        return groups
    return max(d for d in range(1, 11) if groups % d == 0)


# The activations of an IME call are read from the TCM of the core pair a
# tile runs on, through SpacemiT's /dev/tcm (runtime/spacemit/
# BuddySpacemitTcm.c): a core loads 1 KiB from it in ~9 ns whatever the
# other cores do, while cached loads take ~40 ns when 3 or 4 cores of a
# cluster load (k3_ime_hp_step on 8 cores: 61 instead of 102 ns per group).
# The K3 has 4 core pairs of 768 KiB; a call copies its activations and
# their scales into each, in passes over K small enough for them
# (_ime_pass_groups).
K3_TCM_REGIONS = 4
K3_TCM_REGION_BYTES = 768 * 1024
SPACEMIT_TCM_PAIR_FN = "buddy_spacemit_tcm_pair"
SPACEMIT_TCM_HERE_FN = "buddy_spacemit_tcm_here"


def _ime_pass_groups(groups: int) -> int:
    """Groups per pass of an IME call: the activations and scales of a pass
    (IME_A_GROUP + IME_S_GROUP bytes per group) fit in a TCM region, and a
    pass is whole K chunks (_ime_kchunk). k 1536: one pass (144 KiB); k 8960:
    two of 140 groups (420 KiB), the second one adding to the output of the
    first, so that the weights are still read once."""
    kc = _ime_kchunk(groups)
    for passes in range(1, groups + 1):
        gp = groups // passes
        if (
            groups % passes == 0
            and gp % kc == 0
            and gp * (IME_A_GROUP + IME_S_GROUP) <= K3_TCM_REGION_BYTES
        ):
            return gp
    raise ValueError(f"k3_w4: no IME pass for {groups} groups")


def _ime_stage(fn, ty, xq, xs, g0, gp):
    """Copies the activations and scales of groups g0 .. g0 + gp - 1 into
    the TCM region of every core pair: A at 0, S at gp * IME_A_GROUP, with
    the strides of xq / xs. Work item i copies half i % 2 of region i / 2
    (the pool runs item i on the i-th core, so a pair copies into its own
    TCM); nothing when there is no TCM (buddy_spacemit_tcm_pair returns 0)."""
    i64 = ty.i64
    ptr = _llvm_ptr()

    def c64(v):
        return fn.const(i64, v)

    def ptr_of(mem):
        return arith.IndexCastOp(
            i64, memref.ExtractAlignedPointerAsIndexOp(mem).result
        ).result

    xq_p, xs_p = ptr_of(xq), ptr_of(xs)
    nbytes = gp * (IME_A_GROUP + IME_S_GROUP)
    body, (item,) = _parallel(fn, [fn.idx(2 * K3_TCM_REGIONS)])
    with ir.InsertionPoint(body):
        item64 = arith.IndexCastOp(i64, item).result
        region = arith.DivUIOp(item64, c64(2)).result
        part = arith.RemUIOp(item64, c64(2)).result
        dst = func.CallOp(
            [i64],
            SPACEMIT_TCM_PAIR_FN,
            [region, c64(K3_TCM_REGIONS), c64(nbytes)],
        ).result
        have = arith.CmpIOp(arith.CmpIPredicate.ne, dst, c64(0)).result
        cond = scf.IfOp(have)
        with ir.InsertionPoint(cond.then_block):
            # (source, destination, bytes per group) of A and of S
            for src0, dst0, per in (
                (_addi(xq_p, c64(g0 * IME_A_GROUP)), dst, IME_A_GROUP),
                (
                    _addi(xs_p, c64(g0 * IME_S_GROUP)),
                    _addi(dst, c64(gp * IME_A_GROUP)),
                    IME_S_GROUP,
                ),
            ):
                half = gp * per // 2
                off = _muli(part, c64(half))
                src_p = llvm.IntToPtrOp(ptr, _addi(src0, off)).result
                dst_p = llvm.IntToPtrOp(ptr, _addi(dst0, off)).result
                # copy in 512-byte vectors
                vt = _vec(512, ty.i8)
                cl = scf.ForOp(fn.idx(0), fn.idx(half // 512), fn.idx(1))
                with ir.InsertionPoint(cl.body):
                    at = _muli(
                        arith.IndexCastOp(i64, cl.induction_variable).result,
                        c64(512),
                    )
                    v = llvm.LoadOp(vt, _gep(src_p, at), alignment=1).result
                    llvm.StoreOp(v, _gep(dst_p, at), alignment=1)
                    scf.YieldOp([])
            scf.YieldOp([])
        scf.ReduceOp([], 0)


def _llvm_ptr():
    return ir.Type.parse("!llvm.ptr")


def _gep(base, offset):
    """base + offset bytes."""
    return llvm.GEPOp(
        _llvm_ptr(),
        base,
        [offset],
        [-(2**31)],  # one dynamic index
        ir.IntegerType.get_signless(8),
        0,
    ).result


def _vmadot_hp(ty, acc, a, b, scale):
    """ime.intr.vmadot.hp: acc + (a . b^T) * scale, fp16, 8 x 8 per block."""
    return ir.Operation.create(
        "ime.intr.vmadot.hp",
        results=[_svec(4, ty.f16)],
        operands=[acc, a, b, scale],
        attributes={"group": ir.IntegerAttr.get(ty.i32, 0)},
    ).result


def _svec(n, t):
    return ir.VectorType.get([n], t, scalable=[True])


def _ime_step_fn(ty):
    """k3_ime_hp_step(b, ng, a, s, acc, first): one 8-column block of the IME
    layout times 32 rows (4 row blocks of 8) of the chunk, over ng groups
    from b on. A100 vector loads cost ~10 ns each whatever their size (up to
    LMUL 8), so a group takes three loads:
      B: the 144-byte block (VP load, e8 m2): 8 x 32 int4 and the 8 f16
         column scales (the mask-register operand of smt.vmadot.hp, v0 / v1);
      A: 4 row blocks x (k 0-15 | k 16-31), 1 KiB;
      S: the activation scales of the 4 row blocks, 512 bytes;
    then per row block:  c16 = (A0 . B0^T + A1 . B1^T) * ws  (vmadot.hp)
                         acc += c16 * xs                      (f16 -> f32)
    The 4 accumulators (4 x 64 f32, rows r * 8 + c) start at 0 when `first`
    is non-zero, else they are loaded from acc, and are stored back there, as
    one 1 KiB vector each way. a / s point at the 32 rows in group 0 (stride
    IME_A_GROUP / IME_S_GROUP bytes).

    compile_pipeline.py compiles it with -mcpu=spacemit-a100
    -misched-prera-direction=topdown: LLVM then issues the B load first and
    unpacks it while A loads (bottom-up scheduling sinks the loads next to
    their uses: 75 instead of 59 ns per group on one A100 core)."""
    fn = _Fn(IME_STEP_FN, [ty.i64] * 6, [], public=False)
    b, ng, a, s, o, first = fn.args
    v8, v16i8 = _svec(8, ty.i8), _svec(16, ty.i8)
    v4h, v4f, v16f = _svec(4, ty.f16), _svec(4, ty.f32), _svec(16, ty.f32)
    with ir.InsertionPoint(fn.entry):
        ptr = _llvm_ptr()
        bp, ap, sp, op = (llvm.IntToPtrOp(ptr, v).result for v in (b, a, s, o))
        zero_acc = fn.const(v16f, 0.0)
        is_first = arith.CmpIOp(
            arith.CmpIPredicate.ne, first, fn.const(ty.i64, 0)
        ).result
        init = scf.IfOp(is_first, [v16f], has_else=True)
        with ir.InsertionPoint(init.then_block):
            scf.YieldOp([zero_acc])
        with ir.InsertionPoint(init.else_block):
            scf.YieldOp([llvm.LoadOp(v16f, op, alignment=4).result])
        accs = [
            vector.ScalableExtractOp(v4f, init.result, 4 * i).result
            for i in range(4)
        ]
        ones = fn.const(_svec(16, ty.i1), 1)
        four = fn.const(v8, 4)
        zero_h = fn.const(v4h, 0.0)
        n = arith.IndexCastOp(ty.index, ng).result
        loop = scf.ForOp(fn.idx(0), n, fn.idx(1), accs)
        with ir.InsertionPoint(loop.body):
            gi = arith.IndexCastOp(ty.i64, loop.induction_variable).result
            bg = _gep(bp, _muli(gi, fn.const(ty.i64, IME_BLOCK)))
            ag = _gep(ap, _muli(gi, fn.const(ty.i64, IME_A_GROUP)))
            sg = _gep(sp, _muli(gi, fn.const(ty.i64, IME_S_GROUP)))
            bb = ir.Operation.create(
                "llvm.intr.vp.load",
                results=[v16i8],
                operands=[bg, ones, fn.const(ty.i32, IME_BLOCK)],
            ).result
            raw = vector.ScalableExtractOp(v8, bb, 0).result
            wscale = vector.BitCastOp(
                v4h, vector.ScalableExtractOp(v8, bb, 8).result
            ).result
            lo = arith.ShRSIOp(arith.ShLIOp(raw, four).result, four).result
            hi = arith.ShRSIOp(raw, four).result
            aa = llvm.LoadOp(_svec(64, ty.i8), ag, alignment=1).result
            ss = llvm.LoadOp(_svec(16, ty.f16), sg, alignment=2).result
            avs = [
                vector.ScalableExtractOp(v8, aa, 8 * j).result for j in range(8)
            ]
            svs = [
                vector.ScalableExtractOp(v4h, ss, 4 * i).result
                for i in range(4)
            ]
            ps = [
                _vmadot_hp(ty, zero_h, avs[2 * i], lo, wscale) for i in range(4)
            ]
            outs = []
            for i in range(4):
                q = _vmadot_hp(ty, ps[i], avs[2 * i + 1], hi, wscale)
                w = arith.ExtFOp(v4f, q).result
                xs = arith.ExtFOp(v4f, svs[i]).result
                outs.append(_fma(w, xs, loop.inner_iter_args[i]))
            scf.YieldOp(outs)
        res = zero_acc
        for i in range(4):
            res = vector.ScalableInsertOp(loop.results[i], res, 4 * i).result
        llvm.StoreOp(res, op, alignment=4)
        func.ReturnOp([])


def _ime_tile_fn(ty, spec):
    """One prefill tile on the matrix engine: _tile_width(spec) columns (GLU:
    of gate and of up) for the IME_ROWS rows. Loops: row half (32 rows) > K
    chunk > 8-column block; the f32 accumulators of every column block live
    in a scratch buffer between the K chunks. One call covers the pass of
    groups g0 .. g0 + _ime_pass_groups(groups) - 1, its activations read
    from the TCM of the core pair (see _ime_stage)."""
    m, k, ns, kind = spec["m"], spec["k"], spec["ns"], spec["kind"]
    assert m == IME_ROWS, spec
    groups = k // G
    gp = _ime_pass_groups(groups)
    glu = kind == "glu"
    # several passes: a pass resumes the sums of the previous one from the
    # output, which holds them (no GLU product, no bias)
    assert gp == groups or not (glu or spec["bias"]), spec
    blocks = ns[0] // 8 if glu else sum(ns) // 8  # 8-column blocks of a part
    nbw = _tile_width(spec) // 8  # blocks per output tile
    nb_tile = 2 * nbw if glu else nbw  # blocks per call (GLU: gate and up)
    kc = _ime_kchunk(groups)
    names = ["w", "xq", "xs", "wt", "out", "ocol"]
    if glu:
        names.append("out2")
    if spec["bias"]:
        names += ["bias", "bcol"]
    names.append("g0")
    fn = _Fn(f"{spec['name']}__tile", _tile_types(ty, spec), [], public=False)
    a = dict(zip(names, fn.args))
    acc_t = _memref([nb_tile * 512], ty.f32)
    i64 = ty.i64

    def c64(v):
        return fn.const(i64, v)

    def as_i64(v):
        return arith.IndexCastOp(i64, v).result

    with ir.InsertionPoint(fn.entry):
        acc = memref.AllocaOp(acc_t, [], [], alignment=128).result
        meta = memref.ExtractStridedMetadataOp(a["w"])
        w_base = _addi(
            memref.ExtractAlignedPointerAsIndexOp(a["w"]).result, meta.offset
        )
        xq_p, xs_p, acc_p = (
            as_i64(memref.ExtractAlignedPointerAsIndexOp(v).result)
            for v in (a["xq"], a["xs"], acc)
        )
        g0 = as_i64(a["g0"])
        # the pass's activations and scales: in the TCM of this core pair
        # (_ime_stage), else in xq / xs
        tcm = func.CallOp(
            [i64],
            SPACEMIT_TCM_HERE_FN,
            [c64(K3_TCM_REGIONS), c64(gp * (IME_A_GROUP + IME_S_GROUP))],
        ).result
        in_tcm = arith.CmpIOp(arith.CmpIPredicate.ne, tcm, c64(0)).result
        xq_p = arith.SelectOp(
            in_tcm, tcm, _addi(xq_p, _muli(g0, c64(IME_A_GROUP)))
        ).result
        xs_p = arith.SelectOp(
            in_tcm,
            _addi(tcm, c64(gp * IME_A_GROUP)),
            _addi(xs_p, _muli(g0, c64(IME_S_GROUP))),
        ).result
        first_block = _muli(a["wt"], fn.idx(nbw))
        half = nb_tile * 256
        v8f = _vec(8, ty.f32)

        def acc_at(nb, row):
            """Row `row` of column block nb in acc: f32
            ((row / 32) * nb_tile + nb) * 256 + (row % 32) * 8."""
            return _addi(
                _addi(
                    _muli(nb, fn.idx(256)),
                    _muli(_divui(row, fn.idx(32)), fn.idx(half)),
                ),
                _muli(arith.RemUIOp(row, fn.idx(32)).result, fn.idx(8)),
            )

        def rows_of_blocks(body):
            """body(column block, its first output column, row) for every
            row of every column block of the tile."""
            nl = scf.ForOp(fn.idx(0), fn.idx(nbw), fn.idx(1))
            with ir.InsertionPoint(nl.body):
                nb = nl.induction_variable
                col = _addi(a["ocol"], _muli(nb, fn.idx(8)))
                rl = scf.ForOp(fn.idx(0), fn.idx(IME_ROWS), fn.idx(1))
                with ir.InsertionPoint(rl.body):
                    body(nb, col, rl.induction_variable)
                    scf.YieldOp([])
                scf.YieldOp([])

        if gp < groups:
            later = arith.CmpIOp(
                arith.CmpIPredicate.ne, a["g0"], fn.idx(0)
            ).result
            # a later pass: the sums so far, from the output (exact f32, so
            # the result is that of one pass)
            resume = scf.IfOp(later)
            with ir.InsertionPoint(resume.then_block):

                def load_sums(nb, col, row):
                    val = vector.LoadOp(v8f, a["out"], [row, col]).result
                    vector.StoreOp(val, acc, [acc_at(nb, row)])

                rows_of_blocks(load_sums)
                scf.YieldOp([])
        # the accumulators of column block j (0 .. nb_tile), row half h: the
        # KiB h * nb_tile + j of acc
        for h in (0, 1):
            for part in (0, 1) if glu else (0,):
                kl = scf.ForOp(fn.idx(0), fn.idx(gp // kc), fn.idx(1))
                with ir.InsertionPoint(kl.body):
                    kci = as_i64(kl.induction_variable)
                    # zero sums: the first K chunk of the first pass
                    first = arith.CmpIOp(
                        arith.CmpIPredicate.eq,
                        kl.induction_variable,
                        fn.idx(0),
                    ).result
                    if gp < groups:
                        first = arith.AndIOp(
                            first,
                            arith.XOrIOp(later, fn.const(ty.i1, 1)).result,
                        ).result
                    first = arith.ExtUIOp(i64, first).result
                    xa = _addi(
                        _addi(xq_p, _muli(kci, c64(kc * IME_A_GROUP))),
                        c64(h * IME_A_GROUP // 2),
                    )
                    xs = _addi(
                        _addi(xs_p, _muli(kci, c64(kc * IME_S_GROUP))),
                        c64(h * IME_S_GROUP // 2),
                    )
                    kbytes = _muli(
                        _addi(
                            a["g0"], _muli(kl.induction_variable, fn.idx(kc))
                        ),
                        fn.idx(IME_BLOCK),
                    )
                    bl = scf.ForOp(fn.idx(0), fn.idx(nbw), fn.idx(1))
                    with ir.InsertionPoint(bl.body):
                        j = bl.induction_variable
                        # GLU: the up blocks follow all the gate blocks
                        blk = _addi(
                            _addi(first_block, j), fn.idx(part * blocks)
                        )
                        wb = _addi(
                            _addi(
                                w_base, _muli(blk, fn.idx(groups * IME_BLOCK))
                            ),
                            kbytes,
                        )
                        accj = _addi(
                            _addi(acc_p, _muli(as_i64(j), c64(1024))),
                            c64(1024 * (nb_tile * h + nbw * part)),
                        )
                        func.CallOp(
                            [],
                            IME_STEP_FN,
                            [as_i64(wb), c64(kc), xa, xs, accj, first],
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])

        # the results (see acc_at)
        if glu:
            nl = scf.ForOp(fn.idx(0), fn.idx(nbw), fn.idx(1))
            with ir.InsertionPoint(nl.body):
                nb = nl.induction_variable
                col = _addi(a["ocol"], _muli(nb, fn.idx(8)))
                base = _muli(nb, fn.idx(256))
                # SiLU(gate) * up over a half block (32 rows x 8) at once
                v256 = _vec(256, ty.f32)
                for h in (0, 1):
                    gate_at = _addi(base, fn.idx(h * half))
                    up_at = _addi(gate_at, fn.idx(nbw * 256))
                    gate = vector.LoadOp(v256, acc, [gate_at]).result
                    up = vector.LoadOp(v256, acc, [up_at]).result
                    val = _silu_mul(fn, ty, gate, up, 256)
                    for r in range(32):
                        row = vector.ExtractStridedSliceOp(
                            v8f, val, [r * 8], [8], [1]
                        ).result
                        vector.StoreOp(row, a["out"], [fn.idx(32 * h + r), col])
                scf.YieldOp([])
        else:

            def store(nb, col, row):
                val = vector.LoadOp(v8f, acc, [acc_at(nb, row)]).result
                if spec["bias"]:
                    bias = _read(
                        v8f,
                        a["bias"],
                        [_addi(a["bcol"], _muli(nb, fn.idx(8)))],
                        fn.const(ty.f32, 0.0),
                    )
                    val = _addf(val, bias)
                vector.StoreOp(val, a["out"], [row, col])

            rows_of_blocks(store)
        func.ReturnOp([])


# ---------------------------------------------------------------------------
# Prefill attention on the matrix engine ("prefill_ime")
# ---------------------------------------------------------------------------
#
# smt.vfwmadot vd, vs1, vs2 (ime.intr.vfmadot; fp16 x fp16 -> f32, VLEN 1024):
#   C[8r + c] += sum_k A[8r + k] * B[8c + k],  r, c, k = 0..7
# with A and B one register of 64 f16 and C 64 f32 (a register pair).
#
# The f32 KV caches stay the source of truth (decode reads them). A prefill
# call packs the keys 0 .. len - 1 of each KV head into fp16 operands:
#   K: kp[dc][kb][8k + dd] = K[8kb + k][8dc + dd]   (B of S = Q K^T)
#   V: vp[kb][8i + k]      = V[8kb + k][i]          (B of O = P V)
# (dc: chunk of 8 dimensions, kb: block of 8 keys, i: dimension). A work item
# is 8 query rows of a head: Q in fp16 A operands, S = Q K^T for all keys in
# f32, the causal softmax, P in fp16 (exactly the A layout of P V), and
# O = P V / l.

# The IME attention needs head_dim % ATTN_IME_DIMS == 0 (the columns of one
# P V call) and ctx % ATTN_IME_KEYS == 0 (the keys of one Q K^T call);
# k3_w4_rewrite falls back to the RVV kernel otherwise.
ATTN_IME_DIMS = 64
ATTN_IME_KEYS = 64
ATTN_QK_FN = "k3_attn_ime_qk"
ATTN_PV_FN = "k3_attn_ime_pv"


def _const_vec(t, values):
    """A constant vector of the given values."""
    elt = t.element_type
    if isinstance(elt, ir.FloatType):
        dtype = numpy.float16 if elt.width == 16 else numpy.float32
    else:
        dtype = {32: numpy.int32, 64: numpy.int64}[elt.width]
    attr = ir.DenseElementsAttr.get(numpy.array(values, dtype=dtype), type=t)
    return arith.ConstantOp(t, attr).result


def _ptr_i64(ty, mem):
    """The aligned pointer of `mem` as an i64."""
    return arith.IndexCastOp(
        ty.i64, memref.ExtractAlignedPointerAsIndexOp(mem).result
    ).result


def _attn_mma_fns(ty, emulate=False):
    """The two matrix loops of the IME attention, functions of raw pointers
    and sizes (i64), with Q and P as A operands (8 rows) and C stored as
    8 blocks of 64 f32 at `out`:
      k3_attn_ime_qk(q, k, out, chunks, kstride): S of 8 key blocks,
        out[64j + 8r + c] = sum_dc sum_dd q[64dc + 8r + dd] k_dc[64j + 8c + dd]
        with k_dc = k + dc * kstride bytes, over `chunks` chunks dc;
      k3_attn_ime_pv(p, v, out, blocks, vstride): O of 64 dimensions,
        out[64j + 8r + c] = sum_kb sum_k p[64kb + 8r + k] v_kb[64j + 8c + k]
        with v_kb = v + kb * vstride bytes, over `blocks` key blocks kb.
    Both are the same loop: per step one A load (64 f16), one B load (8
    blocks of 64 f16) and 8 smt.vfwmadot. With `emulate`, the same with
    scalar f32 arithmetic, without the IME (tests on a host)."""
    for name in (ATTN_QK_FN, ATTN_PV_FN):
        _attn_mma_fn(ty, name, emulate)


def _attn_mma_fn(ty, name, emulate):
    """One of the loops of _attn_mma_fns."""
    i64, ptr = ty.i64, _llvm_ptr()
    v4h, v4f, v32h = _svec(4, ty.f16), _svec(4, ty.f32), _svec(32, ty.f16)
    fn = _Fn(name, [i64] * 5, [], public=False)
    ap, bp, outp, steps, bstride = fn.args
    with ir.InsertionPoint(fn.entry):
        a = llvm.IntToPtrOp(ptr, ap).result
        b = llvm.IntToPtrOp(ptr, bp).result
        out = llvm.IntToPtrOp(ptr, outp).result
        n = arith.IndexCastOp(ty.index, steps).result
        c0, c1, c8 = fn.idx(0), fn.idx(1), fn.idx(8)
        if emulate:
            f0 = fn.const(ty.f32, 0.0)

            def f16_at(base, index):
                p = _gep(base, _muli(index, fn.const(i64, 2)))
                v = llvm.LoadOp(ty.f16, p, alignment=2).result
                return arith.ExtFOp(ty.f32, v).result

            def i64_of(v):
                return arith.IndexCastOp(i64, v).result

            jl = scf.ForOp(c0, c8, c1)
            with ir.InsertionPoint(jl.body):
                j = i64_of(jl.induction_variable)
                rl = scf.ForOp(c0, c8, c1)
                with ir.InsertionPoint(rl.body):
                    r = i64_of(rl.induction_variable)
                    cl = scf.ForOp(c0, c8, c1)
                    with ir.InsertionPoint(cl.body):
                        c = i64_of(cl.induction_variable)
                        sl = scf.ForOp(c0, n, c1, [f0])
                        with ir.InsertionPoint(sl.body):
                            st = i64_of(sl.induction_variable)
                            bs = _gep(b, _muli(st, bstride))
                            kl = scf.ForOp(c0, c8, c1, [sl.inner_iter_args[0]])
                            with ir.InsertionPoint(kl.body):
                                k = i64_of(kl.induction_variable)
                                ai = _addi(
                                    _muli(st, fn.const(i64, 64)),
                                    _addi(_muli(r, fn.const(i64, 8)), k),
                                )
                                bi = _addi(
                                    _muli(j, fn.const(i64, 64)),
                                    _addi(_muli(c, fn.const(i64, 8)), k),
                                )
                                acc = _fma(
                                    f16_at(a, ai),
                                    f16_at(bs, bi),
                                    kl.inner_iter_args[0],
                                )
                                scf.YieldOp([acc])
                            scf.YieldOp([kl.results[0]])
                        at = _addi(
                            _muli(j, fn.const(i64, 64)),
                            _addi(_muli(r, fn.const(i64, 8)), c),
                        )
                        llvm.StoreOp(
                            sl.results[0],
                            _gep(out, _muli(at, fn.const(i64, 4))),
                            alignment=4,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
        else:
            zero = fn.const(v4f, 0.0)
            sl = scf.ForOp(c0, n, c1, [zero] * 8)
            with ir.InsertionPoint(sl.body):
                st = arith.IndexCastOp(i64, sl.induction_variable).result
                av = llvm.LoadOp(
                    v4h, _gep(a, _muli(st, fn.const(i64, 128))), alignment=2
                ).result
                bv = llvm.LoadOp(
                    v32h, _gep(b, _muli(st, bstride)), alignment=2
                ).result
                accs = []
                for j, acc in enumerate(sl.inner_iter_args):
                    bj = vector.ScalableExtractOp(v4h, bv, 4 * j).result
                    accs.append(
                        ir.Operation.create(
                            "ime.intr.vfmadot",
                            results=[v4f],
                            operands=[acc, av, bj],
                        ).result
                    )
                scf.YieldOp(accs)
            for j, acc in enumerate(sl.results):
                llvm.StoreOp(
                    acc, _gep(out, fn.const(i64, 256 * j)), alignment=4
                )
        func.ReturnOp([])


def _attn_prefill_ime_fn(ty, spec):
    """Prefill attention on the matrix engine (spec["ime"]): RoPE and the KV
    cache update as in _attn_fn, the fp16 packing of the keys 0 .. len - 1
    in parallel over (KV head, range of key blocks), then in parallel over
    (head, 8 query rows) S = Q K^T, the causal softmax and O = P V with the
    matrix loops of _attn_mma_fns."""
    m, h, kvh, d, ctx = (
        spec["m"],
        spec["heads"],
        spec["kv_heads"],
        spec["dim"],
        spec["ctx"],
    )
    assert m % 8 == 0 and d % ATTN_IME_DIMS == 0, spec
    assert ctx % ATTN_IME_KEYS == 0, spec
    kbs = ctx // 8  # key blocks per KV head
    hd = h * d  # row stride of q and o
    half = d // 2  # RoPE halves
    fn, a, outs = _attn_fn(ty, spec)
    i1, i32, i64, f16, f32 = ty.i1, ty.i32, ty.i64, ty.f16, ty.f32
    c0, c1 = fn.idx(0), fn.idx(1)
    v64f, v64i = _vec(64, f32), _vec(64, i32)
    vhf, vhh = _vec(half, f32), _vec(half, f16)
    f0 = fn.const(f32, 0.0)

    def i64_of(v):
        return arith.IndexCastOp(i64, v).result

    with ir.InsertionPoint(fn.entry):
        length = _addi(a["start"], fn.idx(m))
        nkb = _divui(_addi(length, fn.idx(7)), fn.idx(8))
        # the packed operands; K as i64 (4 f16 each) for 64-bit scatters
        kpk = _alloc(_memref([kvh * kbs * d * 2], i64))
        vpk = _alloc(_memref([kvh * kbs * d * 8], f16))
        # K row: i64 element e (dimensions 4e .. 4e + 3) goes to chunk e / 2,
        # half e % 2 of a key; V row: dimension i to 8i
        k_idx = _const_vec(
            _vec(d // 4, i32),
            [(e // 2) * kbs * 16 + e % 2 for e in range(d // 4)],
        )
        v_idx = _const_vec(_vec(d, i32), [8 * i for i in range(d)])
        k_mask = fn.const(_vec(d // 4, i1), 1)
        v_mask = fn.const(_vec(d, i1), 1)
        zero_row = fn.const(_vec(d, f16), 0.0)
        len32 = arith.IndexCastOp(i32, length).result
        body, (kh, part) = _parallel(fn, [fn.idx(kvh), fn.idx(8)])
        with ir.InsertionPoint(body):
            per = _divui(_addi(nkb, fn.idx(7)), fn.idx(8))
            kb0 = arith.MinUIOp(_muli(part, per), nkb).result
            kb1 = arith.MinUIOp(_addi(kb0, per), nkb).result
            bl = scf.ForOp(kb0, kb1, c1)
            with ir.InsertionPoint(bl.body):
                kb = bl.induction_variable
                for k in range(8):
                    key = _addi(_muli(kb, fn.idx(8)), fn.idx(k))
                    inside = arith.CmpIOp(
                        arith.CmpIPredicate.ult,
                        arith.IndexCastOp(i32, key).result,
                        len32,
                    ).result
                    rows = []
                    for cache in ("kc", "vc"):
                        row = _read(
                            _vec(d, f32), a[cache], [c0, kh, key, c0], f0
                        )
                        row = arith.TruncFOp(_vec(d, f16), row).result
                        rows.append(
                            arith.SelectOp(inside, row, zero_row).result
                        )
                    k_at = _addi(
                        _muli(kh, fn.idx(kbs * d * 2)),
                        _addi(_muli(kb, fn.idx(16)), fn.idx(2 * k)),
                    )
                    k64 = vector.BitCastOp(_vec(d // 4, i64), rows[0]).result
                    vector.ScatterOp(None, kpk, [k_at], k_idx, k_mask, k64)
                    v_at = _addi(
                        _muli(kh, fn.idx(kbs * d * 8)),
                        _addi(_muli(kb, fn.idx(d * 8)), fn.idx(k)),
                    )
                    vector.ScatterOp(None, vpk, [v_at], v_idx, v_mask, rows[1])
                scf.YieldOp([])
            scf.ReduceOp([], 0)

        o_flat = memref.CollapseShapeOp(
            _memref([m * hd], f32), a["o"], [[0, 1]]
        ).result
        kp_base, vp_base = _ptr_i64(ty, kpk), _ptr_i64(ty, vpk)
        lanes = range(64)
        row_of = _const_vec(v64i, [i // 8 for i in lanes])
        col_of = _const_vec(v64i, [i % 8 for i in lanes])
        o_idx = _const_vec(v64i, [(i // 8) * hd + i % 8 for i in lanes])
        # Q, per RoPE half of a row: i64 element e goes to chunk e / 2 of the
        # half, half e % 2 of the row's 8 dimensions
        q_idx = _const_vec(
            _vec(d // 8, i32), [(e // 2) * 16 + e % 2 for e in range(d // 8)]
        )
        q_mask = fn.const(_vec(d // 8, i1), 1)
        o_mask = fn.const(_vec(64, i1), 1)
        neg_inf = fn.const(v64f, float("-inf"))
        zero64 = fn.const(v64f, 0.0)
        scale = fn.const(vhf, spec["scale"])

        def per_row(v, op):
            """op over lanes 8r .. 8r + 7 of v, in each of them."""
            for by in (4, 2, 1):
                moved = [i + by if i + by < 64 else i for i in lanes]
                v = op(v, vector.ShuffleOp(v, v, moved).result)
            return vector.ShuffleOp(v, v, [i & ~7 for i in lanes]).result

        def vmax(x, y):
            return arith.MaximumFOp(x, y).result

        body, (hh, rb) = _parallel(fn, [fn.idx(h), fn.idx(m // 8)])
        with ir.InsertionPoint(body):
            kh = _divui(hh, fn.idx(h // kvh))
            r0 = _muli(rb, fn.idx(8))
            p0 = _addi(a["start"], r0)
            qp = memref.AllocaOp(_memref([d * 2], i64), [], [], alignment=128)
            ob = memref.AllocaOp(_memref([512], f32), [], [], alignment=128)
            qp, ob = qp.result, ob.result
            s = _alloc(_memref([kbs * 64], f32))
            pb = _alloc(_memref([kbs * 64], f16))
            # Q: RoPE, scale, fp16 A operands qp[dc][8r + dd]
            col = _muli(hh, fn.idx(d))
            for r in range(8):
                row = _addi(r0, fn.idx(r))
                qa = _read(vhf, a["q"], [row, col], f0)
                qb = _read(vhf, a["q"], [row, _addi(col, fn.idx(half))], f0)
                cv = vector.LoadOp(vhf, a["cs"], [row, c0]).result
                sv = vector.LoadOp(vhf, a["sn"], [row, c0]).result
                lo = _fma(qa, cv, arith.NegFOp(_mulf(qb, sv)).result)
                hi = _fma(qb, cv, _mulf(qa, sv))
                for part, val in enumerate((lo, hi)):
                    val = arith.TruncFOp(vhh, _mulf(val, scale)).result
                    q64 = vector.BitCastOp(_vec(d // 8, i64), val).result
                    at = fn.idx(part * d + 2 * r)
                    vector.ScatterOp(None, qp, [at], q_idx, q_mask, q64)
            # S = Q K^T, 8 key blocks per call
            kh64 = i64_of(kh)
            kp_h = _addi(kp_base, _muli(kh64, fn.const(i64, kbs * d * 16)))
            vp_h = _addi(vp_base, _muli(kh64, fn.const(i64, kbs * d * 16)))
            qp_i, s_i = _ptr_i64(ty, qp), _ptr_i64(ty, s)
            groups = _divui(_addi(nkb, fn.idx(7)), fn.idx(8))
            gl = scf.ForOp(c0, groups, c1)
            with ir.InsertionPoint(gl.body):
                g = i64_of(gl.induction_variable)
                func.CallOp(
                    [],
                    ATTN_QK_FN,
                    [
                        qp_i,
                        _addi(kp_h, _muli(g, fn.const(i64, 8 * 128))),
                        _addi(s_i, _muli(g, fn.const(i64, 8 * 256))),
                        fn.const(i64, d // 8),
                        fn.const(i64, kbs * 128),
                    ],
                )
                scf.YieldOp([])
            # softmax of row r over the keys <= p0 + r
            pos = _addi(_bcast(v64i, arith.IndexCastOp(i32, p0).result), row_of)

            def visible(kb):
                first = arith.IndexCastOp(i32, _muli(kb, fn.idx(8))).result
                key = _addi(_bcast(v64i, first), col_of)
                return arith.CmpIOp(arith.CmpIPredicate.ule, key, pos).result

            ml = scf.ForOp(c0, nkb, c1, [neg_inf])
            with ir.InsertionPoint(ml.body):
                kb = ml.induction_variable
                sv = vector.LoadOp(v64f, s, [_muli(kb, fn.idx(64))]).result
                mx = ml.inner_iter_args[0]
                new = arith.SelectOp(visible(kb), vmax(mx, sv), mx).result
                scf.YieldOp([new])
            mx = per_row(ml.results[0], vmax)
            ll = scf.ForOp(c0, nkb, c1, [zero64])
            with ir.InsertionPoint(ll.body):
                kb = ll.induction_variable
                at = [_muli(kb, fn.idx(64))]
                sv = vector.LoadOp(v64f, s, at).result
                ev = _exp(fn, ty, _subf(sv, mx), 64)
                ev = arith.SelectOp(visible(kb), ev, zero64).result
                vector.StoreOp(arith.TruncFOp(_vec(64, f16), ev).result, pb, at)
                scf.YieldOp([_addf(ll.inner_iter_args[0], ev)])
            total = per_row(ll.results[0], _addf)
            inv = _divf(fn.const(v64f, 1.0), total)
            # lse of the 8 rows (not used by the model)
            firsts = [8 * i for i in range(8)]
            lse = _addf(
                math.LogOp(
                    vector.ShuffleOp(total, total, firsts).result
                ).result,
                vector.ShuffleOp(mx, mx, firsts).result,
            )
            vector.StoreOp(lse, a["lse"], [c0, hh, r0])
            # O = P V / l, 64 dimensions per call
            pb_i, ob_i = _ptr_i64(ty, pb), _ptr_i64(ty, ob)
            for dg in range(d // ATTN_IME_DIMS):
                func.CallOp(
                    [],
                    ATTN_PV_FN,
                    [
                        pb_i,
                        _addi(vp_h, fn.const(i64, dg * 1024)),
                        ob_i,
                        i64_of(nkb),
                        fn.const(i64, d * 16),
                    ],
                )
                for j in range(8):
                    ov = vector.LoadOp(v64f, ob, [fn.idx(64 * j)]).result
                    at = _addi(
                        _muli(r0, fn.idx(hd)),
                        _addi(col, fn.idx(ATTN_IME_DIMS * dg + 8 * j)),
                    )
                    vector.ScatterOp(
                        None, o_flat, [at], o_idx, o_mask, _mulf(ov, inv)
                    )
            memref.DeallocOp(s)
            memref.DeallocOp(pb)
            scf.ReduceOp([], 0)
        memref.DeallocOp(kpk)
        memref.DeallocOp(vpk)
        _attn_epilogue(fn, a, outs)


def _declare(ty, name, types, results=()):
    """A private declaration of a function defined in another module (or in
    the runtime)."""
    op = func.FuncOp(name, ir.FunctionType.get(types, list(results)))
    op.attributes["sym_visibility"] = ir.StringAttr.get("private")


def build_kernels(specs, part: str = "main", emulate_ime=False) -> ir.Module:
    """The module of the kernels `specs` (k3_w4_rewrite) describe, in the
    active context; verified. part "main": the kernels, with the RVV tiles
    and declarations of the IME functions; part "ime": the IME tiles, their
    step and the matrix loops of the IME attention (compiled for the A100
    cores). With `emulate_ime`, the main module defines the attention's
    matrix loops without the IME instead (tests on a host)."""
    ty = _Types()
    module = ir.Module.create()
    ime_attn = any(s["kind"] == "attn" and s.get("ime") for s in specs)
    ime_mm = any(s["kind"] != "attn" and s.get("ime") for s in specs)
    with ir.InsertionPoint(module.body):
        if ime_mm:
            # runtime/spacemit/BuddySpacemitTcm.c
            if part == "ime":
                _declare(ty, SPACEMIT_TCM_HERE_FN, [ty.i64] * 2, [ty.i64])
            else:
                _declare(ty, SPACEMIT_TCM_PAIR_FN, [ty.i64] * 3, [ty.i64])
        if part == "main" and any(s.get("prefill_logits") for s in specs):
            _prefill_logits_flag(ty)
        if part == "ime":
            _ime_step_fn(ty)
            if ime_attn:
                _attn_mma_fns(ty)
        elif ime_attn:
            if emulate_ime:
                _attn_mma_fns(ty, emulate=True)
            else:
                for name in (ATTN_QK_FN, ATTN_PV_FN):
                    _declare(ty, name, [ty.i64] * 5)
        for s in specs:
            ime = bool(s.get("ime"))
            if part == "ime":
                if ime and s["kind"] != "attn":
                    _ime_tile_fn(ty, s)
            elif s["kind"] == "attn":
                if s["m"] == 1:
                    _attn_decode_fn(ty, s)
                elif ime:
                    _attn_prefill_ime_fn(ty, s)
                else:
                    _attn_prefill_fn(ty, s)
            else:
                _matmul_fn(ty, s)
                if ime:
                    _declare(ty, f"{s['name']}__tile", _tile_types(ty, s))
                else:
                    _tile_fn(ty, s)
    if not module.operation.verify():
        raise RuntimeError("k3_w4: the generated kernels do not verify")
    return module


def gen_kernels(specs, part: str = "main", emulate_ime=False) -> str:
    """build_kernels() in a context of its own, as MLIR text."""
    with ir.Context(), ir.Location.unknown():
        return str(build_kernels(specs, part, emulate_ime))
