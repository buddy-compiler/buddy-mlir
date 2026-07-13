# ===- pack_decode_weights.py --------------------------------------------------
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
# Repack matmul weights into N-panel layout for the decode (m == 1, GEMV) phase.
# See docs/DecodeWeightPacking.md.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import AddMMOp, MatmulOp, PlaceholderOp

# Which operand is the weight (B in A @ B): mm(lhs, rhs), addmm(bias, lhs, rhs).
# Both op types matter -- Qwen2's q/k/v projections carry a bias and so import as
# AddMMOp, everything else as MatmulOp -- and missing one is not a missed
# optimisation but a wrong model; see the all-or-nothing check at the end.
_WEIGHT_OPERAND = {MatmulOp: 1, AddMMOp: 2}


def pack_decode_matmul_weights(graph: Graph, vecsize: int = 32):
    """Rewrite every matmul weight in `graph` into panel-blocked layout:

        Bpacked[n / V][k][n % V] == B[k, n]        V = vecsize

    The declared shape stays [K, N]; only the byte order changes, so nothing
    downstream sees a difference. -matmul-vectorization-decode-packed is the
    kernel that reads it. See docs/DecodeWeightPacking.md for why.

    ONLY call this on the decode graph. Prefill's matmul kernel expects
    row-major B and would read packed bytes as if they were row-major, without
    complaint. What keeps it safe is that this *rebinds* entries in
    `graph._params_ref` -- a per-graph list -- rather than mutating the tensors,
    which prefill's graph still shares by reference. Do not "optimise" the clone
    away into an in-place permute.

    Args:
        graph (Graph): the decode Graph. Mutated in place.
        vecsize (int): panel width. Must equal the vector-size given to
            -matmul-vectorization-decode-packed, and must divide every
            weight's N.

    Returns:
        list[int]: indices into graph._params_ref that were repacked.
    """
    if graph._params_ref is None:
        raise RuntimeError(
            "pack_decode_matmul_weights: graph has no _params_ref; it can "
            "only run on a graph imported with its parameters."
        )

    # graph.params[i] is the placeholder for graph._params_ref[i], so a name
    # gives the index outright. (Matching on shape and dtype, the way
    # eliminate_transpose does, is ambiguous the moment two weights share a
    # shape -- and here many do.)
    name_to_index = {p.name: i for i, p in enumerate(graph.params)}

    packed_indices = []
    unpacked = 0
    for node in graph.body:
        operand = _WEIGHT_OPERAND.get(type(node))
        if operand is None or len(node.args) <= operand:
            continue

        weight_name = str(node.args[operand])
        index = name_to_index.get(weight_name)
        if index is None or not isinstance(
            graph.node_table.get(weight_name), PlaceholderOp
        ):
            unpacked += 1  # B is a computed value: nothing static to prepack.
            continue
        if index in packed_indices:
            # Packing it twice would apply the permutation twice.
            raise RuntimeError(
                f"pack_decode_matmul_weights: weight '{weight_name}' feeds "
                "more than one matmul; refusing to pack it twice."
            )

        weight = graph._params_ref[index]
        if weight.dim() != 2:
            unpacked += 1
            continue
        k, n = weight.shape
        if n % vecsize != 0:
            raise RuntimeError(
                f"pack_decode_matmul_weights: weight '{weight_name}' has "
                f"N={n}, not divisible by vecsize={vecsize}."
            )

        # Rebind, never mutate: prefill's graph holds the same tensor object.
        graph._params_ref[index] = (
            weight.detach()
            .reshape(k, n // vecsize, vecsize)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(k, n)
        )
        packed_indices.append(index)

    # The packed kernel selects per matmul, by shape; it cannot tell a packed
    # weight from a plain one. So packing *every* matmul weight is not a nicety
    # but the precondition for compiling this graph with it -- one weight left
    # behind is a wrong model, with no crash and no failing test.
    if unpacked:
        raise RuntimeError(
            f"pack_decode_matmul_weights: {unpacked} matmul(s) in this graph "
            f"take a weight that could not be packed ({len(packed_indices)} "
            "were packed). The packed decode kernel cannot distinguish them "
            "and would read those weights as if they were panel-packed. "
            "Either make them packable or do not pack this graph."
        )

    return packed_indices
