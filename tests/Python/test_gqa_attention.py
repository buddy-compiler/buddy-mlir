# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch.nn.functional as F
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph.transform import (
    gqa_attention_fusion,
)
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


def foo(query, k_cache, v_cache, index, mask, scale):
    k_updated = torch.index_put(k_cache, (index,), k_cache[0:1])
    k_unsqueeze = torch.unsqueeze(k_updated, 2)
    k_expanded = k_unsqueeze.expand(1, 2, 6, 1024, 128)
    k_clone = k_expanded.clone()
    k_view = k_clone.view(1, 12, 1024, 128)

    v_updated = torch.index_put(v_cache, (index,), v_cache[0:1])
    v_unsqueeze = torch.unsqueeze(v_updated, 2)
    v_expanded = v_unsqueeze.expand(1, 2, 6, 1024, 128)
    v_clone = v_expanded.clone()
    v_view = v_clone.view(1, 12, 1024, 128)

    attn_output = F.scaled_dot_product_attention(
        query, k_view, v_view, attn_mask=mask, scale=scale
    )

    return attn_output


in1 = torch.randn(1, 12, 1, 128)  # [Batch, Head, MaxSeq, Dim]
in2 = torch.randn(1, 2, 1024, 128)
in3 = torch.randn(1, 2, 1024, 128)  # [Batch, Head, 1, Dim]
in4 = torch.tensor([0], dtype=torch.int64)
in5 = torch.randn(1, 1, 1, 1024)
in6 = 1.0 / (128**0.5)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=False,
)

graphs = dynamo_compiler.importer(foo, in1, in2, in3, in4, in5, in6)
assert len(graphs) == 1
graph = graphs[0]

pattern_list = [gqa_attention_fusion]
graphs[0].fuse_ops(pattern_list)

graph.lower_to_top_level_ir()
print(graph._imported_module)

# gqa_attention_fused_op emits a fused, single-pass (online-softmax) decode
# attention kernel. It never materializes the [b, h, q, k] score tensor, so
# none of the old three-phase spelling (tosa.reduce_max / tosa.reduce_sum /
# tosa.log over a score tensor) survives -- see the CHECK-NOT below.

# CHECK-LABEL: func.func @forward

# Flash-Decoding: the iteration space is (head, k-split), run in parallel.
# 12 heads x 4 splits = 48 independent units, which is what fills the machine;
# heads alone would cap the parallel degree at 12. Lowers to OpenMP via
# -lower-affine + -convert-scf-to-openmp.
# CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (12, 4)
# h_kv = h / group_size, then this split's k range [s*256, s*256 + 256)
# CHECK:   arith.divsi
# CHECK:   arith.muli
# CHECK:   arith.addi

# Per (head, split): the blocked online-softmax loop over its own k range.
# Its 10 iter_args are the running (m, l) plus the head_dim/16 = 8 accumulator
# chunks, all kept in registers -- no memref for the accumulator.
# CHECK:   scf.for
# CHECK:     scf.for

# Gather this block's 16 scores into one vector<16xf32>: a dot product per k
# (vectorized over head_dim, then reduced), plus the mask, then inserted at
# the block-local lane.
# CHECK:       scf.for
# CHECK:         vector.transfer_read
# CHECK:         vector.transfer_read
# CHECK:         vector.fma
# CHECK:       vector.reduction <add>
# CHECK:       tensor.extract
# CHECK:       vector.insert

# The old implementation reduced over a materialized score tensor here.
# CHECK-NOT: tosa.reduce_max
# CHECK-NOT: tosa.reduce_sum

# Block max, then rebase on the *combined* running max (never -inf, so a fully
# masked block cannot produce exp(-inf - -inf) = NaN), then ONE vectorized exp
# covering all 16 of the block's weights -- not 16 scalar libm calls.
# CHECK:     vector.reduction <maximumf>
# CHECK:     arith.select
# CHECK:     math.exp %{{.*}} : f32
# CHECK:     math.exp %{{.*}} : vector<16xf32>
# CHECK:     vector.reduction <add>

# Weighted V accumulation for the block.
# CHECK:     scf.for
# CHECK:       vector.extract
# CHECK:       vector.broadcast
# CHECK:       vector.transfer_read
# CHECK:       vector.fma

# This split's partial (m, l, acc) -- not the final answer yet.
# CHECK:   memref.store
# CHECK:   vector.store

# Combine pass: fold the 4 partials back together with the same online-softmax
# recurrence, then normalize by l and write the output.
# CHECK: scf.for
# CHECK:   scf.for
# CHECK:     scf.for
# CHECK:       memref.load
# CHECK:       memref.load
# CHECK:       math.exp
# CHECK:       math.exp
# CHECK:       vector.load
# CHECK:       vector.fma
# CHECK:     math.log
# CHECK:     arith.divf
# CHECK:     vector.store

# CHECK: return %{{.*}} : tensor<1x12x1x128xf32>
