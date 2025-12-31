# RUN: %PYTHON %s 2>&1 | FileCheck %s

from tabnanny import verbose
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.graph.transform import (
    gqa_attention_fusion,
)

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(query, k_cache, v_cache, mask, scale):

    k_unsqueeze = torch.unsqueeze(k_cache, 2)
    k_slice1 = torch.narrow(k_unsqueeze, 3, 0, k_unsqueeze.size(3))
    k_slice2 = torch.narrow(k_slice1, 4, 0, k_slice1.size(4))
    k_expanded = k_slice2.expand(1, 2, 6, 1024, 128)
    k_clone = k_expanded.clone()
    k_view = k_clone.view(1, 12, 1024, 128)

    v_unsqueeze = torch.unsqueeze(v_cache, 2)
    v_slice1 = torch.narrow(v_unsqueeze, 3, 0, v_unsqueeze.size(3))
    v_slice2 = torch.narrow(v_slice1, 4, 0, v_slice1.size(4))
    v_expanded = v_slice2.expand(1, 2, 6, 1024, 128)
    v_clone = v_expanded.clone()
    v_view = v_clone.view(1, 12, 1024, 128)

    attn_output = F.scaled_dot_product_attention(
        query, k_view, v_view, attn_mask=mask, scale=scale
    )

    return attn_output


in1 = torch.randn(1, 12, 1, 128)  # [Batch, Head, MaxSeq, Dim]
in2 = torch.randn(1, 2, 1024, 128)
in3 = torch.randn(1, 2, 1024, 128)  # [Batch, Head, 1, Dim]
in4 = torch.randn(1, 1, 1, 1024)
in5 = 1.0 / (128**0.5)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=True,
)

graphs = dynamo_compiler.importer(foo, in1, in2, in3, in4, in5)
assert len(graphs) == 1
graph = graphs[0]

pattern_list = [gqa_attention_fusion]
graphs[0].fuse_ops(pattern_list)

graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK-LABEL: func.func @forward
# CHECK: affine.for %[[B:.*]] = 0 to {{.*}}
# CHECK:   affine.for %[[H:.*]] = 0 to {{.*}}
# CHECK:     arith.divsi
# CHECK: %{{.*}}:2 = affine.for %[[S:.*]] = 0 to {{.*}} iter_args(%[[M:.*]] = {{.*}}, %[[SUM:.*]] = {{.*}})
# CHECK: %[[VEC_RES:.*]] = affine.for %{{.*}} = 0 to {{.*}} step 16 iter_args(%{{.*}} = %{{.*}})
# CHECK:   vector.load
# CHECK:   vector.load
# CHECK:   %[[FMA:.*]] = vector.fma
# CHECK:   affine.yield %[[FMA]]
# CHECK: vector.reduction <add>, %[[VEC_RES]]
# CHECK: math.exp
# CHECK: arith.select
# CHECK: arith.select
# CHECK: affine.for %{{.*}} = 0 to {{.*}}
# CHECK:   %[[LOAD:.*]] = memref.load
# CHECK:   arith.divf %[[LOAD]]
