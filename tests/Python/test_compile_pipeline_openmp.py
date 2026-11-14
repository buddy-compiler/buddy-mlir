# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# The "subgraph" (prefill) and "standard" pipelines of compile_pipeline.py
# fork the OpenMP threads over a real loop, not over the unit batch dimension
# that -affine-parallelize makes the outermost scf.parallel dimension.

import os
import subprocess
import sys

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402

OPENMP = "-convert-scf-to-openmp"


def stage3(pipeline_type):
    """The buddy-opt arguments of the stage that converts to OpenMP."""
    stages = compile_pipeline.build_stages(
        pipeline_type, num_threads=4, llc_attrs="", variant="f32"
    )
    for _, args in stages:
        if any(a.startswith(OPENMP) for a in args):
            return args
    raise AssertionError(f"no {OPENMP} in the {pipeline_type} pipeline")


# -canonicalize runs right before the conversion to OpenMP.
for pipeline_type in ("subgraph", "standard"):
    args = stage3(pipeline_type)
    i = next(i for i, a in enumerate(args) if a.startswith(OPENMP))
    print(f"{pipeline_type}: {args[i - 1]} {args[i]}")
# CHECK: subgraph: -canonicalize -convert-scf-to-openmp=num-threads=4
# CHECK: standard: -canonicalize -convert-scf-to-openmp=num-threads=4

# An elementwise op with a unit batch dimension, through the "subgraph" stage
# up to the conversion to OpenMP: the loop nest the threads are forked over
# starts with the 64 rows, not with the single batch element.
MLIR = """
#id = affine_map<(b, r, c) -> (b, r, c)>
func.func @add(%a: tensor<1x64x32xf32>, %b: tensor<1x64x32xf32>)
    -> tensor<1x64x32xf32> {
  %e = tensor.empty() : tensor<1x64x32xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%a, %b : tensor<1x64x32xf32>, tensor<1x64x32xf32>)
      outs(%e : tensor<1x64x32xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<1x64x32xf32>
  return %r : tensor<1x64x32xf32>
}
"""
args = stage3("subgraph")
args = args[: next(i for i, a in enumerate(args) if a.startswith(OPENMP)) + 1]
out = subprocess.run(
    [sys.argv[1], *args], input=MLIR, capture_output=True, text=True, check=True
).stdout
print(out)
# Without -canonicalize, the threads are forked over `0 to 1` and the rows
# run in a nested, serialized omp.parallel.
# CHECK-LABEL: func.func @add
# CHECK-DAG: %[[COLS:.*]] = arith.constant 32 : index
# CHECK-DAG: %[[ROWS:.*]] = arith.constant 64 : index
# CHECK: omp.parallel
# CHECK-NEXT: omp.wsloop
# CHECK-NEXT: omp.loop_nest {{.*}} to (%[[ROWS]], %[[COLS]])
# CHECK-NOT: omp.parallel
