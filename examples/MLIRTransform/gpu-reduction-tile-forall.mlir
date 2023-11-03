func.func @reduction_tile_parallel_cyclic_dist(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
    by num_threads = [0, 5], tile_sizes = [0, 3], mapping = [#gpu.thread<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
