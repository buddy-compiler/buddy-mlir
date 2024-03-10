transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
      // Generate gpu kernel.
  %func = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
  %gpu_launch = transform.gpu.map_forall_to_blocks %func generate_gpu_launch
      : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %gpu_launch block_dims = [4, 8, 4]
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}