!type4d = memref<32x64x4x32xf32>

func.func @saxpy4d(%x: !type4d, %y: !type4d, %alpha : f32) -> !type4d {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  scf.forall (%i, %j) in (%c32, %c64) {
    scf.forall (%k, %l) in (%c4, %c32) {
      %4 = memref.load %x[%i, %j, %k, %l] : !type4d
      %5 = memref.load %y[%i, %j, %k, %l] : !type4d
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j, %k, %l] : !type4d
    }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
  }  { mapping = [#gpu.block<x>, #gpu.block<y>] }
  return %y : !type4d
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %gpuLaunch = transform.gpu.map_forall_to_blocks %funcop { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %gpuLaunch block_dims = [32, 4, 1] : (!transform.any_op) -> !transform.any_op
}
