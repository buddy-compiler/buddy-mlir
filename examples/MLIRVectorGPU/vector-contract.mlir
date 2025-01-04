#map0 = affine_map<(i, j, k) -> (i, j)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (i, k)>

module attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @vector_contract() kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %v0 = arith.constant dense<[[1., 2., 3., 4.], 
                                 [5., 6., 7., 8.], 
                                 [9., 10., 11., 12.]]> : vector<3x4xf32>
      %v1 = arith.constant dense<[[1., 2., 3.], 
                                 [4., 5., 6.], 
                                 [7., 8., 9.], 
                                 [10., 11., 12.]]> : vector<4x3xf32>
      %v2 = arith.constant dense<[[0., 0., 0.], 
                                 [0., 0., 0.], 
                                 [0., 0., 0.]]> : vector<3x3xf32>
      %v3 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]}
      %v0, %v1, %v2 : vector<3x4xf32>, vector<4x3xf32> into vector<3x3xf32>
      // vector.store %v3, %result[] : memref<vector<3x3xf32>>, vector<3x3xf32>
      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernel::@vector_contract blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args()
    func.return
  }
  func.func private @printMemrefF32(%ptr : memref<*xvector<3x3xf32>>)
}
