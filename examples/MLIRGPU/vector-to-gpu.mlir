#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0, 0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @matmul(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_f32 = arith.constant 0.000000e+00 : f32
      %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
      %B = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
      %C = vector.transfer_read %arg2[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
      %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
      vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>
      gpu.return
    }
  }

  func.func @main() {
    %A = memref.alloc() : memref<16x16xf16>
    %B = memref.alloc() : memref<16x16xf16>
    %C = memref.alloc() : memref<16x16xf32>
    %A_cast = memref.cast %A : memref<16x16xf16> to memref<*xf16>
    %B_cast = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    %C_cast = memref.cast %C : memref<16x16xf32> to memref<*xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst0 = arith.constant 0. : f32
    %cst1 = arith.constant 1. : f16
    %cst2 = arith.constant 2. : f16
    linalg.fill ins(%cst1 : f16) outs(%A : memref<16x16xf16>)
    linalg.fill ins(%cst2 : f16) outs(%B : memref<16x16xf16>)
    linalg.fill ins(%cst0 : f32) outs(%C : memref<16x16xf32>)
    gpu.host_register %A_cast : memref<*xf16>
    gpu.host_register %B_cast : memref<*xf16>
    gpu.host_register %C_cast : memref<*xf32>

    gpu.launch_func @kernels::@matmul blocks in (%c1, %c1, %c1) threads in (%c32,%c1,%c1) args(%A:memref<16x16xf16>, %B:memref<16x16xf16>, %C:memref<16x16xf32>)
    
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()
    func.return

  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) 
}

