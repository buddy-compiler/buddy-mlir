#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }
  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %2 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %3 = memref.dim %arg2, %c1 : memref<?x?xf32>
    scf.for %arg3 = %c0 to %2 step %c1 {
      scf.for %arg4 = %c0 to %3 step %c1 {
        scf.for %arg5 = %c0 to %0 step %c1 {
          scf.for %arg6 = %c0 to %1 step %c1 {
            %4 = affine.apply #map(%arg3, %arg5)
            %5 = affine.apply #map(%arg4, %arg6)
            %6 = memref.load %arg0[%4, %5] : memref<?x?xf32>
            %7 = memref.load %arg1[%arg5, %arg6] : memref<?x?xf32>
            %8 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
            %9 = arith.mulf %6, %7 : f32
            %10 = arith.addf %8, %9 : f32
            memref.store %10, %arg2[%arg3, %arg4] : memref<?x?xf32>
          }
        }
      }
    }
    return
  }
  func.func @main() {
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %0 = call @alloc_2d_filled_f32(%c3, %c3, %cst) : (index, index, f32) -> memref<?x?xf32>
    %1 = call @alloc_2d_filled_f32(%c10, %c10, %cst) : (index, index, f32) -> memref<?x?xf32>
    %2 = call @alloc_2d_filled_f32(%c8, %c8, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    call @conv_2d(%1, %0, %2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    %3 = memref.cast %2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    memref.dealloc %1 : memref<?x?xf32>
    memref.dealloc %0 : memref<?x?xf32>
    memref.dealloc %2 : memref<?x?xf32>
    return
  }
}

