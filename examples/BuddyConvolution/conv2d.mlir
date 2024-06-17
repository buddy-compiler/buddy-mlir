#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                  outs (%arg2: memref<?x?xf32>)
    return
  }

  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
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

  func.func @main() {
    %c0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    
    %current_v1 = arith.constant 3 : index
    %current_v2 = arith.constant 8 : index
    %current_v0 = affine.apply #map0(%current_v2, %current_v1)
    
    %v0 = call @alloc_f32(%current_v0, %current_v0, %c1) : (index, index, f32) -> memref<?x?xf32>
    %v1 = call @alloc_f32(%current_v1, %current_v1, %c1) : (index, index, f32) -> memref<?x?xf32>
    %v2 = call @alloc_f32(%current_v2, %current_v2, %c0) : (index, index, f32) -> memref<?x?xf32>

    call @conv_2d(%v0, %v1, %v2) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    %print_v0 = memref.cast %v0 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_v0) : (memref<*xf32>) -> ()

    %print_v1 = memref.cast %v1 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_v1) : (memref<*xf32>) -> ()

    %print_v2 = memref.cast %v2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_v2) : (memref<*xf32>) -> ()

    memref.dealloc %v0 : memref<?x?xf32>
    memref.dealloc %v1 : memref<?x?xf32>
    memref.dealloc %v2 : memref<?x?xf32>
    return
  }
}