#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
module {
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)

  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
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

  func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                  outs (%arg2: memref<?x?xf32>)
    return
  }

  func @main() {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Image and Output value.
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %current_filter = arith.constant 3 : index
    %current_output = arith.constant 8 : index
    %current_image = affine.apply #map0(%current_output, %current_filter)

    // Filter.
    %filter = call @alloc_2d_filled_f32(%current_filter, %current_filter, %cst) : (index, index, f32) -> memref<?x?xf32>
    // Image.
    %image = call @alloc_2d_filled_f32(%current_image, %current_image, %cst) : (index, index, f32) -> memref<?x?xf32>
    // Output.
    %output = call @alloc_2d_filled_f32(%current_output, %current_output, %cst_0) : (index, index, f32) -> memref<?x?xf32>

    call @conv_2d(%image, %filter, %output) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Print output.
    %print_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %image : memref<?x?xf32>
    memref.dealloc %filter : memref<?x?xf32>
    memref.dealloc %output : memref<?x?xf32>
    return
  }
}
