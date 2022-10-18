#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %argx2: index, %argx3: index, %arg2: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %argx2, %argx3) : memref<?x?x?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
    scf.for %arg5 = %c0 to %argx2 step %c1 {
      scf.for %arg6 = %c0 to %argx3 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
      }
    }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  func.func @conv_2d_nchw_fchw(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
	linalg.conv_2d_nchw_fchw 
    ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?x?xf32>)

    return
  }

  func.func @main() {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Image and Output value.
    %cst = arith.constant 5.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32

    %current_filter = arith.constant 16 : index
    %current_output = arith.constant 16 : index
    %current_image = arith.constant 16 : index

    // Filter.
    %filter = call @alloc_2d_filled_f32(%current_filter, %current_filter, %current_filter, %current_filter,  %cst) : (index, index,index, index, f32) -> memref<?x?x?x?xf32>
    // Image.
    %image = call @alloc_2d_filled_f32(%current_image, %current_image, %current_image, %current_image, %cst) : (index, index, index, index,f32) -> memref<?x?x?x?xf32>
    // Output.
    %output = call @alloc_2d_filled_f32(%current_output, %current_output, %current_output, %current_output,%cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    call @conv_2d_nchw_fchw(%image, %filter, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

    // Print output.
    %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %output : memref<?x?x?x?xf32>
    memref.dealloc %image : memref<?x?x?x?xf32>
    memref.dealloc %filter : memref<?x?x?x?xf32>
    return
  }
}
