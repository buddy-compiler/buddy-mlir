module {
  func.func private @printMemrefF32(memref<*xf32>)
  // Allocate and fill the memref according to the given layout.
  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  // Convoluation implementation.
  func.func @conv_2d_nchw_fchw(%input: memref<?x?x?x?xf32>,
                               %kernel: memref<?x?x?x?xf32>,
                               %output: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nchw_fchw
      ins(%input, %kernel : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%output : memref<?x?x?x?xf32>)

    return
  }

  func.func @main() {
    // Intput and kernel value.
    %cst = arith.constant 1.000000e+00 : f32
    // Output value.
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Define layout.
    %input_n = arith.constant 1 : index
    %input_c = arith.constant 64 : index
    %input_h = arith.constant 58 : index
    %input_w = arith.constant 58 : index

    %kernel_n = arith.constant 64 : index
    %kernel_c = arith.constant 64 : index
    %kernel_h = arith.constant 3 : index
    %kernel_w = arith.constant 3 : index

    %output_n = arith.constant 1 : index
    %output_c = arith.constant 64 : index
    %output_h = arith.constant 56 : index
    %output_w = arith.constant 56 : index

    // Define input, kernel, and output memref.
    %input = call @alloc_f32(%input_n, %input_c, %input_h, %input_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %kernel = call @alloc_f32(%kernel_n, %kernel_c, %kernel_h, %kernel_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %output = call @alloc_f32(%output_n, %output_c, %output_h, %output_w, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    // Perform convolution
    call @conv_2d_nchw_fchw(%input, %kernel, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

    // Print the output
    %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %output : memref<?x?x?x?xf32>
    memref.dealloc %input : memref<?x?x?x?xf32>
    memref.dealloc %kernel : memref<?x?x?x?xf32>
    return
  }
}
