module {
  func private @print_memref_f32(memref<*xf32>)

  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func @main() {
    %current_input = constant 9 : index
    %current_output = constant 7 : index
    %cst0 = constant 0.000000e+00 : f32
    %cst1 = constant 1.000000e+00 : f32

    %input = call @alloc_2d_filled_f32(%current_input, %current_input, %cst1) : (index, index, f32) -> memref<?x?xf32>
    %kernel = img.prewitt : memref<?x?xf32>
    %output = call @alloc_2d_filled_f32(%current_output, %current_output, %cst0) : (index, index, f32) -> memref<?x?xf32>
    
    linalg.conv_2d ins (%input, %kernel: memref<?x?xf32>, memref<?x?xf32>)
                   outs (%output: memref<?x?xf32>)

    %print_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @print_memref_f32(%print_output) : (memref<*xf32>) -> ()

    return
  }
}