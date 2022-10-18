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

  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.matmul ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                  outs (%arg2: memref<?x?xf32>)
    return
  }

  func.func @main() {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index

    // Initial data of input and output.
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %input1 = call @alloc_2d_filled_f32(%c5, %c3, %cst) : (index, index, f32) -> memref<?x?xf32>
    %input2 = call @alloc_2d_filled_f32(%c3, %c2, %cst) : (index, index, f32) -> memref<?x?xf32>
    %output = call @alloc_2d_filled_f32(%c5, %c2, %cst_0) : (index, index, f32) -> memref<?x?xf32>

    call @matmul(%input1, %input2, %output) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Print output.
    %print_output = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %input1 : memref<?x?xf32>
    memref.dealloc %input2 : memref<?x?xf32>
    memref.dealloc %output : memref<?x?xf32>

    return
  }
}
