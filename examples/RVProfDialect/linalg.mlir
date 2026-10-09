// Example: Input MLIR with multiple linalg operations
// This file shows the ORIGINAL code WITHOUT profiling markers
//
// running the instrumentation pass:
//   buddy-opt input.mlir -rvprof-instrument="rvprof-granularity=linalg"

func.func @matmul_add_mul(%A: memref<8x8xf32>, %B: memref<8x8xf32>,
                          %C: memref<8x8xf32>, %D: memref<8x8xf32>) {
  // After instrumentation, this will be wrapped with:
  //   rvprof.region_begin "linalg.matmul_0"
  //   linalg.matmul ...
  //   rvprof.region_end "linalg.matmul_0"
  linalg.matmul ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
                outs(%C : memref<8x8xf32>)

  // This tosa op should not be instrumented when granularity is linalg.
  %t0 = arith.constant dense<0.0> : tensor<8x8xf32>
  %t1 = arith.constant dense<1.0> : tensor<8x8xf32>
  %t2 = tosa.add %t0, %t1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>

  // After instrumentation:
  //   rvprof.region_begin "linalg.add_0"
  //   linalg.add ...
  //   rvprof.region_end "linalg.add_0"
  linalg.add ins(%C, %A : memref<8x8xf32>, memref<8x8xf32>)
             outs(%D : memref<8x8xf32>)

  // After instrumentation:
  //   rvprof.region_begin "linalg.mul_0"
  //   linalg.mul ...
  //   rvprof.region_end "linalg.mul_0"
  linalg.mul ins(%D, %B : memref<8x8xf32>, memref<8x8xf32>)
             outs(%D : memref<8x8xf32>)

  return
}

func.func @rvprof_test() attributes {llvm.emit_c_interface} {
  %A = memref.alloc() : memref<8x8xf32>
  %B = memref.alloc() : memref<8x8xf32>
  %C = memref.alloc() : memref<8x8xf32>
  %D = memref.alloc() : memref<8x8xf32>
  func.call @matmul_add_mul(%A, %B, %C, %D) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
  memref.dealloc %A : memref<8x8xf32>
  memref.dealloc %B : memref<8x8xf32>
  memref.dealloc %C : memref<8x8xf32>
  memref.dealloc %D : memref<8x8xf32>
  return
}
