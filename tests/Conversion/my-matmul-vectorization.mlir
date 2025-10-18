// RUN: buddy-opt %s -my-matmul-vectorization="vector-size=32" | FileCheck %s

// Test for my matmul vectorization pass based on MyMatMulVectorization.cpp
// This test verifies that linalg.matmul operations are converted to BLIS-style hand-written implementations
// Specifically tests the conversion of linalg.matmul to optimized vectorized implementation with data packing

// CHECK-LABEL: func.func @matmul_test
// CHECK-NOT: linalg.matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.load
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK: vector.store
// CHECK: memref.dealloc
// CHECK: memref.dealloc

module {
  func.func private @printMemrefF32(memref<*xf32>)

  // Simple matrix multiplication function using linalg.matmul
  // This implementation should be converted to BLIS-style hand-written implementation by MyMatMulVectorizationPass
  func.func @matmul_test(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    linalg.matmul
      ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
      outs(%C: memref<?x?xf32>)
    return
  }

  func.func @main() {
    %c64 = arith.constant 64 : index
    
    %A = memref.alloc(%c64, %c64) : memref<?x?xf32>
    %B = memref.alloc(%c64, %c64) : memref<?x?xf32>
    %C = memref.alloc(%c64, %c64) : memref<?x?xf32>

    // Initialize matrices
    %cf0 = arith.constant 0.0 : f32
    %cf1 = arith.constant 1.0 : f32
    
    linalg.fill ins(%cf1 : f32) outs(%A : memref<?x?xf32>)
    linalg.fill ins(%cf1 : f32) outs(%B : memref<?x?xf32>)
    linalg.fill ins(%cf0 : f32) outs(%C : memref<?x?xf32>)

    call @matmul_test(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // Print output for verification
    %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
