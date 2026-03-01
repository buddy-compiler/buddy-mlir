// This file is not a test file - it is a kernel function for runtime testing.
// It is compiled and linked with runtime_matmul_boundary.c
//
// Test case: C[7x5] = A[7x10] * B[10x5] with non-aligned dimensions
// For int8: TILE_M=4, TILE_K=8, TILE_N=4

module {
  func.func @matmul_boundary(%A: memref<7x10xi8>, %B: memref<10x5xi8>,
                              %C: memref<7x5xi32>) {
    linalg.matmul ins(%A, %B : memref<7x10xi8>, memref<10x5xi8>)
                  outs(%C : memref<7x5xi32>)
    return
  }
}
