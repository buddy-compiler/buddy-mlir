// RUN: buddy-opt %s \
// RUN:     -matmul-vectorization="vector-type=scalable vector-size=4" \
// RUN: | FileCheck %s

module{
  // CHECK-LABEL: func.func @matmul_scalable
  func.func @matmul_scalable(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    // CHECK-NOT: linalg.matmul
    // CHECK: vector.vscale
    // CHECK: vector<[4]xf32>
    linalg.matmul
      ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
      outs(%c:memref<?x?xf32>)
    return
  }
}

