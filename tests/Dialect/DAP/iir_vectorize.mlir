// RUN: buddy-opt -vectorize-dap %s | FileCheck %s

func.func @buddy_iir_vectorize(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
  dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: func.func @buddy_iir_vectorize
// CHECK: vector.extract
// CHECK-SAME: %[[IDX:.*]]
// CHECK-NOT: : (vector<{{.*}}xf32>, i64) -> f32
