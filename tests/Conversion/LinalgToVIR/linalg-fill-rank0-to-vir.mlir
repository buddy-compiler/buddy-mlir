// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

// A linalg.fill of a rank-0 memref (the accumulator of a reduction) is one
// scalar store; a fill of a vector of elements goes to VIR.

// CHECK-LABEL: func.func @fill_rank0
// CHECK-SAME: (%[[V:.*]]: f32, %[[M:.*]]: memref<f32>)
// CHECK: memref.store %[[V]], %[[M]][] : memref<f32>
// CHECK-NOT: linalg.fill
func.func @fill_rank0(%v: f32, %m: memref<f32>) {
  linalg.fill ins(%v : f32) outs(%m : memref<f32>)
  return
}

// CHECK-LABEL: func.func @fill_rank0_i1
// CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<i1>
// CHECK-NOT: linalg.fill
func.func @fill_rank0_i1(%m: memref<i1>) {
  %t = arith.constant true
  linalg.fill ins(%t : i1) outs(%m : memref<i1>)
  return
}

// CHECK-LABEL: func.func @fill_rank1
// CHECK: vir.set_vl
// CHECK: vir.broadcast
// CHECK-NOT: linalg.fill
func.func @fill_rank1(%v: i32, %m: memref<?xi32>) {
  linalg.fill ins(%v : i32) outs(%m : memref<?xi32>)
  return
}
