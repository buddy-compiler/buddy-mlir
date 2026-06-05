// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

func.func @scatter_implicit_base(%base: memref<16xf32>) {
  %vl = arith.constant 8 : index
  %idx = arith.constant 1 : i32
  vir.set_vl %vl : index {
    %value = vir.constant { value = 2.000000e+00 : f32 } : !vir.vec<?xf32>
    %index = vir.broadcast %idx : i32 -> !vir.vec<?xi32>
    %true = arith.constant true
    %mask = vir.broadcast %true : i1 -> !vir.vec<?xi1>
    vir.scatter %value, %base[][%index], %mask : !vir.vec<?xf32>, !vir.vec<?xi32>, !vir.vec<?xi1> -> memref<16xf32>
    vector.yield
  }
  return
}

func.func @scatter_explicit_base(%base: memref<4x16xf32>) {
  %vl = arith.constant 8 : index
  %row = arith.constant 1 : index
  %col = arith.constant 2 : index
  %idx = arith.constant 3 : i32
  vir.set_vl %vl : index {
    %value = vir.constant { value = 4.000000e+00 : f32 } : !vir.vec<?xf32>
    %index = vir.broadcast %idx : i32 -> !vir.vec<?xi32>
    %true = arith.constant true
    %mask = vir.broadcast %true : i1 -> !vir.vec<?xi1>
    vir.scatter %value, %base[%row, %col][%index], %mask : !vir.vec<?xf32>, !vir.vec<?xi32>, !vir.vec<?xi1> -> memref<4x16xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @scatter_implicit_base
// CHECK: vector.scatter
// CHECK-SAME: memref<16xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32>
// CHECK: memref.store
// CHECK-SAME: memref<16xf32>

// CHECK-LABEL: func.func @scatter_explicit_base
// CHECK: vector.scatter
// CHECK-SAME: memref<4x16xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32>
// CHECK: memref.store
// CHECK-SAME: memref<4x16xf32>
