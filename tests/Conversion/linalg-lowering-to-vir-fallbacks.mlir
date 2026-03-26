// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse | FileCheck %s --check-prefix=CHECK-VEC-REDUCE

// -----
// CASE: linalg-reduce-to-vir-scalar-addf-f32.mlir
func.func @reduce_addf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_addf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_addf
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-scalar-maxnumf-f32.mlir
func.func @reduce_maxnumf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_maxnumf
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-addf-f32.mlir
func.func @reduce_2d_to_1d_addf(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_addf
// CHECK-NOT: linalg.reduce
// CHECK: scf.for %[[I:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[I]]]
// CHECK: vir.set_vl
// CHECK: vir.load %arg0[%[[I]]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_to_1d_addf
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-maxnumf-f32.mlir
func.func @reduce_2d_to_1d_maxnumf(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK: scf.for %[[I:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[I]]]
// CHECK: vir.set_vl
// CHECK: vir.load %arg0[%[[I]]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_to_1d_maxnumf
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-index-generic-to-scf-fallback.mlir
func.func @index_init(%out: memref<16xi32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      outs(%out : memref<16xi32>) {
    ^bb0(%o: i32):
      %idx = linalg.index 0 : index
      %v = arith.index_cast %idx : index to i32
      linalg.yield %v : i32
  }
  return
}

// CHECK-LABEL: func.func @index_init
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.index
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.store

// -----
// CASE: linalg-generic-cast-to-scf-fallback-extf-f16-f32.mlir
func.func @cast_extf(%arg0: memref<16xf16>, %arg1: memref<16xf32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xf16>)
      outs(%arg1 : memref<16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
  }
  return
}

// CHECK-LABEL: func.func @cast_extf
// CHECK-NOT: linalg.generic
// CHECK: vir.set_vl
// CHECK: vir.extf
// CHECK: vir.store

// -----
// CASE: linalg-generic-cast-to-scf-fallback-truncf-f32-f16.mlir
func.func @cast_truncf(%arg0: memref<16xf32>, %arg1: memref<16xf16>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xf32>)
      outs(%arg1 : memref<16xf16>) {
    ^bb0(%in: f32, %out: f16):
      %v = arith.truncf %in : f32 to f16
      linalg.yield %v : f16
  }
  return
}

// CHECK-LABEL: func.func @cast_truncf
// CHECK-NOT: linalg.generic
// CHECK: vir.set_vl
// CHECK: vir.truncf
// CHECK: vir.store

// -----
// CASE: non-extf/truncf casts still use scalar fallback.
func.func @cast_sitofp_fallback(%arg0: memref<16xi32>, %arg1: memref<16xf32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xi32>)
      outs(%arg1 : memref<16xf32>) {
    ^bb0(%in: i32, %out: f32):
      %v = arith.sitofp %in : i32 to f32
      linalg.yield %v : f32
  }
  return
}

// CHECK-LABEL: func.func @cast_sitofp_fallback
// CHECK-NOT: linalg.generic
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.sitofp
// CHECK: memref.store
