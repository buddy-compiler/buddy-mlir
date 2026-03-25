// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

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
// CHECK: %[[INIT:.+]] = memref.load %arg1[]
// CHECK: %[[FOR:.+]] = scf.for
// CHECK: %[[IN:.+]] = memref.load %arg0
// CHECK: %[[NEXT:.+]] = arith.addf %[[IN]], %{{.+}} : f32
// CHECK: scf.yield %[[NEXT]] : f32
// CHECK: memref.store %[[FOR]], %arg1[]

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
// CHECK: %[[INIT:.+]] = memref.load %arg1[]
// CHECK: %[[FOR:.+]] = scf.for
// CHECK: %[[IN:.+]] = memref.load %arg0
// CHECK: %[[NEXT:.+]] = arith.maxnumf %[[IN]], %{{.+}} : f32
// CHECK: scf.yield %[[NEXT]] : f32
// CHECK: memref.store %[[FOR]], %arg1[]

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
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.extf
// CHECK: memref.store

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
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.truncf
// CHECK: memref.store
