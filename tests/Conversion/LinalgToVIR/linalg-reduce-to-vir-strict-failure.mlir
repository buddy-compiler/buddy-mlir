// RUN: buddy-opt -verify-diagnostics %s -split-input-file -lower-linalg-to-vir

// Unsupported 2D->1D dim-0 reduction with non-static shapes.
func.func @reduce_2d_to_1d_dim0_dynamic_shape(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported 2D->1D dim-0 reduction with dynamic M and static N.
func.func @reduce_2d_to_1d_dim0_dynamic_m(%arg0: memref<?x8xf32>, %arg1: memref<8xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<?x8xf32>) outs(%arg1 : memref<8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported 2D->1D dim-0 reduction with static M and dynamic N.
func.func @reduce_2d_to_1d_dim0_dynamic_n(%arg0: memref<4x?xf32>, %arg1: memref<?xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<4x?xf32>) outs(%arg1 : memref<?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported element type.
func.func @reduce_1d_to_0d_i32(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %sum = arith.addi %in, %init : i32
      linalg.yield %sum : i32
    }
  return
}

// -----

// Unsupported combiner.
func.func @reduce_1d_to_0d_mulf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %mul = arith.mulf %in, %init : f32
      linalg.yield %mul : f32
    }
  return
}

// -----

// Unsupported higher-rank reduction shape.
func.func @reduce_3d_to_2d(%arg0: memref<2x4x8xf32>, %arg1: memref<2x4xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<2x4x8xf32>) outs(%arg1 : memref<2x4xf32>) dimensions = [2]
    (%in: f32, %init0: f32) {
      %sum = arith.addf %in, %init0 : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported 3D->2D dim-0 reduction with dynamic M.
func.func @reduce_3d_to_2d_dim0_dynamic_m(%arg0: memref<?x4x8xf32>, %arg1: memref<4x8xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<?x4x8xf32>) outs(%arg1 : memref<4x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported 3D->2D dim-0 reduction with dynamic N.
func.func @reduce_3d_to_2d_dim0_dynamic_n(%arg0: memref<2x?x8xf32>, %arg1: memref<?x8xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<2x?x8xf32>) outs(%arg1 : memref<?x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// Unsupported 3D->2D dim-0 reduction with dynamic K.
func.func @reduce_3d_to_2d_dim0_dynamic_k(%arg0: memref<2x4x?xf32>, %arg1: memref<4x?xf32>) {
  // expected-error @+1 {{unsupported linalg.reduce for -lower-linalg-to-vir}}
  linalg.reduce ins(%arg0 : memref<2x4x?xf32>) outs(%arg1 : memref<4x?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}
