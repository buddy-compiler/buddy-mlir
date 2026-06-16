// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

func.func @generic_reduction_scalar_loop_with_vir_touch(
    %input: memref<1x6x6x4xf32>, %filter: memref<3x3x4x8xf32>, %out: memref<1x4x4x8xf32>) {
  linalg.generic {
      indexing_maps = [
        affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh + kh, ow + kw, c)>,
        affine_map<(n, oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>,
        affine_map<(n, oh, ow, f, kh, kw, c) -> (n, oh, ow, f)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%input, %filter : memref<1x6x6x4xf32>, memref<3x3x4x8xf32>)
      outs(%out : memref<1x4x4x8xf32>) {
    ^bb0(%in: f32, %flt: f32, %acc: f32):
      %mul = arith.mulf %in, %flt : f32
      %sum = arith.addf %acc, %mul : f32
      linalg.yield %sum : f32
  }
  return
}

// CHECK-LABEL: func.func @generic_reduction_scalar_loop_with_vir_touch
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.yield
// CHECK: scf.for
// CHECK: affine.apply
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: memref.store
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.store

// -----

func.func @index_generic_scalar_loop_with_vir_touch(%out: memref<16xf32>) {
  linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      outs(%out : memref<16xf32>) {
    ^bb0(%o: f32):
      %idx = linalg.index 0 : index
      %idx_i32 = arith.index_cast %idx : index to i32
      %idx_f32 = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %idx_f32 : f32
  }
  return
}

// CHECK-LABEL: func.func @index_generic_scalar_loop_with_vir_touch
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.index
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: arith.sitofp
// CHECK: memref.store
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.store

// -----

func.func @fill_rng_2d(%out: memref<4x8xf32>) {
  %min = arith.constant 0.0 : f64
  %max = arith.constant 1.0 : f64
  %seed = arith.constant 7 : i32
  linalg.fill_rng_2d
      ins(%min, %max, %seed : f64, f64, i32)
      outs(%out : memref<4x8xf32>)
  return
}

// CHECK-LABEL: func.func @fill_rng_2d
// CHECK-NOT: linalg.fill_rng_2d
// CHECK: arith.constant 5.000000e-01 : f32
// CHECK: scf.for
// CHECK: memref.store
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.store

// -----

func.func @softmax_dim2(%input: memref<2x4x8xf32>, %out: memref<2x4x8xf32>) {
  linalg.softmax dimension(2)
      ins(%input : memref<2x4x8xf32>)
      outs(%out : memref<2x4x8xf32>)
  return
}

// CHECK-LABEL: func.func @softmax_dim2
// CHECK-NOT: linalg.softmax
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: memref.store
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.store
