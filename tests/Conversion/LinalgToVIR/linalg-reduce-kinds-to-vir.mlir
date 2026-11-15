// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN: | FileCheck %s --check-prefix=VEC
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -convert-linalg-to-loops -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -O2 -o - \
// RUN: | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='rvv-vlen=128' \
// RUN:     -convert-linalg-to-loops -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -O2 -filetype=obj -o /dev/null

// The combiner of a linalg.reduce, one binary arith op of its arguments, is
// the kind of the vir.reduce (a vector.reduction kind), on integers or
// f16 / f32 / f64.

// CHECK-LABEL: func.func @sum_i32
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add"} : !vir.vec<?xi32>, i32 -> i32
// VEC-LABEL: func.func @sum_i32
// VEC: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<[{{[0-9]+}}]xi32> into i32
// VEC: arith.addi %{{.*}}, %{{.*}} : i32
// ASM-LABEL: sum_i32:
// ASM: vredsum.vs
func.func @sum_i32(%in: memref<?xi32>, %out: memref<i32>) {
  linalg.reduce ins(%in : memref<?xi32>) outs(%out : memref<i32>) dimensions = [0]
    (%x: i32, %acc: i32) {
      %r = arith.addi %x, %acc : i32
      linalg.yield %r : i32
    }
  return
}

// CHECK-LABEL: func.func @all_i1
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "and"} : !vir.vec<?xi1>, i1 -> i1
// ASM-LABEL: all_i1:
// ASM: vcpop.m
func.func @all_i1(%in: memref<?xi1>, %out: memref<i1>) {
  linalg.reduce ins(%in : memref<?xi1>) outs(%out : memref<i1>) dimensions = [0]
    (%x: i1, %acc: i1) {
      %r = arith.andi %acc, %x : i1
      linalg.yield %r : i1
    }
  return
}

// CHECK-LABEL: func.func @prod_f32
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "mul"} : !vir.vec<?xf32>, f32 -> f32
func.func @prod_f32(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.mulf %x, %acc : f32
      linalg.yield %r : f32
    }
  return
}

// CHECK-LABEL: func.func @min_f32
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "minnumf"} : !vir.vec<?xf32>, f32 -> f32
// ASM-LABEL: min_f32:
// ASM: vfredmin.vs
func.func @min_f32(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.minnumf %x, %acc : f32
      linalg.yield %r : f32
    }
  return
}

// Rows: rank 2 -> rank 1, the last dim.
// CHECK-LABEL: func.func @row_min_i64
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "minsi"} : !vir.vec<?xi64>, i64 -> i64
// ASM-LABEL: row_min_i64:
// ASM: vredmin.vs
func.func @row_min_i64(%in: memref<?x?xi64>, %out: memref<?xi64>) {
  linalg.reduce ins(%in : memref<?x?xi64>) outs(%out : memref<?xi64>) dimensions = [1]
    (%x: i64, %acc: i64) {
      %r = arith.minsi %x, %acc : i64
      linalg.yield %r : i64
    }
  return
}

// Columns: static rank 2 -> rank 1, dim 0.
// CHECK-LABEL: func.func @col_max_ui
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "maxui"} : !vir.vec<?xi32>, i32 -> i32
func.func @col_max_ui(%in: memref<8x16xi32>, %out: memref<16xi32>) {
  linalg.reduce ins(%in : memref<8x16xi32>) outs(%out : memref<16xi32>) dimensions = [0]
    (%x: i32, %acc: i32) {
      %r = arith.maxui %x, %acc : i32
      linalg.yield %r : i32
    }
  return
}

// A middle dim, the last one kept (bitonic sort): the elementwise combiner
// along the last dim, no horizontal reduction.
// CHECK-LABEL: func.func @middle_dim_i32
// CHECK: vir.set_vl
// CHECK: scf.for
// CHECK: vir.load {{.*}} -> !vir.vec<?xi32>
// CHECK: arith.addi %{{.*}}, %{{.*}} : !vir.vec<?xi32>
// CHECK-NOT: vir.reduce
// CHECK: return
// ASM-LABEL: middle_dim_i32:
// ASM: vadd.vv
func.func @middle_dim_i32(%in: memref<64x2x8xi32>, %out: memref<64x8xi32>) {
  linalg.reduce ins(%in : memref<64x2x8xi32>) outs(%out : memref<64x8xi32>) dimensions = [1]
    (%x: i32, %acc: i32) {
      %r = arith.addi %x, %acc : i32
      linalg.yield %r : i32
    }
  return
}

// CHECK-LABEL: func.func @sum_f64
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add"} : !vir.vec<?xf64>, f64 -> f64
// ASM-LABEL: sum_f64:
// ASM: vfredosum.vs
func.func @sum_f64(%in: memref<?xf64>, %out: memref<f64>) {
  linalg.reduce ins(%in : memref<?xf64>) outs(%out : memref<f64>) dimensions = [0]
    (%x: f64, %acc: f64) {
      %r = arith.addf %x, %acc : f64
      linalg.yield %r : f64
    }
  return
}

// maximumnumf: the NaNs are dropped, as maxnumf, but -0 < +0, as maximumf.
// CHECK-LABEL: func.func @max_num_f32
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "maximumnumf"} : !vir.vec<?xf32>, f32 -> f32
// VEC-LABEL: func.func @max_num_f32
// LLVM has no scalable fmaximumnum reduction on RVV: the NaN lanes become
// -inf for a maximumf reduction, NaN if all are.
// VEC: %[[N:.*]] = arith.cmpf uno, %[[V:.*]], %[[V]] : vector<[{{[0-9]+}}]xf32>
// VEC: vector.reduction <maximumf>, %{{.*}}, %{{.*}} : vector<[{{[0-9]+}}]xf32> into f32
// VEC: vector.reduction <and>, %[[N]], %{{.*}} : vector<[{{[0-9]+}}]xi1> into i1
// VEC: arith.maximumnumf %{{.*}}, %{{.*}} : f32
// ASM-LABEL: max_num_f32:
// ASM: vfredmax.vs
func.func @max_num_f32(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.maximumnumf %x, %acc : f32
      linalg.yield %r : f32
    }
  return
}
