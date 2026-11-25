// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='rvv-vlen=256' \
// RUN:     -convert-linalg-to-loops -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+zvl256b -O2 -o - \
// RUN: | FileCheck %s --check-prefix=ASM

// The vir.reduce of a linalg.reduce whose combiner has the reassoc fast-math
// flag (reassoc, or fast, which has it) is reassoc: the source allows the
// lanes to be combined in any order, an unordered reduction (vfredusum on
// RVV). Without the flag it stays ordered (vfredosum), as the source sums.

// CHECK-LABEL: func.func @sum_ordered
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add"} :
// ASM-LABEL: sum_ordered:
// ASM: vfredosum.vs
// ASM-NOT: vfredusum.vs
// ASM: ret
func.func @sum_ordered(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc : f32
      linalg.yield %r : f32
    }
  return
}

// Other fast-math flags than reassoc: ordered.
// CHECK-LABEL: func.func @sum_nnan
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add"} :
func.func @sum_nnan(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc fastmath<nnan,ninf,nsz,contract> : f32
      linalg.yield %r : f32
    }
  return
}

// Rank 1 to 0.
// CHECK-LABEL: func.func @sum_reassoc
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add", reassoc} :
// ASM-LABEL: sum_reassoc:
// ASM: vfredusum.vs
// ASM-NOT: vfredosum.vs
// ASM: ret
func.func @sum_reassoc(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc fastmath<reassoc> : f32
      linalg.yield %r : f32
    }
  return
}

// fast has reassoc.
// CHECK-LABEL: func.func @max_fast
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "maxnumf", reassoc} :
func.func @max_fast(%in: memref<?xf32>, %out: memref<f32>) {
  linalg.reduce ins(%in : memref<?xf32>) outs(%out : memref<f32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.maxnumf %x, %acc fastmath<fast> : f32
      linalg.yield %r : f32
    }
  return
}

// Rank 2 to 1, the rows.
// CHECK-LABEL: func.func @rows_reassoc
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add", reassoc} :
func.func @rows_reassoc(%in: memref<8x?xf32>, %out: memref<8xf32>) {
  linalg.reduce ins(%in : memref<8x?xf32>) outs(%out : memref<8xf32>) dimensions = [1]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc fastmath<reassoc> : f32
      linalg.yield %r : f32
    }
  return
}

// Rank 2 to 1, the columns (static shapes).
// CHECK-LABEL: func.func @cols_reassoc
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add", reassoc} :
func.func @cols_reassoc(%in: memref<16x8xf32>, %out: memref<8xf32>) {
  linalg.reduce ins(%in : memref<16x8xf32>) outs(%out : memref<8xf32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc fastmath<reassoc> : f32
      linalg.yield %r : f32
    }
  return
}

// Rank 3 to 2, the leading dimension (static shapes).
// CHECK-LABEL: func.func @slices_reassoc
// CHECK: vir.reduce %{{.*}}, %{{.*}} {kind = "add", reassoc} :
func.func @slices_reassoc(%in: memref<16x4x8xf32>, %out: memref<4x8xf32>) {
  linalg.reduce ins(%in : memref<16x4x8xf32>) outs(%out : memref<4x8xf32>) dimensions = [0]
    (%x: f32, %acc: f32) {
      %r = arith.addf %x, %acc fastmath<reassoc> : f32
      linalg.yield %r : f32
    }
  return
}
