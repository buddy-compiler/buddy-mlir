// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -expand-strided-metadata -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='rvv-vlen=128' \
// RUN:     -expand-strided-metadata -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -expand-strided-metadata -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -O2 -o - \
// RUN: | FileCheck %s --check-prefix=ASM

// A memref.copy that finalize-memref-to-llvm makes a call of memrefCopy (an
// element at a time: a dynamic shape with an offset, or rows with a stride)
// with a unit stride in the last dim on both sides is a vector load and
// store per row.

// The copy of a masked load: a dynamic length, an offset.
// CHECK-LABEL: func.func @masked
// CHECK: vir.set_vl %{{.*}} : index {
// CHECK:   %[[V:.*]] = vir.load %{{.*}}[%{{.*}}] : memref<?xf32, strided<[1], offset: ?>> -> !vir.vec<?xf32>
// CHECK:   vir.store %[[V]], %{{.*}}[%{{.*}}] : !vir.vec<?xf32> -> memref<?xf32, strided<[1], offset: ?>>
// CHECK-NOT: memref.copy
// CHECK: return
// LLVM-LABEL: @masked
// LLVM-NOT: @memrefCopy
// LLVM: return
// ASM-LABEL: masked:
// ASM: vl8re32.v
// ASM: vs8r.v
func.func @masked(%src: memref<?xf32, strided<[1], offset: ?>>,
                  %dst: memref<?xf32, strided<[1], offset: ?>>) {
  memref.copy %src, %dst : memref<?xf32, strided<[1], offset: ?>>
                        to memref<?xf32, strided<[1], offset: ?>>
  return
}

// A tile of a matrix: rows with a stride, in a loop (its vl is computed in
// the loop: the loops of fixed vectors are scf.for too).
// CHECK-LABEL: func.func @tiles
// CHECK: scf.for
// CHECK:   vir.set_vl %{{.*}} : index {
// CHECK:     scf.for %[[R:.*]] = %{{.*}} to %{{.*}} step
// CHECK:       %[[V:.*]] = vir.load %{{.*}}[%[[R]], %{{.*}}] : memref<?x?xi32, strided<[?, 1], offset: ?>> -> !vir.vec<?xi32>
// CHECK:       vir.store %[[V]], %{{.*}}[%[[R]], %{{.*}}] : !vir.vec<?xi32> -> memref<?x?xi32, strided<[64, 1]>>
// CHECK-NOT: memref.copy
// CHECK: return
// LLVM-LABEL: @tiles
// LLVM-NOT: @memrefCopy
// LLVM: return
func.func @tiles(%src: memref<?x?xi32, strided<[?, 1], offset: ?>>,
                 %dst: memref<64x64xi32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %n step %c1 {
    %rest = arith.subi %n, %i : index
    %m = arith.minsi %rest, %c64 : index
    %s = memref.subview %src[%i, 0] [%m, %m] [1, 1]
        : memref<?x?xi32, strided<[?, 1], offset: ?>>
       to memref<?x?xi32, strided<[?, 1], offset: ?>>
    %d = memref.subview %dst[0, 0] [%m, %m] [1, 1]
        : memref<64x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    memref.copy %s, %d : memref<?x?xi32, strided<[?, 1], offset: ?>>
                      to memref<?x?xi32, strided<[64, 1]>>
  }
  return
}

// Left as they are: a copy that is an llvm.memcpy, a stride in the last
// dim, and elements of i1.
// CHECK-LABEL: func.func @kept
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64xf32> to memref<64xf32>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?xf32, strided<[2]>> to memref<?xf32>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?xi1, strided<[1], offset: ?>> to memref<?xi1, strided<[1], offset: ?>>
// CHECK-NOT: vir.
// CHECK: return
func.func @kept(%a: memref<64xf32>, %b: memref<64xf32>,
                %c: memref<?xf32, strided<[2]>>, %d: memref<?xf32>,
                %e: memref<?xi1, strided<[1], offset: ?>>,
                %f: memref<?xi1, strided<[1], offset: ?>>) {
  memref.copy %a, %b : memref<64xf32> to memref<64xf32>
  memref.copy %c, %d : memref<?xf32, strided<[2]>> to memref<?xf32>
  memref.copy %e, %f : memref<?xi1, strided<[1], offset: ?>>
                    to memref<?xi1, strided<[1], offset: ?>>
  return
}
