// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
// RUN: buddy-opt %s \
// RUN:   -lower-linalg-to-ime \
// RUN:   -lower-ime \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -expand-strided-metadata \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64 \
// RUN:   -mattr=+m,+v,+xsmtime -o - | FileCheck %s --check-prefix=ASM
//
// Test case: C[7x5] = A[7x10] * B[10x5] with non-aligned dimensions
// For int8: TILE_M=4, TILE_K=8, TILE_N=4
// This file can also be compiled and linked with runtime_matmul_boundary.c

// CHECK-LABEL: func.func @matmul_boundary
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_boundary(%A: memref<7x10xi8>, %B: memref<10x5xi8>,
                            %C: memref<7x5xi32>) {
  linalg.matmul ins(%A, %B : memref<7x10xi8>, memref<10x5xi8>)
                outs(%C : memref<7x5xi32>)
  return
}

// ASM: .attribute 5, "{{.*xsmtime.*}}"
// ASM: vmadot{{[ \t]}}
