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
// RUN:   -mattr=+m,+v,+zvfh,+xsmtime -o - | FileCheck %s --check-prefix=ASM
//
// Test case: C[7x5] = A[7x6] * B[6x5] with fp16, non-aligned dimensions
// FP16 tile sizes: TILE_M=4, TILE_K=4, TILE_N=4
//   M=7: 1 full tile (4) + 3 remaining
//   K=6: 1 full tile (4) + 2 remaining
//   N=5: 1 full tile (4) + 1 remaining
//
// This file is designed to also be compiled and linked with
// runtime_matmul_fp16_boundary.c for functional verification on hardware.

// CHECK-LABEL: func.func @matmul_fp16_boundary
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_fp16_boundary(%A: memref<7x6xf16>, %B: memref<6x5xf16>,
                                 %C: memref<7x5xf16>) {
  linalg.matmul ins(%A, %B : memref<7x6xf16>, memref<6x5xf16>)
                outs(%C : memref<7x5xf16>)
  return
}

// ASM: .attribute 5, "{{.*zvfh.*xsmtime.*}}"
// ASM: vfmadot{{[ \t]}}
