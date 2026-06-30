// RUN: buddy-opt %s | FileCheck %s
// RUN: buddy-opt %s \
// RUN:   -lower-ime \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
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

// This example demonstrates the vmadotus operation for mixed signedness matrix multiplication.
// vmadotus performs: C += A × B where A is unsigned int8, B is signed int8, and C is int32 accumulator.
//
// Matrix dimensions:
// A: 4×8 (M×K) - unsigned int8
// B: 8×4 (K×N) - signed int8
// C: 4×4 (M×N) - int32 accumulator

memref.global "private" @matA : memref<4x8xui8> = dense<[
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8]
]>

memref.global "private" @matB : memref<8x4xi8> = dense<[
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1]
]>

func.func @main() -> i32 {
  // Get input matrices
  %a = memref.get_global @matA : memref<4x8xui8>
  %b = memref.get_global @matB : memref<8x4xi8>

  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xi32>

  // Initialize accumulator to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)

  // Perform unsigned × signed matrix multiply-accumulate using IME
  // CHECK: ime.vmadotus
  ime.vmadotus %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xi8>

  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}

// ASM: .attribute 5, "{{.*xsmtime.*}}"
// ASM: vmadotus{{[ \t]}}
