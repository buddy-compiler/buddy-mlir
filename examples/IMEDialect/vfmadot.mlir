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
// RUN:   -mattr=+m,+v,+zvfh,+xsmtime -o - | FileCheck %s --check-prefix=ASM

// This example demonstrates the vfmadot operation for fp16 matrix multiplication.
// vfmadot performs: C += A × B where A, B are fp16 matrices and C is fp16 accumulator.
//
// Matrix dimensions for VLEN=256, SEW=16:
// A: 4×4 (M×K) - float16
// B: 4×4 (K×N) - float16
// C: 4×4 (M×N) - float16 accumulator

memref.global "private" @matA : memref<4x4xf16> = dense<[
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0]
]>

memref.global "private" @matB : memref<4x4xf16> = dense<[
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0]
]>

func.func @main() -> i32 {
  // Get input matrices
  %a = memref.get_global @matA : memref<4x4xf16>
  %b = memref.get_global @matB : memref<4x4xf16>

  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xf16>

  // Initialize accumulator to zero
  %zero = arith.constant 0.0 : f16
  linalg.fill ins(%zero : f16) outs(%c : memref<4x4xf16>)

  // Perform float16 matrix multiply-accumulate using IME
  // CHECK: ime.vfmadot
  ime.vfmadot %c, %a, %b : memref<4x4xf16>, memref<4x4xf16>, memref<4x4xf16>

  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}

// ASM: .attribute 5, "{{.*zvfh.*xsmtime.*}}"
// ASM: vfmadot{{[ \t]}}
