// RUN: buddy-opt %s "-lower-linalg-to-ime=target=k3 a100-fp16=true a100-gemm=true" "-lower-ime=target=k3" --verify-each | FileCheck %s

module {
  func.func @row_tail(%a: memref<12x8xf16>, %b: memref<8x32xf16>,
                      %c: memref<12x32xf16>) {
    linalg.matmul ins(%a, %b : memref<12x8xf16>, memref<8x32xf16>)
        outs(%c : memref<12x32xf16>)
    return
  }
}

// CHECK-LABEL: func.func @row_tail
// CHECK: llvm.inline_asm
// CHECK: scf.parallel
// CHECK: arith.extf
// CHECK: arith.truncf
