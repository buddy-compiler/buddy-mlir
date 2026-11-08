// RUN: buddy-opt %s -lower-linalg-to-xtame | FileCheck %s
// RUN: not buddy-opt %s -lower-linalg-to-xtame="max-temporary-bytes=0" \
// RUN:   2>&1 | FileCheck %s --check-prefix=BUDGET
// RUN: buddy-opt %s -lower-linalg-to-xtame -lower-xt-ame \
// RUN:   -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm -expand-strided-metadata -lower-affine \
// RUN:   -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
// RUN:   -mattr=+m,+a,+c,+d,+xtheadame -o - | FileCheck %s --check-prefix=ASM

// The batch form walks the batch dimension with a loop over contiguous slices,
// so each iteration lowers to one matmul on a rank-reduced subview.
// CHECK: memref.global "private" constant @xtame_ones_4_8 : memref<4x8xi8> = dense<1>
// CHECK: memref.global "private" constant @xtame_ones_8_1 : memref<8x1xi8> = dense<1>
// CHECK-LABEL: func.func @batch_matmul_i8(
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: xt_ame.th.mcfgni 4
// CHECK-COUNT-1: xt_ame.th.mmacc.w.b

// ASM-LABEL: batch_matmul_i8:
// ASM: th.mcfgni 4
// ASM: th.mmacc.w.b
// ASM: th.mste32

module {
  func.func @batch_matmul_i8(%a: memref<2x4x8xi8>,
                             %b: memref<2x8x4xi8>,
                             %c: memref<2x4x4xi32>) {
    linalg.batch_matmul ins(%a, %b : memref<2x4x8xi8>, memref<2x8x4xi8>)
                        outs(%c : memref<2x4x4xi32>)
    return
  }

  // The quantized form accumulates (A - AZp) * (B - BZp): the matmul produces
  // sum_k A*B, then two more matmuls with a vector of ones produce the row sums
  // of A and the column sums of B, and a scalar loop adds the correction.
  // CHECK-LABEL: func.func @quantized_matmul_i8(
  // CHECK: xt_ame.th.mmacc.w.b
  // CHECK: memref.alloca() : memref<4x1xi32>
  // CHECK: memref.alloca() : memref<4x4xi32>
  // CHECK: memref.alloca() : memref<4xi32>
  // CHECK-COUNT-2: xt_ame.th.mmacc.w.b
  // CHECK: arith.muli
  // CHECK: arith.subi
  //
  // The temporaries of the correction are a stack allocation, so a shape above
  // the budget is rejected instead of overflowing the stack at run time.
  // BUDGET: error: the zero-point correction needs 96 bytes of stack temporaries
  func.func @quantized_matmul_i8(%a: memref<4x8xi8>,
                                 %b: memref<8x4xi8>,
                                 %azp: i32, %bzp: i32,
                                 %c: memref<4x4xi32>) {
    linalg.quantized_matmul ins(%a, %b, %azp, %bzp :
                                  memref<4x8xi8>, memref<8x4xi8>, i32, i32)
                            outs(%c : memref<4x4xi32>)
    return
  }
}
