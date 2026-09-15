// RUN: buddy-opt %s -lower-linalg-to-boscame --lower-bosc-ame | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-boscame --lower-bosc-ame | \
// RUN:   FileCheck %s --check-prefix=ABSENT
//
// The same separation must hold after translation to LLVM IR - the
// immediate-form configuration intrinsic is used, and no FPGA target feature,
// register-form configuration or bit-field constant appears anywhere.
// RUN: buddy-opt %s -lower-linalg-to-boscame --lower-bosc-ame \
// RUN:   -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   -expand-strided-metadata -lower-affine -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts | \
// RUN:   buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=IR
//
// IR: @llvm.riscv.bosc.msettypei.i64
// IR-NOT: xboscame-fpga
// IR-NOT: target-features
// IR-NOT: bosc.msettype.i64
// IR-NOT: 65552
// IR-NOT: 65602
//
// A lone set of CHECK-NOT directives asserts absence over the whole input.
// ABSENT-NOT: 65552
// ABSENT-NOT: 65602
// ABSENT-NOT: xboscame-fpga
// ABSENT-NOT: bosc_ame.target
// ABSENT-NOT: bosc_ame.msettype %
//
// Dual-target separation (plan phase 4): with the default profile the whole
// chain stays upstream.  The BOSCAME export programs the immediate raw-width
// configuration, and none of the FPGA markers appear anywhere in the output -
// not the bit-field constants, not the FPGA target feature, not the profile
// attribute and not the register-form msettype.
//
// CHECK: bosc_ame.intr.msettypei
// CHECK: bosc_ame.intr.mma.w.mm
module {
  func.func @matmul_i32_4x4x4(%A: memref<4x4xi32>, %B: memref<4x4xi32>, %C: memref<4x4xi32>) {
    linalg.matmul ins(%A, %B : memref<4x4xi32>, memref<4x4xi32>) outs(%C : memref<4x4xi32>)
    return
  }
}

// IR: declare {{.*}}@llvm.riscv.bosc.mma.w.mm{{.*}} #[[MMA_EFFECTS:[0-9]+]]
// IR: attributes #[[MMA_EFFECTS]] = { nounwind memory(none) }
