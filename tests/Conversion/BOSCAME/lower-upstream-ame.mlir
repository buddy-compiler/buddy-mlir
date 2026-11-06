// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s

// The upstream AME path keeps working: an operation set that uses the old
// memref-based AME load/store operations (no `bosc_ame.tile.*`) is lowered to
// the upstream `bosc_ame.intr.mlae8.m`-style intrinsics, takes no FPGA register
// registers that the backend allocates, and gets no extra target feature.
// FPGA semantics are only
// applied to a module that both uses tile operations and selects a contract.

module {
  // CHECK-LABEL: llvm.func @upstream_load_store
  func.func @upstream_load_store(%c: memref<64x64xi32>) -> i32 {
    %c256 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    // CHECK: "bosc_ame.intr.mlce32.m"(%{{.*}}, %{{.*}}) : (!llvm.ptr, i64) -> vector<[32]xi32>
    %acc = bosc_ame.mlce32.m %c, %c256 : memref<64x64xi32> -> vector<4x4xi32>
    // CHECK: "bosc_ame.intr.msce32.m"(%{{.*}}, %{{.*}}, %{{.*}}) : (vector<[32]xi32>, !llvm.ptr, i64) -> ()
    bosc_ame.msce32.m %acc, %c, %c256 : vector<4x4xi32>, memref<64x64xi32>
    // No FPGA-specific intrinsic may appear on this path.
    // CHECK-NOT: bosc.intr.fpga.
    // CHECK-NOT: target_features
    return %c0_i32 : i32
  }
}
