// RUN: buddy-opt %s --lower-bosc-ame --split-input-file | FileCheck %s
//
// CPU DDR policy inlining: NR target gets CSR 0xbc0 fences for vector
// load/store; calls without compiler provenance keep the runtime helper ABI;
// non-NR targets keep explicit policy calls. CPU-only modules still need an
// explicit NR target before inlining.
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  // CHECK-LABEL: llvm.func @nr_vector_copy
  func.func @nr_vector_copy(%source: memref<16xi32>, %target: memref<16xi32>) {
    %zero = arith.constant 0 : index
    // CHECK: %[[READ:.*]] = llvm.mlir.constant(1085086099645071375 : i64)
    // CHECK-NEXT: llvm.inline_asm has_side_effects "fence iorw, iorw\0A\09csrw 0xbc0, $0", "r,~{memory}" %[[READ]] : (i64) -> ()
    %value = vector.load %source[%zero] : memref<16xi32>, vector<16xi32>
    // CHECK: %[[WRITE:.*]] = llvm.mlir.constant(1085102592571150095 : i64)
    // CHECK-NEXT: llvm.inline_asm has_side_effects "fence iorw, iorw\0A\09csrw 0xbc0, $0", "r,~{memory}" %[[WRITE]] : (i64) -> ()
    vector.store %value, %target[%zero] : memref<16xi32>, vector<16xi32>
    return
  }
}

// -----
// No compiler provenance: a call to the reserved name retains its ABI.
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  func.func private @bosc_ame_ddr_read_begin()
  // CHECK-LABEL: llvm.func @user_call
  func.func @user_call() {
    // CHECK: llvm.call @bosc_ame_ddr_read_begin() : () -> ()
    func.call @bosc_ame_ddr_read_begin() : () -> ()
    return
  }
}

// -----
// Another target keeps runtime policy calls even for compiler transitions.
module attributes {bosc_ame.target = "other", bosc_ame.cpu_ddr_policy} {
  // CHECK-LABEL: llvm.func @other_target
  func.func @other_target(%source: memref<1xi32>) -> i32 {
    %zero = arith.constant 0 : index
    // CHECK: llvm.call @bosc_ame_ddr_read_begin(){{.*}} : () -> ()
    %value = memref.load %source[%zero] : memref<1xi32>
    return %value : i32
  }
}
