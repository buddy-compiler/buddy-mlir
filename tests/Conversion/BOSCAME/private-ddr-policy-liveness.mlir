// RUN: buddy-opt %s --lower-bosc-ame --verify-diagnostics

// The source contains no calls inside the ACC lifetime. Final lowering must
// recheck after inserting CPU DDR policy calls and reject the new barrier.
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  func.func @policy_inside_accumulator(%output: memref<1xi8>) {
    %frame = bosc_ame.ddr.frame : memref<256xi8>
    %zero = arith.constant 0 : index
    %p = memref.view %frame[%zero][] : memref<256xi8> to memref<2x16xi32>
    %acc = bosc_ame.tile.load_acc %p {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
    // expected-error @+1 {{cannot call code while an AME accumulator is live}}
    %value = memref.load %frame[%zero] : memref<256xi8>
    bosc_ame.tile.store_acc %acc, %p {bosc_ame.direct_ddr} : vector<2x16xi32>, memref<2x16xi32>
    memref.store %value, %output[%zero] : memref<1xi8>
    bosc_ame.ddr.release %frame : memref<256xi8>
    return
  }
}
