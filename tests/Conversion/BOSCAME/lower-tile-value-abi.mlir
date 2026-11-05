// RUN: buddy-opt -verify-diagnostics %s --lower-bosc-ame

// The first version has no matrix-value ABI: an AME matrix value only exists
// inside the block that produced it, and it may only be consumed by a
// `bosc_ame.tile.mma` or a `bosc_ame.tile.store_acc`.  A matrix value that
// arrives from outside (a function argument, a region yield, a call) has no
// defined producer in the lowered code, so it is rejected.  This is a scope
// rule on the values - which register holds a tile is the backend's decision
// and is not part of the check.

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @foreign_matrix_value(%ta: vector<8x16xi8>,
                                  %tb: vector<16x32xi8>) {
    %acc = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<8x32xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>
    // expected-error@+1 {{must be produced by a `bosc_ame.tile.*` operation}}
    %r = bosc_ame.tile.mma %tc, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    bosc_ame.tile.store_acc %r, %acc : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
    return
  }
}
