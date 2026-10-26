// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @one_dim(%x: !vir.vec<?xf32>) -> !vir.vec<?xf32> {
  // expected-error @below {{needs a dynamic vector with a leading static dim}}
  %r = vir.shuffle %x, %x [0] : !vir.vec<?xf32>
  return %r : !vir.vec<?xf32>
}

// -----

func.func @mask_size(%x: !vir.vec<2x?xf32>) -> !vir.vec<2x?xf32> {
  // expected-error @below {{needs one mask index per row, 2}}
  %r = vir.shuffle %x, %x [0, 1, 2] : !vir.vec<2x?xf32>
  return %r : !vir.vec<2x?xf32>
}

// -----

func.func @mask_range(%x: !vir.vec<2x?xf32>) -> !vir.vec<2x?xf32> {
  // expected-error @below {{mask index 4 is out of the 4 rows}}
  %r = vir.shuffle %x, %x [4, 0] : !vir.vec<2x?xf32>
  return %r : !vir.vec<2x?xf32>
}
