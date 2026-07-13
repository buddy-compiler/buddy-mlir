// RUN: buddy-opt %s -matmul-vectorization-decode-packed="vector-size=32" | FileCheck %s

// An empty `packed-shapes` means "every m == 1 matmul weight in this module is
// panel-packed", so all of them are rewritten regardless of shape. This is what
// the pack_decode_matmul_weights graph transform guarantees: it packs every
// matmul weight in the decode graph, and refuses to run if it cannot, so there
// is no opt-in list to keep in step with it.
//
// Both functions below have a different (K, N); both must be rewritten.

func.func @matmul_decode_packed_128x64(%A: memref<1x128xf32>,
                                       %B: memref<128x64xf32>,
                                       %C: memref<1x64xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x128xf32>, memref<128x64xf32>)
    outs(%C: memref<1x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_packed_128x64
// CHECK: memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [8192], strides: [1]
// CHECK: scf.parallel
// CHECK: vector.load
// CHECK: vector.fma
// CHECK-NOT: linalg.matmul

func.func @matmul_decode_packed_64x96(%A: memref<1x64xf32>,
                                      %B: memref<64x96xf32>,
                                      %C: memref<1x96xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x64xf32>, memref<64x96xf32>)
    outs(%C: memref<1x96xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_packed_64x96
// CHECK: memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [6144], strides: [1]
// CHECK: scf.parallel
// CHECK: vector.load
// CHECK: vector.fma
// CHECK-NOT: linalg.matmul

// N not divisible by vecSize cannot be panel-packed at this width, so it is
// left alone even in "pack everything" mode -- the kernel's addressing assumes
// whole panels.
func.func @matmul_decode_ragged_n_untouched(%A: memref<1x64xf32>,
                                            %B: memref<64x50xf32>,
                                            %C: memref<1x50xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x64xf32>, memref<64x50xf32>)
    outs(%C: memref<1x50xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_ragged_n_untouched
// CHECK: linalg.matmul
