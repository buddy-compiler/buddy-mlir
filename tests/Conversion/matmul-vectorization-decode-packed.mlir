// RUN: buddy-opt %s -matmul-vectorization-decode-packed="vector-size=32 packed-shapes=128x64" | FileCheck %s

// %B is expected to already be physically repacked offline into N-tile
// panels: Bpacked[nt][k][v] == B[k, nt*32 + v]. This pass only changes how
// the kernel addresses the buffer (linear offset = n*K + k*vecSize), not
// the declared memref type, so unpacked shapes are left untouched (see the
// second function below).
func.func @matmul_decode_packed(%A: memref<1x128xf32>,
                                 %B: memref<128x64xf32>,
                                 %C: memref<1x64xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x128xf32>, memref<128x64xf32>)
    outs(%C: memref<1x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_packed
// CHECK: memref.extract_strided_metadata %arg1
// CHECK: memref.reinterpret_cast %base_buffer to offset: [%offset], sizes: [8192], strides: [1]
// CHECK: scf.parallel
// CHECK: %[[PANEL_BASE:.*]] = arith.muli %arg3, %c128
// CHECK: scf.for
// CHECK: %[[K_OFF:.*]] = arith.muli %arg4, %c32
// CHECK: %[[LIN:.*]] = arith.addi %[[PANEL_BASE]], %[[K_OFF]]
// CHECK: vector.load %reinterpret_cast[%[[LIN]]]
// CHECK: vector.fma
// CHECK-NOT: linalg.matmul

// A (K, N) shape not listed in packed-shapes must be left as a plain
// linalg.matmul (no matching pattern fires).
func.func @matmul_decode_unpacked_shape_untouched(%A: memref<1x128xf32>,
                                                   %B: memref<128x96xf32>,
                                                   %C: memref<1x96xf32>) {
  linalg.matmul
    ins(%A, %B: memref<1x128xf32>, memref<128x96xf32>)
    outs(%C: memref<1x96xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_decode_unpacked_shape_untouched
// CHECK: linalg.matmul
