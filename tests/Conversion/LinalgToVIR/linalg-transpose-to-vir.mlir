// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

func.func @transpose_smoke_2d(%src: memref<8x4xf32>, %dst: memref<4x8xf32>) {
  linalg.transpose ins(%src : memref<8x4xf32>)
                   outs(%dst : memref<4x8xf32>)
                   permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @transpose_smoke_2d
// CHECK-NOT: linalg.transpose
// CHECK: vir.set_vl
// CHECK: memref.transpose %arg0
// CHECK: memref.transpose %arg1
// CHECK: vir.load
// CHECK: vir.store
// CHECK-NOT: linalg.transpose
// CHECK: return

// -----

func.func @transpose_i32_rank3(%src: memref<128x2x4xi32>, %dst: memref<4x128x2xi32>) {
  linalg.transpose ins(%src : memref<128x2x4xi32>)
                   outs(%dst : memref<4x128x2xi32>)
                   permutation = [2, 0, 1]
  return
}

// CHECK-LABEL: func.func @transpose_i32_rank3
// CHECK-NOT: linalg.transpose
// CHECK: vir.set_vl
// CHECK: memref.transpose %arg0
// CHECK: memref.transpose %arg1
// CHECK: vir.load
// CHECK: vir.store
// CHECK-NOT: linalg.transpose
// CHECK: return

// -----

func.func @transpose_f32_rank3(%src: memref<128x2x4xf32>, %dst: memref<4x128x2xf32>) {
  linalg.transpose ins(%src : memref<128x2x4xf32>)
                   outs(%dst : memref<4x128x2xf32>)
                   permutation = [2, 0, 1]
  return
}

// CHECK-LABEL: func.func @transpose_f32_rank3
// CHECK-NOT: linalg.transpose
// CHECK: vir.set_vl
// CHECK: memref.transpose %arg0
// CHECK: memref.transpose %arg1
// CHECK: vir.load
// CHECK: vir.store
// CHECK-NOT: linalg.transpose
// CHECK: return

// -----

func.func @transpose_zero_extent(%src: memref<0x4xf32>, %dst: memref<4x0xf32>) {
  linalg.transpose ins(%src : memref<0x4xf32>)
                   outs(%dst : memref<4x0xf32>)
                   permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @transpose_zero_extent
// CHECK-NOT: linalg.transpose
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.store
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transpose
// CHECK-NOT: vector.transfer_write
// CHECK: return
