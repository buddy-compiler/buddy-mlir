// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

func.func @transpose_smoke_2d(%src: memref<8x4xf32>, %dst: memref<4x8xf32>) {
  linalg.transpose ins(%src : memref<8x4xf32>)
                   outs(%dst : memref<4x8xf32>)
                   permutation = [1, 0]
  return
}

// CHECK-LABEL: func.func @transpose_smoke_2d
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %{{.*}} : memref<8x4xf32>, vector<8x4xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x4xf32> to vector<4x8xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_write %{{.*}}, %arg1[%{{.*}}, %{{.*}}]{{.*}} : vector<4x8xf32>, memref<4x8xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
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
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_read %arg0[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<128x2x4xi32>, vector<128x2x4xi32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transpose %{{.*}}, [2, 0, 1] : vector<128x2x4xi32> to vector<4x128x2xi32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_write %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}]{{.*}} : vector<4x128x2xi32>, memref<4x128x2xi32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
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
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_read %arg0[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<128x2x4xf32>, vector<128x2x4xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transpose %{{.*}}, [2, 0, 1] : vector<128x2x4xf32> to vector<4x128x2xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
// CHECK: vector.transfer_write %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}]{{.*}} : vector<4x128x2xf32>, memref<4x128x2xf32>
// CHECK-NOT: linalg.transpose
// CHECK-NOT: vir.set_vl
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
// CHECK-NOT: vir.set_vl
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transpose
// CHECK-NOT: vector.transfer_write
// CHECK: return
