// RUN: buddy-opt %s -staticize-memref-layout | FileCheck %s

module {
  func.func @kernel() {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index

    %base = memref.alloc() : memref<32xf32>
    %dst = memref.alloc() : memref<4x8xf32>

    %src = memref.reinterpret_cast %base to
      offset: [%c0],
      sizes: [4, 8],
      strides: [%c8, 1]
      : memref<32xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

    memref.copy %src, %dst
      : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<4x8xf32>

    return
  }

  // The second of two heads of 4 x 8: the offset and the strides are kept.
  func.func @second_head() {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index

    %base = memref.alloc() : memref<64xf32>
    %dst = memref.alloc() : memref<4x8xf32>

    %src = memref.reinterpret_cast %base to
      offset: [%c32],
      sizes: [4, 8],
      strides: [%c16, 1]
      : memref<64xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

    memref.copy %src, %dst
      : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<4x8xf32>

    return
  }

  // An offset known at run time only: the layout stays dynamic.
  func.func @runtime_offset(%offset: index) {
    %c8 = arith.constant 8 : index

    %base = memref.alloc() : memref<64xf32>
    %dst = memref.alloc() : memref<4x8xf32>

    %src = memref.reinterpret_cast %base to
      offset: [%offset],
      sizes: [4, 8],
      strides: [%c8, 1]
      : memref<64xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

    memref.copy %src, %dst
      : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<4x8xf32>

    return
  }
}

// CHECK-LABEL: func.func @kernel
// CHECK: memref.reinterpret_cast {{.*}} : memref<32xf32> to memref<4x8xf32, strided<[8, 1]{{(, offset: 0)?}}>>
// CHECK: memref.copy {{.*}} : memref<4x8xf32, strided<[8, 1]{{(, offset: 0)?}}>> to memref<4x8xf32>
// CHECK-NOT: memref<4x8xf32, strided<[?, 1], offset: ?>>

// CHECK-LABEL: func.func @second_head
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [32], sizes: [4, 8], strides: [16, 1] : memref<64xf32> to memref<4x8xf32, strided<[16, 1], offset: 32>>
// CHECK: memref.copy {{.*}} : memref<4x8xf32, strided<[16, 1], offset: 32>> to memref<4x8xf32>

// CHECK-LABEL: func.func @runtime_offset
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [4, 8], strides: [%{{.*}}, 1] : memref<64xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>
// CHECK: memref.copy {{.*}} : memref<4x8xf32, strided<[?, 1], offset: ?>> to memref<4x8xf32>
