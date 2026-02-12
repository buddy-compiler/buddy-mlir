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
}

// CHECK-LABEL: func.func @kernel
// CHECK: memref.reinterpret_cast {{.*}} : memref<32xf32> to memref<4x8xf32, strided<[8, 1]{{(, offset: 0)?}}>>
// CHECK: memref.copy {{.*}} : memref<4x8xf32, strided<[8, 1]{{(, offset: 0)?}}>> to memref<4x8xf32>
// CHECK-NOT: memref<4x8xf32, strided<[?, 1], offset: ?>>


