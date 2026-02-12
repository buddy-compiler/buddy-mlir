// RUN: buddy-opt %s -eliminate-memref-copy | FileCheck %s

module {
  // Test basic case: copy from function argument to allocation
  func.func @test_basic(%arg0: memref<10xf32, strided<[?], offset: ?>>) -> memref<10xf32> {
    %alloc = memref.alloc() : memref<10xf32>
    memref.copy %arg0, %alloc : memref<10xf32, strided<[?], offset: ?>> to memref<10xf32>
    // Use the allocated memref
    %c0 = arith.constant 0 : index
    %val = memref.load %alloc[%c0] : memref<10xf32>
    memref.store %val, %alloc[%c0] : memref<10xf32>
    return %alloc : memref<10xf32>
  }

  // Test multi-dimensional case with strided layout
  func.func @test_multidim(%arg0: memref<1x2x1024x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x2x1024x128xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x1024x128xf32>
    memref.copy %arg0, %alloc : memref<1x2x1024x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x2x1024x128xf32>
    // Use the allocated memref
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = memref.load %alloc[%c0, %c0, %c0, %c0] : memref<1x2x1024x128xf32>
    memref.store %val, %alloc[%c0, %c0, %c0, %c0] : memref<1x2x1024x128xf32>
    return %alloc : memref<1x2x1024x128xf32>
  }

  // Test case where copy is followed by operations
  func.func @test_with_ops(%arg0: memref<5xf32, strided<[?], offset: ?>>) -> memref<5xf32> {
    %alloc = memref.alloc() : memref<5xf32>
    memref.copy %arg0, %alloc : memref<5xf32, strided<[?], offset: ?>> to memref<5xf32>
    // Some operations using the allocated memref
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val0 = memref.load %alloc[%c0] : memref<5xf32>
    %val1 = memref.load %alloc[%c1] : memref<5xf32>
    %sum = arith.addf %val0, %val1 : f32
    memref.store %sum, %alloc[%c0] : memref<5xf32>
    return %alloc : memref<5xf32>
  }
}

// CHECK-LABEL: func.func @test_basic
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strided<[1]
// CHECK: memref.cast
// CHECK-SAME: memref<10xf32>
// CHECK: memref.load
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_multidim
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strided<[?, ?, ?, 1]
// CHECK: memref.cast
// CHECK-SAME: memref<1x2x1024x128xf32>
// CHECK: memref.load
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_with_ops
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strided<[1]
// CHECK: memref.cast
// CHECK-SAME: memref<5xf32>
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store
// CHECK: return
