// RUN: buddy-opt %s --elide-masked-tile-copies | FileCheck %s
// RUN: buddy-opt %s --elide-masked-tile-copies --elide-masked-tile-copies | FileCheck %s

// CHECK-LABEL: func.func @masked_1d(
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi ule
// CHECK: scf.if
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: linalg.generic
// CHECK: } else {
// CHECK-NOT: scf.if
// CHECK: memref.alloca
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK: memref.copy
func.func @masked_1d(%src: memref<16xf32>, %dst: memref<16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<16xf32>
  %s = memref.subview %src[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %a1 = memref.subview %a[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %s, %a1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %x : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %d = memref.subview %dst[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %b1, %d : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  return
}

// CHECK-LABEL: func.func @masked_2d_other(
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi ule
// CHECK: scf.if
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: linalg.generic
// CHECK: } else {
// CHECK-NOT: scf.if
// CHECK: memref.alloca
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK: memref.copy
func.func @masked_2d_other(%src: memref<16x16xf32>, %dst: memref<16x16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16x16xf32>
  %b = memref.alloca() : memref<16x16xf32>
  %other = arith.constant 9.0 : f32
  linalg.fill ins(%other : f32) outs(%a : memref<16x16xf32>)
  %s = memref.subview %src[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  %a1 = memref.subview %a[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %s, %a1 : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[16, 1]>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : memref<16x16xf32>) outs(%b : memref<16x16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %x : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  %d = memref.subview %dst[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %b1, %d : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[16, 1]>>
  return
}

// CHECK-LABEL: func.func @different_sizes(
// CHECK-NOT: scf.if
func.func @different_sizes(%src: memref<16xf32>, %dst: memref<16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<16xf32>
  %s = memref.subview %src[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %a1 = memref.subview %a[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %s, %a1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %x : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %d = memref.subview %dst[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %b1, %d : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  return
}

// CHECK-LABEL: func.func @reads_out(
// CHECK-NOT: scf.if
func.func @reads_out(%src: memref<16xf32>, %dst: memref<16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<16xf32>
  %s = memref.subview %src[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %a1 = memref.subview %a[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %s, %a1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %old : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %d = memref.subview %dst[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %b1, %d : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  return
}

// CHECK-LABEL: func.func @non_identity(
// CHECK-NOT: scf.if
func.func @non_identity(%src: memref<16x16xf32>, %dst: memref<16x16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16x16xf32>
  %b = memref.alloca() : memref<16x16xf32>
  %s = memref.subview %src[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  %a1 = memref.subview %a[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %s, %a1 : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[16, 1]>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : memref<16x16xf32>) outs(%b : memref<16x16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %x : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  %d = memref.subview %dst[0, 0] [%n, %n] [1, 1] : memref<16x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %b1, %d : memref<?x?xf32, strided<[16, 1]>> to memref<?x?xf32, strided<[16, 1]>>
  return
}

// CHECK-LABEL: func.func @extra_user(
// CHECK-NOT: scf.if
func.func @extra_user(%src: memref<16xf32>, %dst: memref<16xf32>, %n: index, %m: index) {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<16xf32>
  %s = memref.subview %src[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %a1 = memref.subview %a[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %s, %a1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  %extra = memref.load %a[%n] : memref<16xf32>
  func.call @consume(%extra) : (f32) -> ()
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %v = arith.addf %x, %x : f32
    linalg.yield %v : f32
  }
  %b1 = memref.subview %b[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %d = memref.subview %dst[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  memref.copy %b1, %d : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  return
}

func.func private @consume(f32)

// Staging allocations in an enclosing block cannot move into the fallback.
// CHECK-LABEL: func.func @hoisted_buffer(
// CHECK: scf.if
// CHECK-NOT: memref.extract_aligned_pointer_as_index
// CHECK: return
func.func @hoisted_buffer(%src: memref<16xf32>, %dst: memref<16xf32>, %n: index, %cond: i1) {
  %a = memref.alloca() : memref<16xf32>
  %b = memref.alloca() : memref<16xf32>
  scf.if %cond {
    %s = memref.subview %src[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
    %a1 = memref.subview %a[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
    memref.copy %s, %a1 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
    ^bb0(%x: f32, %old: f32):
      linalg.yield %x : f32
    }
    %b1 = memref.subview %b[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
    %d = memref.subview %dst[0] [%n] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
    memref.copy %b1, %d : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
  }
  return
}
