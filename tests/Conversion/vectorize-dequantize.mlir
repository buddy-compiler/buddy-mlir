// RUN: buddy-opt %s --vectorize-dequantize --canonicalize --cse | FileCheck %s

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

// Dynamic lengths and nonzero offsets are supported; there is no speculative
// load past the last full vector. The two multiplies must remain ordered.
// CHECK-LABEL: func.func @dequantize
// CHECK: %[[END:.*]] = arith.subi
// CHECK: scf.for %[[IV:.*]] = {{.*}} to %[[END]] step
// CHECK: %[[I:.*]] = vector.load %arg0[%[[IV]]]
// CHECK: %[[W:.*]] = vector.load %arg1[%[[IV]]]
// CHECK: %[[F:.*]] = arith.sitofp %[[I]] : vector<16xi32> to vector<16xf32>
// CHECK: %[[A:.*]] = vector.broadcast %arg2 : f32 to vector<16xf32>
// CHECK: %[[FA:.*]] = arith.mulf %[[F]], %[[A]] : vector<16xf32>
// CHECK: %[[FAW:.*]] = arith.mulf %[[FA]], %[[W]] : vector<16xf32>
// CHECK: vector.store %[[FAW]]
// CHECK: scf.for %[[TAIL:.*]] = %[[END]] to
// CHECK: memref.load %arg0[%[[TAIL]]]
// CHECK: arith.sitofp {{.*}} : i32 to f32
// CHECK: arith.mulf {{.*}} : f32
// CHECK: arith.mulf {{.*}} : f32
// CHECK: memref.store
func.func @dequantize(
    %acc: memref<?xi32, strided<[1], offset: ?>>,
    %weightScale: memref<?xf32, strided<[1], offset: ?>>,
    %activationScale: f32) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %n = memref.dim %acc, %c0 : memref<?xi32, strided<[1], offset: ?>>
  %out = memref.alloc(%n) : memref<?xf32>
  linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
      ins(%acc, %weightScale : memref<?xi32, strided<[1], offset: ?>>,
                              memref<?xf32, strided<[1], offset: ?>>)
      outs(%out : memref<?xf32>) {
    ^bb0(%a: i32, %w: f32, %unused: f32):
      %f = arith.sitofp %a : i32 to f32
      %fa = arith.mulf %f, %activationScale : f32
      %faw = arith.mulf %fa, %w : f32
      linalg.yield %faw : f32
  }
  return %out : memref<?xf32>
}

// Rank-zero memrefs and scalar operands are broadcast, not loaded per lane.
// CHECK-LABEL: func.func @scalar_inputs
// CHECK: scf.for
// CHECK: vector.load
// CHECK: memref.load %arg1[]
// CHECK: vector.broadcast
// CHECK: vector.broadcast %arg2
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: vector.store
func.func @scalar_inputs(%x: memref<32xf32>, %a: memref<f32>, %b: f32)
    -> memref<32xf32> {
  %out = memref.alloc() : memref<32xf32>
  linalg.generic {indexing_maps = [#id, #scalar, #scalar, #id],
                  iterator_types = ["parallel"]}
      ins(%x, %a, %b : memref<32xf32>, memref<f32>, f32)
      outs(%out : memref<32xf32>) {
    ^bb0(%v: f32, %s: f32, %t: f32, %unused: f32):
      %vs = arith.mulf %v, %s : f32
      %vst = arith.mulf %vs, %t : f32
      linalg.yield %vst : f32
  }
  return %out : memref<32xf32>
}

// Exact in-place identity updates are legal.
// CHECK-LABEL: func.func @in_place
// CHECK: vector.load %arg0
// CHECK: arith.mulf {{.*}} : vector<16xf32>
// CHECK: vector.store {{.*}}, %arg0
func.func @in_place(%x: memref<16xf32>, %scale: f32) {
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<16xf32>) outs(%x : memref<16xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %vs = arith.mulf %v, %scale : f32
      linalg.yield %vs : f32
  }
  return
}

// Unknown aliasing of function arguments cannot establish independent lanes.
// CHECK-LABEL: func.func @may_alias
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @may_alias(%x: memref<16xf32>, %out: memref<16xf32>, %scale: f32) {
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %vs = arith.mulf %v, %scale : f32
      linalg.yield %vs : f32
  }
  return
}

// A dynamic stride is not evidence of contiguous access.
// CHECK-LABEL: func.func @dynamic_stride
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @dynamic_stride(%x: memref<16xf32, strided<[?], offset: ?>>,
                          %scale: f32) -> memref<16xf32> {
  %out = memref.alloc() : memref<16xf32>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<16xf32, strided<[?], offset: ?>>)
      outs(%out : memref<16xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %vs = arith.mulf %v, %scale : f32
      linalg.yield %vs : f32
  }
  return %out : memref<16xf32>
}

// The pass does not rewrite arbitrary arithmetic or alter reduction semantics.
// CHECK-LABEL: func.func @unsupported
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: return
func.func @unsupported(%x: memref<16xf32>, %scale: f32) -> memref<16xf32> {
  %out = memref.alloc() : memref<16xf32>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<16xf32>) outs(%out : memref<16xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %vs = arith.addf %v, %scale : f32
      linalg.yield %vs : f32
  }
  return %out : memref<16xf32>
}

// Overlapping shifted subviews have a real cross-lane dependence. They must
// not be confused with the exact in-place identity case above.
// CHECK-LABEL: func.func @shifted_alias
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @shifted_alias(%scale: f32) -> memref<33xf32> {
  %storage = memref.alloc() : memref<33xf32>
  %src = memref.subview %storage[0] [32] [1]
      : memref<33xf32> to memref<32xf32, strided<[1]>>
  %dst = memref.subview %storage[1] [32] [1]
      : memref<33xf32> to memref<32xf32, strided<[1], offset: 1>>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%src : memref<32xf32, strided<[1]>>)
      outs(%dst : memref<32xf32, strided<[1], offset: 1>>) {
    ^bb0(%v: f32, %unused: f32):
      %vs = arith.mulf %v, %scale : f32
      linalg.yield %vs : f32
  }
  return %storage : memref<33xf32>
}

// Reductions are deliberately outside the pass's contract, even if their
// scalar body contains only multiply.
// CHECK-LABEL: func.func @reduction
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @reduction(%x: memref<16xf32>) -> memref<f32> {
  %out = memref.alloc() : memref<f32>
  linalg.generic {indexing_maps = [#id, #scalar], iterator_types = ["reduction"]}
      ins(%x : memref<16xf32>) outs(%out : memref<f32>) {
    ^bb0(%v: f32, %previous: f32):
      %vs = arith.mulf %v, %previous : f32
      linalg.yield %vs : f32
  }
  return %out : memref<f32>
}
