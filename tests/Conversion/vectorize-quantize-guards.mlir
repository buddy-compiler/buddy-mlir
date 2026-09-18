// RUN: buddy-opt %s --vectorize-quantize | FileCheck %s
#id = affine_map<(d0) -> (d0)>

// i1 comparison masks may be vector temporaries, but not packed memref stores.
// CHECK-LABEL: func.func @boolean_buffer
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @boolean_buffer(%x: memref<32xf32>) -> memref<32xi1> {
  %out = memref.alloc() : memref<32xi1>
  %zero = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<32xf32>) outs(%out : memref<32xi1>) {
    ^bb0(%v: f32, %unused: i1):
      %positive = arith.cmpf oge, %v, %zero : f32
      linalg.yield %positive : i1
  }
  return %out : memref<32xi1>
}

// Unknown aliasing and strides must not turn memory dependencies into SIMD.
// CHECK-LABEL: func.func @may_alias
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @may_alias(%x: memref<32xf32>, %out: memref<32xf32>) {
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<32xf32>) outs(%out : memref<32xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %a = math.absf %v : f32
      linalg.yield %a : f32
  }
  return
}
// CHECK-LABEL: func.func @dynamic_stride
// CHECK-NOT: vector.
// CHECK: linalg.generic
// CHECK: return
func.func @dynamic_stride(%x: memref<32xf32, strided<[?], offset: ?>>) -> memref<32xf32> {
  %out = memref.alloc() : memref<32xf32>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<32xf32, strided<[?], offset: ?>>) outs(%out : memref<32xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %a = math.absf %v : f32
      linalg.yield %a : f32
  }
  return %out : memref<32xf32>
}
// Floating max with no abs proof may have signed-zero order differences.
// CHECK-LABEL: func.func @unknown_sign
// CHECK-NOT: vector.
// CHECK: linalg.reduce
// CHECK: return
func.func @unknown_sign(%x: memref<32xf32>) -> memref<f32> {
  %out = memref.alloc() : memref<f32>
  %seed = arith.constant 0xFF800000 : f32
  linalg.fill ins(%seed : f32) outs(%out : memref<f32>)
  linalg.reduce ins(%x : memref<32xf32>) outs(%out : memref<f32>) dimensions = [0]
      (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  return %out : memref<f32>
}
// CHECK-LABEL: func.func @sum
// CHECK-NOT: vector.
// CHECK: linalg.reduce
// CHECK: return
func.func @sum(%x: memref<32xf32>) -> memref<f32> {
  %out = memref.alloc() : memref<f32>
  %seed = arith.constant 0.0 : f32
  linalg.fill ins(%seed : f32) outs(%out : memref<f32>)
  linalg.reduce ins(%x : memref<32xf32>) outs(%out : memref<f32>) dimensions = [0]
      (%a: f32, %b: f32) {
    %m = arith.addf %a, %b : f32
    linalg.yield %m : f32
  }
  return %out : memref<f32>
}
