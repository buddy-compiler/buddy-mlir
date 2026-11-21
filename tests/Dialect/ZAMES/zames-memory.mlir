// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @static_views
func.func @static_views(%a: memref<4x16xi8>, %b: memref<8x16xi8>,
                        %c: memref<4x8xi32>) {
  // CHECK: zames.load {{.*}} : memref<4x16xi8> -> !zames.tile<a, 4x16, i8>
  %ta = zames.load %a : memref<4x16xi8> -> !zames.tile<a, 4x16, i8>
  %tb = zames.load %b : memref<8x16xi8> -> !zames.tile<b, 8x16, i8>
  %tc = zames.load %c : memref<4x8xi32> -> !zames.tile<c, 4x8, i32>
  // CHECK: zames.store {{.*}} : !zames.tile<a, 4x16, i8>, memref<4x16xi8>
  zames.store %ta, %a : !zames.tile<a, 4x16, i8>, memref<4x16xi8>
  zames.store %tb, %b : !zames.tile<b, 8x16, i8>, memref<8x16xi8>
  zames.store %tc, %c : !zames.tile<c, 4x8, i32>, memref<4x8xi32>
  return
}

// Transpose accepts either normal or transposed view orientation. Inherent
// properties and discardable test attributes use their separate dictionaries.
// CHECK-LABEL: func.func @transpose_views
func.func @transpose_views(%normal: memref<4x16xf16>,
                           %transposed: memref<16x4xf16>) {
  // CHECK: zames.load {{.*}}transpose = true{{.*}} : memref<16x4xf16>
  %a = zames.load %transposed <{transpose = true}> : memref<16x4xf16> -> !zames.tile<a, 4x16, fp16>
  // CHECK: zames.load {{.*}}transpose = true{{.*}}test.tag = "normal"
  %b = zames.load %normal <{transpose = true}> {test.tag = "normal"} : memref<4x16xf16> -> !zames.tile<b, 4x16, fp16>
  // CHECK: zames.store {{.*}}transpose = true{{.*}} : !zames.tile<a, 4x16, fp16>, memref<16x4xf16>
  zames.store %a, %transposed <{transpose = true}> : !zames.tile<a, 4x16, fp16>, memref<16x4xf16>
  zames.store %b, %normal <{transpose = true}> : !zames.tile<b, 4x16, fp16>, memref<4x16xf16>
  return
}

// CHECK-LABEL: func.func @dynamic_views
func.func @dynamic_views(%dynamic: memref<?x?xi8>, %static: memref<4x16xi8>,
                         %mixed: memref<?x16xi8>, %m: i64, %k: i64) {
  // A static tile may be loaded from a dynamic view with no size operands.
  %a = zames.load %dynamic : memref<?x?xi8> -> !zames.tile<a, 4x16, i8>
  // Dynamic tile sizes are explicit even if the memory view itself is static.
  // CHECK: zames.load {{.*}}[{{.*}}] : memref<4x16xi8> -> !zames.tile<a, ?x?, i8>
  %b = zames.load %static[%m, %k] : memref<4x16xi8> -> !zames.tile<a, ?x?, i8>
  %c = zames.load %mixed[%m] : memref<?x16xi8> -> !zames.tile<a, ?x16, i8>
  zames.store %a, %dynamic : !zames.tile<a, 4x16, i8>, memref<?x?xi8>
  zames.store %b, %static[%m, %k] : !zames.tile<a, ?x?, i8>, memref<4x16xi8>
  // CHECK: zames.store {{.*}}[{{.*}}] : !zames.tile<a, ?x16, i8>, memref<?x16xi8>{{$}}
  zames.store %c, %mixed[%m] : !zames.tile<a, ?x16, i8>, memref<?x16xi8>
  return
}

// CHECK-LABEL: func.func @typed_fp8_storage
func.func @typed_fp8_storage(%e4: memref<4x16xf8E4M3FN>,
                             %e5: memref<8x16xf8E5M2>) {
  // CHECK: zames.load {{.*}} : memref<4x16xf8E4M3FN> -> !zames.tile<a, 4x16, fp8e4m3>
  %a = zames.load %e4 : memref<4x16xf8E4M3FN> -> !zames.tile<a, 4x16, fp8e4m3>
  %b = zames.load %e5 : memref<8x16xf8E5M2> -> !zames.tile<b, 8x16, fp8e5m2>
  zames.store %a, %e4 : !zames.tile<a, 4x16, fp8e4m3>, memref<4x16xf8E4M3FN>
  // CHECK: zames.store {{.*}} : !zames.tile<b, 8x16, fp8e5m2>, memref<8x16xf8E5M2>
  zames.store %b, %e5 : !zames.tile<b, 8x16, fp8e5m2>, memref<8x16xf8E5M2>
  return
}

// Layout and memory space belong to the view rather than separate operands.
// CHECK-LABEL: func.func @strided_view
func.func @strided_view(%view: memref<4x16xi8, strided<[?, 1], offset: ?>, 3>) {
  %a = zames.load %view : memref<4x16xi8, strided<[?, 1], offset: ?>, 3> -> !zames.tile<a, 4x16, i8>
  zames.store %a, %view : !zames.tile<a, 4x16, i8>, memref<4x16xi8, strided<[?, 1], offset: ?>, 3>
  return
}
