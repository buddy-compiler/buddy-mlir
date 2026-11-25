// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @static_views
func.func @static_views(%a: memref<4x16xi8>, %b: memref<8x16xi8>,
                        %c: memref<4x8xi32>) {
  // CHECK: xsai_ame.load {{.*}} : memref<4x16xi8> -> !xsai_ame.tile<a, 4x16, i8>
  %ta = xsai_ame.load %a : memref<4x16xi8> -> !xsai_ame.tile<a, 4x16, i8>
  %tb = xsai_ame.load %b : memref<8x16xi8> -> !xsai_ame.tile<b, 8x16, i8>
  %tc = xsai_ame.load %c : memref<4x8xi32> -> !xsai_ame.tile<c, 4x8, i32>
  // CHECK: xsai_ame.store {{.*}} : !xsai_ame.tile<a, 4x16, i8>, memref<4x16xi8>
  xsai_ame.store %ta, %a : !xsai_ame.tile<a, 4x16, i8>, memref<4x16xi8>
  xsai_ame.store %tb, %b : !xsai_ame.tile<b, 8x16, i8>, memref<8x16xi8>
  xsai_ame.store %tc, %c : !xsai_ame.tile<c, 4x8, i32>, memref<4x8xi32>
  return
}

// Transpose accepts either normal or transposed view orientation. Inherent
// properties and discardable test attributes use their separate dictionaries.
// CHECK-LABEL: func.func @transpose_views
func.func @transpose_views(%normal: memref<4x16xf16>,
                           %transposed: memref<16x4xf16>) {
  // CHECK: xsai_ame.load {{.*}}transpose = true{{.*}} : memref<16x4xf16>
  %a = xsai_ame.load %transposed <{transpose = true}> : memref<16x4xf16> -> !xsai_ame.tile<a, 4x16, fp16>
  // CHECK: xsai_ame.load {{.*}}transpose = true{{.*}}test.tag = "normal"
  %b = xsai_ame.load %normal <{transpose = true}> {test.tag = "normal"} : memref<4x16xf16> -> !xsai_ame.tile<b, 4x16, fp16>
  // CHECK: xsai_ame.store {{.*}}transpose = true{{.*}} : !xsai_ame.tile<a, 4x16, fp16>, memref<16x4xf16>
  xsai_ame.store %a, %transposed <{transpose = true}> : !xsai_ame.tile<a, 4x16, fp16>, memref<16x4xf16>
  xsai_ame.store %b, %normal <{transpose = true}> : !xsai_ame.tile<b, 4x16, fp16>, memref<4x16xf16>
  return
}

// CHECK-LABEL: func.func @dynamic_views
func.func @dynamic_views(%dynamic: memref<?x?xi8>, %static: memref<4x16xi8>,
                         %mixed: memref<?x16xi8>, %m: i64, %k: i64) {
  // A static tile may be loaded from a dynamic view with no size operands.
  %a = xsai_ame.load %dynamic : memref<?x?xi8> -> !xsai_ame.tile<a, 4x16, i8>
  // Dynamic tile sizes are explicit even if the memory view itself is static.
  // CHECK: xsai_ame.load {{.*}}[{{.*}}] : memref<4x16xi8> -> !xsai_ame.tile<a, ?x?, i8>
  %b = xsai_ame.load %static[%m, %k] : memref<4x16xi8> -> !xsai_ame.tile<a, ?x?, i8>
  %c = xsai_ame.load %mixed[%m] : memref<?x16xi8> -> !xsai_ame.tile<a, ?x16, i8>
  xsai_ame.store %a, %dynamic : !xsai_ame.tile<a, 4x16, i8>, memref<?x?xi8>
  xsai_ame.store %b, %static[%m, %k] : !xsai_ame.tile<a, ?x?, i8>, memref<4x16xi8>
  // CHECK: xsai_ame.store {{.*}}[{{.*}}] : !xsai_ame.tile<a, ?x16, i8>, memref<?x16xi8>{{$}}
  xsai_ame.store %c, %mixed[%m] : !xsai_ame.tile<a, ?x16, i8>, memref<?x16xi8>
  return
}

// CHECK-LABEL: func.func @typed_fp8_storage
func.func @typed_fp8_storage(%e4: memref<4x16xf8E4M3FN>,
                             %e5: memref<8x16xf8E5M2>) {
  // CHECK: xsai_ame.load {{.*}} : memref<4x16xf8E4M3FN> -> !xsai_ame.tile<a, 4x16, fp8e4m3>
  %a = xsai_ame.load %e4 : memref<4x16xf8E4M3FN> -> !xsai_ame.tile<a, 4x16, fp8e4m3>
  %b = xsai_ame.load %e5 : memref<8x16xf8E5M2> -> !xsai_ame.tile<b, 8x16, fp8e5m2>
  xsai_ame.store %a, %e4 : !xsai_ame.tile<a, 4x16, fp8e4m3>, memref<4x16xf8E4M3FN>
  // CHECK: xsai_ame.store {{.*}} : !xsai_ame.tile<b, 8x16, fp8e5m2>, memref<8x16xf8E5M2>
  xsai_ame.store %b, %e5 : !xsai_ame.tile<b, 8x16, fp8e5m2>, memref<8x16xf8E5M2>
  return
}

// Layout and memory space belong to the view rather than separate operands.
// CHECK-LABEL: func.func @strided_view
func.func @strided_view(%view: memref<4x16xi8, strided<[?, 1], offset: ?>, 3>) {
  %a = xsai_ame.load %view : memref<4x16xi8, strided<[?, 1], offset: ?>, 3> -> !xsai_ame.tile<a, 4x16, i8>
  xsai_ame.store %a, %view : !xsai_ame.tile<a, 4x16, i8>, memref<4x16xi8, strided<[?, 1], offset: ?>, 3>
  return
}
