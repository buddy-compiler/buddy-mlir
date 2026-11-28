// RUN: not buddy-opt %s -lower-xsai-ame -split-input-file 2>&1 | FileCheck %s

// Historical L1 spelling remains parseable, but native lowering rejects it.
// CHECK: lower-xsai-ame does not support XSAIAME element token i4
func.func @unsupported_token() {
  %tile = xsai_ame.mzero : !xsai_ame.tile<a, 4x8, i4>
  return
}

// -----

// CHECK: lower-xsai-ame does not support load/store for tile kind a with element i32
func.func @unsupported_a_role(%view: memref<128x64xi32>) {
  %tile = xsai_ame.load %view : memref<128x64xi32> -> !xsai_ame.tile<a, 128x64, i32>
  return
}

// -----

// CHECK: lower-xsai-ame does not support load/store for tile kind c with element fp16
func.func @unsupported_c_role(%view: memref<128x128xf16>) {
  %tile = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp16>
  xsai_ame.store %tile, %view : !xsai_ame.tile<c, 128x128, fp16>, memref<128x128xf16>
  return
}

// -----

// CHECK: lower-xsai-ame does not support mma element combination i8 x i8 -> fp16
func.func @unsupported_accumulator() {
  %a = xsai_ame.mzero : !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.mzero : !xsai_ame.tile<b, 128x64, i8>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp16>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, fp16>
  return
}

// -----

// Integer inputs require an integer accumulator before explicit conversion.
// CHECK: lower-xsai-ame does not support mma element combination i8 x i8 -> fp32
func.func @unsupported_i8_fp32() {
  %a = xsai_ame.mzero : !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.mzero : !xsai_ame.tile<b, 128x64, i8>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, fp32>
  return
}

// -----

// CHECK: lower-xsai-ame does not support mma element combination fp8e4m3 x fp8e5m2 -> fp32
func.func @unsupported_mixed_input() {
  %a = xsai_ame.mzero : !xsai_ame.tile<a, 128x64, fp8e4m3>
  %b = xsai_ame.mzero : !xsai_ame.tile<b, 128x64, fp8e5m2>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, fp8e4m3>, !xsai_ame.tile<b, 128x64, fp8e5m2>, !xsai_ame.tile<c, 128x128, fp32>
  return
}

// -----

// CHECK: lower-xsai-ame requires a rank-2 memref view with static unit minor stride
func.func @nonunit_minor_load(%view: memref<128x64xi8, strided<[256, 2]>>) {
  %tile = xsai_ame.load %view : memref<128x64xi8, strided<[256, 2]>> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// -----

// CHECK: lower-xsai-ame requires a rank-2 memref view with static unit minor stride
func.func @dynamic_minor_store(%view: memref<128x128xi32, strided<[?, ?], offset: ?>>) {
  %tile = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %tile, %view : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32, strided<[?, ?], offset: ?>>
  return
}

// -----

// Native matrix intrinsics use address space zero; do not silently erase it.
// CHECK: lower-xsai-ame requires the default memory address space
func.func @nondefault_address_space(%view: memref<128x64xi8, 3>) {
  %tile = xsai_ame.load %view : memref<128x64xi8, 3> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// -----

// Symbolic spaces are legal in L1 but must diagnose rather than assert here.
// CHECK: lower-xsai-ame requires the default memory address space
func.func @symbolic_address_space(%view: memref<128x64xi8, "custom">) {
  %tile = xsai_ame.load %view : memref<128x64xi8, "custom"> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// -----

// The matrix unit computes only complete tiles; partial tiles are rejected at
// compile time instead of silently corrupting the accumulator.
// CHECK: lower-xsai-ame requires a complete a tile 128x64 for element i8, got '!xsai_ame.tile<a, 4x8, i8>'
func.func @partial_load(%view: memref<4x8xi8>) {
  %tile = xsai_ame.load %view : memref<4x8xi8> -> !xsai_ame.tile<a, 4x8, i8>
  return
}

// -----

// Oversized tiles are not complete tiles either.
// CHECK: lower-xsai-ame requires a complete a tile 128x64 for element i8, got '!xsai_ame.tile<a, 1024x64, i8>'
func.func @oversized_load(%view: memref<1024x64xi8>) {
  %tile = xsai_ame.load %view : memref<1024x64xi8> -> !xsai_ame.tile<a, 1024x64, i8>
  return
}

// -----

// K is 512 bits: 32 half-width elements, not 64.
// CHECK: lower-xsai-ame requires a complete b tile 128x32 for element fp16, got '!xsai_ame.tile<b, 128x64, fp16>'
func.func @half_k_mismatch(%view: memref<128x64xf16>) {
  %tile = xsai_ame.load %view : memref<128x64xf16> -> !xsai_ame.tile<b, 128x64, fp16>
  return
}

// -----

// Dynamic tile extents never reach the matrix unit.
// CHECK: lower-xsai-ame requires a complete a tile 128x64 for element i8, got '!xsai_ame.tile<a, ?x64, i8>'
func.func @dynamic_load(%view: memref<?x64xi8>, %m: i64) {
  %tile = xsai_ame.load %view[%m] : memref<?x64xi8> -> !xsai_ame.tile<a, ?x64, i8>
  return
}

// -----

// CHECK: lower-xsai-ame requires a complete c tile 128x128 for element fp32, got '!xsai_ame.tile<c, 128x64, fp32>'
func.func @partial_c_store(%view: memref<128x64xf32>) {
  %tile = xsai_ame.mzero : !xsai_ame.tile<c, 128x64, fp32>
  xsai_ame.store %tile, %view : !xsai_ame.tile<c, 128x64, fp32>, memref<128x64xf32>
  return
}

// -----

// CHECK: lower-xsai-ame requires a complete b tile 128x64 for element i8, got '!xsai_ame.tile<b, ?x64, i8>'
func.func @dynamic_store(%view: memref<?x?xi8>, %m: i64) {
  %tile = xsai_ame.mzero[%m] : !xsai_ame.tile<b, ?x64, i8>
  xsai_ame.store %tile, %view[%m] : !xsai_ame.tile<b, ?x64, i8>, memref<?x?xi8>
  return
}

// -----

// CHECK: lower-xsai-ame requires a complete a tile 128x64 for element i8, got '!xsai_ame.tile<a, ?x?, i8>'
func.func @dynamic_mma(%m: i64, %n: i64, %k: i64) {
  %a = xsai_ame.mzero[%m, %k] : !xsai_ame.tile<a, ?x?, i8>
  %b = xsai_ame.mzero[%n, %k] : !xsai_ame.tile<b, ?x?, i8>
  %c = xsai_ame.mzero[%m, %n] : !xsai_ame.tile<c, ?x?, i32>
  %r = xsai_ame.mma %a, %b, %c[%m, %n, %k] : !xsai_ame.tile<a, ?x?, i8>, !xsai_ame.tile<b, ?x?, i8>, !xsai_ame.tile<c, ?x?, i32>
  return
}

// -----

// CHECK: lower-xsai-ame requires a complete b tile 128x32 for element fp16, got '!xsai_ame.tile<b, ?x32, fp16>'
func.func @mixed_mma(%n: i64) {
  %a = xsai_ame.mzero : !xsai_ame.tile<a, 128x32, fp16>
  %b = xsai_ame.mzero[%n] : !xsai_ame.tile<b, ?x32, fp16>
  %c = xsai_ame.mzero[%n] : !xsai_ame.tile<c, 128x?, fp32>
  %r = xsai_ame.mma %a, %b, %c[%n] : !xsai_ame.tile<a, 128x32, fp16>, !xsai_ame.tile<b, ?x32, fp16>, !xsai_ame.tile<c, 128x?, fp32>
  return
}

// -----

// The native target has no calling convention for tiles or sync handles.
// CHECK: failed to legalize operation 'func.func'
func.func private @tile_argument(!xsai_ame.tile<c, 128x128, i32>)

// -----

// Producers carry tiles through SCF; a cf block argument is not converted.
// CHECK: failed to legalize operation 'func.func'
func.func @tile_block_argument() {
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, i32>
  cf.br ^next(%c : !xsai_ame.tile<c, 128x128, i32>)
^next(%t: !xsai_ame.tile<c, 128x128, i32>):
  return
}
