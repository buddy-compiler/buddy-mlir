// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @load_rank_one(%view: memref<16xi8>) {
  // expected-error@+1 {{expected rank-2 tile view, but got rank 1}}
  %a = zames.load %view : memref<16xi8> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @store_rank_three(%a: !zames.tile<a, 4x16, i8>, %view: memref<4x4x4xi8>) {
  // expected-error@+1 {{expected rank-2 tile view, but got rank 3}}
  zames.store %a, %view : !zames.tile<a, 4x16, i8>, memref<4x4x4xi8>
  return
}

// -----

func.func @load_shape(%view: memref<4x8xi8>) {
  // expected-error@+1 {{expected tile view shape to match tile shape 4x16, but got 4x8}}
  %a = zames.load %view : memref<4x8xi8> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @transpose_requires_flag(%view: memref<16x4xi8>) {
  // expected-error@+1 {{expected tile view shape to match tile shape 4x16, but got 16x4}}
  %a = zames.load %view : memref<16x4xi8> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @bad_transpose_shape(%view: memref<8x4xi8>) {
  // expected-error@+1 {{expected tile view shape to match tile shape 4x16 or its transpose, but got 8x4}}
  %a = zames.load %view <{transpose = true}> : memref<8x4xi8> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @store_shape(%a: !zames.tile<a, 4x16, i8>, %view: memref<4x8xi8>) {
  // expected-error@+1 {{expected tile view shape to match tile shape 4x16, but got 4x8}}
  zames.store %a, %view : !zames.tile<a, 4x16, i8>, memref<4x8xi8>
  return
}

// -----

func.func @mixed_store_shape(%a: !zames.tile<a, ?x16, i8>, %view: memref<?x8xi8>, %m: i64) {
  // expected-error@+1 {{expected tile view shape to match tile shape}}
  zames.store %a, %view[%m] : !zames.tile<a, ?x16, i8>, memref<?x8xi8>
  return
}

// -----

func.func @missing_load_sizes(%view: memref<?x?xi8>) {
  // expected-error@+1 {{expected 2 dynamic size operands, but got 0}}
  %a = zames.load %view : memref<?x?xi8> -> !zames.tile<a, ?x?, i8>
  return
}

// -----

func.func @extra_load_sizes(%view: memref<4x16xi8>, %m: i64, %k: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 2}}
  %a = zames.load %view[%m, %k] : memref<4x16xi8> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @mixed_load_count(%view: memref<?x16xi8>, %m: i64, %k: i64) {
  // expected-error@+1 {{expected 1 dynamic size operands, but got 2}}
  %a = zames.load %view[%m, %k] : memref<?x16xi8> -> !zames.tile<a, ?x16, i8>
  return
}

// -----

func.func @static_store_sizes(%a: !zames.tile<a, 4x16, i8>, %view: memref<4x16xi8>, %m: i64, %k: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 2}}
  zames.store %a, %view[%m, %k] : !zames.tile<a, 4x16, i8>, memref<4x16xi8>
  return
}

// -----

func.func @dynamic_store_missing_sizes(%a: !zames.tile<a, ?x16, i8>, %view: memref<?x16xi8>) {
  // expected-error@+1 {{expected 1 dynamic size operands, but got 0}}
  zames.store %a, %view : !zames.tile<a, ?x16, i8>, memref<?x16xi8>
  return
}

// -----

func.func @store_too_many_sizes(%a: !zames.tile<a, ?x?, i8>, %view: memref<?x?xi8>, %m: i64, %n: i64, %k: i64) {
  // expected-error@+1 {{expected 2 dynamic size operands, but got 3}}
  zames.store %a, %view[%m, %n, %k] : !zames.tile<a, ?x?, i8>, memref<?x?xi8>
  return
}

// -----

func.func @load_storage_type(%view: memref<4x16xf32>) {
  // expected-error@+1 {{for ZAMES element token i8}}
  %a = zames.load %view : memref<4x16xf32> -> !zames.tile<a, 4x16, i8>
  return
}

// -----

func.func @unsigned_storage_type(%view: memref<4x16xi8>) {
  // expected-error@+1 {{for ZAMES element token u8}}
  %a = zames.load %view : memref<4x16xi8> -> !zames.tile<a, 4x16, u8>
  return
}

// -----

func.func @fp8_e4_requires_typed_storage(%view: memref<4x16xi8>) {
  // expected-error@+1 {{for ZAMES element token fp8e4m3}}
  %a = zames.load %view : memref<4x16xi8> -> !zames.tile<a, 4x16, fp8e4m3>
  return
}

// -----

func.func @fp8_e5_requires_typed_storage(%view: memref<4x16xi8>) {
  // expected-error@+1 {{for ZAMES element token fp8e5m2}}
  %a = zames.load %view : memref<4x16xi8> -> !zames.tile<a, 4x16, fp8e5m2>
  return
}

// -----

func.func @fp8_formats_are_distinct(%view: memref<4x16xf8E5M2>) {
  // expected-error@+1 {{for ZAMES element token fp8e4m3}}
  %a = zames.load %view : memref<4x16xf8E5M2> -> !zames.tile<a, 4x16, fp8e4m3>
  return
}

// -----

func.func @fp8_store_requires_typed_storage(%a: !zames.tile<a, 4x16, fp8e4m3>, %view: memref<4x16xi8>) {
  // expected-error@+1 {{for ZAMES element token fp8e4m3}}
  zames.store %a, %view : !zames.tile<a, 4x16, fp8e4m3>, memref<4x16xi8>
  return
}

// -----

func.func @dynamic_size_must_be_i64(%view: memref<?x16xi8>, %m: index) {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%m' expects different type than prior uses: 'i64' vs 'index'}}
  %a = zames.load %view[%m] : memref<?x16xi8> -> !zames.tile<a, ?x16, i8>
  return
}
