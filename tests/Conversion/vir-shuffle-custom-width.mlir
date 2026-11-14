// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128 vector-width=8' \
// RUN: | FileCheck %s

// With vector-width, the scalable vectors are vector<[vector-width]xT>, of
// vector-width * rvv-vlen / 64 lanes: vector<[8]xf64> holds the 16 lanes of
// a 2x? of vl 8 at VLEN 128.
func.func @f64(%a: memref<2x8xf64>) -> f64 {
  %vl = arith.constant 8 : index
  %z = arith.constant 0.0 : f64
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf64> -> !vir.vec<2x?xf64>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf64>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf64>, f64 -> f64
    vector.yield %s : f64
  } -> f64
  return %r : f64
}

// CHECK-LABEL: func.func @f64
// CHECK: vector.scalable.extract %{{.*}}[4] : vector<[4]xf64> from vector<[8]xf64>

// Index elements.
func.func @index(%a: memref<2x8xindex>, %o: memref<2x8xindex>) -> i32 {
  %vl = arith.constant 8 : index
  %zero = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xindex> -> !vir.vec<2x?xindex>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xindex>
    vir.store %y, %o[] : !vir.vec<2x?xindex> -> memref<2x8xindex>
    vector.yield %zero : i32
  } -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @index
// CHECK: %[[X:.*]] = vector.load {{.*}} : memref<16xindex>, vector<[8]xindex>
// CHECK: vector.scalable.extract %[[X]][4] : vector<[4]xindex> from vector<[8]xindex>
