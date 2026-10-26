// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true vector-width=16' \
// RUN: | FileCheck %s

// With vector-width and no rvv-vlen, the scalable target is not RVV (e.g.
// SVE): its register groups are not limited to LMUL 8. vector<[16]xf64>
// holds the 16 lanes of a 2x? of vl 8.
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
// CHECK: llvm.call_intrinsic "llvm.vector.splice.left"({{.*}}) : (vector<[16]xf64>, vector<[16]xf64>, i32) -> vector<[16]xf64>

// The intrinsic takes LLVM types: index elements are cast to integers of
// the index width around it.
func.func @index(%a: memref<2x2xindex>, %o: memref<2x2xindex>) -> i32 {
  %vl = arith.constant 2 : index
  %zero = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x2xindex> -> !vir.vec<2x?xindex>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xindex>
    vir.store %y, %o[] : !vir.vec<2x?xindex> -> memref<2x2xindex>
    vector.yield %zero : i32
  } -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @index
// CHECK: %[[X:.*]] = vector.maskedload {{.*}} into vector<[16]xindex>
// CHECK: %[[I:.*]] = arith.index_cast %[[X]] : vector<[16]xindex> to vector<[16]xi64>
// CHECK: %[[S:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[I]], %{{.*}}, %{{.*}}) : (vector<[16]xi64>, vector<[16]xi64>, i32) -> vector<[16]xi64>
// CHECK: arith.index_cast %[[S]] : vector<[16]xi64> to vector<[16]xindex>
