// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 vector-width=1' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 vector-width=1' \
// RUN:   -convert-vector-to-llvm | FileCheck %s --check-prefix=LLVM

// A broadcast to vector<[1]xT> (vscale lanes) is one to vector<[2]xT> and
// its first half: the LLVM lowering of vector.broadcast to vector<[1]xT>
// fills lane 0 only (its splat shufflevector of mask [0] folds away), the
// other lanes are poison.
func.func @strip_mined(%a: memref<?xf64>, %n: index, %w: f64) {
  vir.set_vl %n : index {
    %x = vir.load %a[] : memref<?xf64> -> !vir.vec<?xf64>
    %b = vir.broadcast %w : f64 -> !vir.vec<?xf64>
    %y = arith.mulf %x, %b : !vir.vec<?xf64>
    vir.store %y, %a[] : !vir.vec<?xf64> -> memref<?xf64>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @strip_mined
// CHECK: %[[W:.*]] = vector.broadcast %{{.*}} : f64 to vector<[2]xf64>
// CHECK: %[[B:.*]] = vector.scalable.extract %[[W]][0] : vector<[1]xf64> from vector<[2]xf64>
// CHECK: arith.mulf %{{.*}}, %[[B]] : vector<[1]xf64>
// LLVM-LABEL: func.func @strip_mined
// LLVM: llvm.shufflevector {{.*}} [0, 0] : vector<[2]xf64>

func.func @results(%a: memref<8xf64>, %w: f64) -> f64 {
  %vl = arith.constant 8 : index
  %z = arith.constant 0.0 : f64
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf64> -> !vir.vec<?xf64>
    %b = vir.broadcast %w : f64 -> !vir.vec<?xf64>
    %y = arith.mulf %x, %b : !vir.vec<?xf64>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<?xf64>, f64 -> f64
    vector.yield %s : f64
  } -> f64
  return %r : f64
}

// CHECK-LABEL: func.func @results
// CHECK: %[[W:.*]] = vector.broadcast %{{.*}} : f64 to vector<[2]xf64>
// CHECK: %[[B:.*]] = vector.scalable.extract %[[W]][0] : vector<[1]xf64> from vector<[2]xf64>
// CHECK: arith.mulf %{{.*}}, %[[B]] : vector<[1]xf64>
