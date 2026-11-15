// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-mattr=-zvfbfmin' \
// RUN: | FileCheck %s

// Without Zvfbfmin, the bits lowering of bf16 rewrites the ops of the
// lowered vir.set_vl regions only: the ops on scalable bf16 vectors that
// were there before stay as they are.

// CHECK-LABEL: func.func @existing_bf16
// CHECK-SAME: (%[[X:.*]]: vector<[4]xbf16>)
// CHECK-NEXT: %[[Y:.*]] = arith.extf %[[X]] : vector<[4]xbf16> to vector<[4]xf32>
// CHECK-NEXT: return %[[Y]] : vector<[4]xf32>
func.func @existing_bf16(%x: vector<[4]xbf16>) -> vector<[4]xf32> {
  %y = arith.extf %x : vector<[4]xbf16> to vector<[4]xf32>
  return %y : vector<[4]xf32>
}

// CHECK-LABEL: func.func @both
// CHECK: %[[E:.*]] = arith.extf %{{.*}} : vector<[4]xbf16> to vector<[4]xf32>
// CHECK: scf.for
// CHECK-NOT: vector<[{{[0-9]+}}]xbf16>
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> vector<[{{[0-9]+}}]xi16>
// CHECK-NOT: vector<[{{[0-9]+}}]xbf16>
// CHECK: return %[[E]]
func.func @both(%x: vector<[4]xbf16>, %a: memref<128xbf16>,
                %o: memref<128xf32>) -> vector<[4]xf32> {
  %e = arith.extf %x : vector<[4]xbf16> to vector<[4]xf32>
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %v = vir.load %a[] : memref<128xbf16> -> !vir.vec<?xbf16>
    %f = vir.extf %v : !vir.vec<?xbf16> -> !vir.vec<?xf32>
    vir.store %f, %o[] : !vir.vec<?xf32> -> memref<128xf32>
    vector.yield
  }
  return %e : vector<[4]xf32>
}
