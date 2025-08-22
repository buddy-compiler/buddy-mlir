// RUN: buddy-opt %s \
// RUN: | FileCheck %s

func.func @kernel() {
  %vl = arith.constant 10 : index

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    %f1 = arith.constant 1.0 : f32
    // CHECK: vir.constant {{.*}}
    %v1 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
    // CHECK: vir.broadcast {{.*}}
    %v2 = vir.broadcast %f1 : f32 -> !vir.vec<?xf32>
    vector.yield
  }

  return
}
