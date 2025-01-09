// RUN: buddy-opt %s \
// RUN: | FileCheck %s

func.func @kernel() {
  %vl = arith.constant 10 : index
  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {}

  // CHECK: {{.*}} = vir.set_vl {{.*}}
  %ret = vir.set_vl %vl : index {
    %c1 = arith.constant 1.0 : f32
    vector.yield %c1 : f32
  } -> f32
  return
}
