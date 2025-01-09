// RUN: buddy-opt %s \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>

func.func @kernel() {
  %vl = arith.constant 10 : index
  %mem = memref.get_global @gv : memref<10xf32>
  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    %c0 = arith.constant 0 : index
    %ele = vector.load %mem[%c0] : memref<10xf32>, vector<[1]xf32>
    vector.yield
  }

  // CHECK: {{.*}} = vir.set_vl {{.*}}
  %ret = vir.set_vl %vl : index {
    %c1 = arith.constant 1.0 : f32
    vector.yield %c1 : f32
  } -> f32
  return
}
