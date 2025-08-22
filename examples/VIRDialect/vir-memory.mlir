// RUN: buddy-opt %s \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>

func.func @kernel() {
  %vl = arith.constant 10 : index
  %mem = memref.get_global @gv : memref<10xf32>

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    %c0 = arith.constant 0 : index
    // CHECK: vir.load {{.*}}
    %ele = vir.load %mem[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    // CHECK: vir.store {{.*}}
    vir.store %ele, %mem[%c0] : !vir.vec<?xf32> -> memref<10xf32>
    vector.yield
  }

  return
}
