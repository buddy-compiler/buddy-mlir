// RUN: buddy-opt %s \
// RUN: | FileCheck %s

// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector="vector-width=4" \
// RUN:     -cse \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s --check-prefix=CHECK-EXEC

memref.global "private" @gv : memref<10xf32> = dense<[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.]>
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %vl = arith.constant 5 : index
  %f1 = arith.constant 1.0 : f32
  %mem = memref.get_global @gv : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    // CHECK: vir.constant {{.*}}
    %v1 = vir.constant { value = 2.0 : f32 } : !vir.vec<?xf32>
    // CHECK: vir.broadcast {{.*}}
    %v2 = vir.broadcast %f1 : f32 -> !vir.vec<?xf32>
    // CHECK: vir.store {{.*}}
    vir.store %v1, %mem[%c0] : !vir.vec<?xf32> -> memref<10xf32>
    vir.store %v2, %mem[%c5] : !vir.vec<?xf32> -> memref<10xf32>
    vector.yield
  }

  %print_mem =  memref.cast %mem : memref<10xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
  // CHECK-EXEC: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
  // CHECK-EXEC: [2,  2,  2,  2,  2,  1,  1,  1,  1,  1]

  return
}
