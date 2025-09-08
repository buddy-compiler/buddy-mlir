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

memref.global "private" @gv1 : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>
memref.global "private" @gv2 : memref<10xf32> = dense<[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.]>
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %vl = arith.constant 9 : index
  %mem1 = memref.get_global @gv1 : memref<10xf32>
  %mem2 = memref.get_global @gv2 : memref<10xf32>
  %c0 = arith.constant 0 : index

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    // CHECK: vir.load {{.*}}
    %ele = vir.load %mem1[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    // CHECK: vir.store {{.*}}
    vir.store %ele, %mem2[%c0] : !vir.vec<?xf32> -> memref<10xf32>
    vector.yield
  }

  %print_mem =  memref.cast %mem2 : memref<10xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
  // CHECK-EXEC: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
  // CHECK-EXEC: [0,  1,  2,  3,  4,  5,  6,  7,  8,  0]

  return
}
