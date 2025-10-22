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

memref.global "private" @gv : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %vl = arith.constant 10 : index
  %mem = memref.get_global @gv : memref<10xf32>
  %c0 = arith.constant 0 : index
  %f2 = arith.constant 2.0 : f32

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    // CHECK: vir.constant {{.*}}
    %v1 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
    // CHECK: vir.broadcast {{.*}}
    %v2 = vir.broadcast %f2 : f32 -> !vir.vec<?xf32>
    // CHECK: vir.load {{.*}}
    %ele = vir.load %mem[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    // CHECK: vir.fma {{.*}}
    %res = vir.fma %ele, %v2, %v1 : !vir.vec<?xf32>
    // CHECK: vir.store {{.*}}
    vir.store %res, %mem[%c0] : !vir.vec<?xf32> -> memref<10xf32>
    vector.yield
  }

  %print_mem =  memref.cast %mem : memref<10xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
  // CHECK-EXEC: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
  // CHECK-EXEC: [0,  2,  4,  6,  8,  10,  12,  14,  16,  18]

  return
}
