// RUN: buddy-opt %s \
// RUN: | FileCheck %s

// RUN: buddy-opt %s \
// RUN:		  -convert-linalg-to-loops \
// RUN:     -lower-vir-to-vector \
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


func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %vl = arith.constant 134 : index
  %c1 = arith.constant 2.0 : f32
  %mem = memref.alloca() : memref<134xf32>
  linalg.fill ins(%c1 : f32) outs(%mem : memref<134xf32>)
  %c0 = arith.constant 0 : index
  %f2 = arith.constant 2.0 : f32

  // CHECK: vir.set_vl {{.*}}
  vir.set_vl %vl : index {
    // CHECK: vir.constant {{.*}}
    %v1 = vir.constant { value = 1.0 : f32 } : !vir.vec<?xf32>
    // CHECK: vir.broadcast {{.*}}
    %v2 = vir.broadcast %f2 : f32 -> !vir.vec<?xf32>
    // CHECK: vir.load {{.*}}
    %ele = vir.load %mem[%c0] : memref<134xf32> -> !vir.vec<?xf32>
    // CHECK: vir.fma {{.*}}
    %res = vir.fma %ele, %v2, %v1 : !vir.vec<?xf32>
    // CHECK: vir.store {{.*}}
    vir.store %res, %mem[%c0] : !vir.vec<?xf32> -> memref<134xf32>
    vector.yield
  }

  %print_mem =  memref.cast %mem : memref<134xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
  // CHECK-EXEC: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [134] strides = [1] data =
  // CHECK-EXEC: [5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5]
  return
}
