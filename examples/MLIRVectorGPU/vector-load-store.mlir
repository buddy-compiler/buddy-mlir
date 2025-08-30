// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s


module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_load(%arg0: memref<8xf32>, %arg1: memref<3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %v0 = vector.load %arg0[%c0] : memref<8xf32>, vector<3xf32>
      vector.store %v0, %arg1[%c0] : memref<3xf32>, vector<3xf32>
      gpu.return
    }
  }
  memref.global "private" @gv : memref<8xf32> = dense<[0., 1., 2., 3., 4., 5., 6., 7.]>
  func.func @main() {
    %A = memref.get_global @gv : memref<8xf32>
    %B = memref.alloc() : memref<3xf32>
    %A_cast = memref.cast %A : memref<8xf32> to memref<*xf32>
    %B_cast = memref.cast %B : memref<3xf32> to memref<*xf32>
    %c1 = arith.constant 1 : index
    gpu.host_register %A_cast : memref<*xf32>
    gpu.host_register %B_cast : memref<*xf32>
    gpu.launch_func @kernels::@vector_load blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%A : memref<8xf32>, %B : memref<3xf32>)

    call @printMemrefF32(%B_cast) : (memref<*xf32>) -> ()

    func.return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
