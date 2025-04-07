// RUN: buddy-opt %s \
// RUN:     -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm \
// RUN:		  -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:		  -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  memref.global "private" @gv0 : memref<2x5xf32> = dense<[[0., 1., 2., 3. ,4.],
                                                          [5., 6., 7., 8. ,9.]]>

  memref.global "private" @gv1 : memref<5x2xf32> = dense<[[0., 0.],
                                                          [0., 0.],
                                                          [0., 0.],
                                                          [0., 0.],
                                                          [0., 0.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %mem0 = memref.get_global @gv0 : memref<2x5xf32>
    %mem1 = memref.get_global @gv1 : memref<5x2xf32>
    %c0 = arith.constant 1 : index
    affine.parallel (%i, %j) = (0, 0) to (2, 5) {
      %0 = affine.load %mem0[%i, %j] : memref<2x5xf32>
      affine.store %0, %mem1[%j, %i] : memref<5x2xf32>
    }
    %print_output0 = memref.cast %mem1 : memref<5x2xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [5, 2] strides = [2, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [0, 5],
    // CHECK-NEXT:  [1, 6],
    // CHECK-NEXT:  [2, 7],
    // CHECK-NEXT:  [3, 8],
    // CHECK-NEXT:  [4, 9]
    // CHECK-SAME: ]
    func.call @printMemrefF32(%print_output0) : (memref<*xf32>) -> ()
    func.return
    }
}
