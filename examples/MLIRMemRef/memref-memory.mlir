// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0., 1., 2., 3.],
                                                         [4., 5., 6., 7.],
                                                         [8., 9., 10., 12.],
                                                         [13., 14., 15., 16.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %mem0 = memref.get_global @gv : memref<4x4xf32>
    %mem1 = memref.cast %mem0 : memref<4x4xf32> to memref<?x?xf32>
    %ele = memref.load %mem1[%c0, %c1] : memref<?x?xf32>
    vector.print %ele : f32

    memref.store %ele, %mem1[%c3, %c1] : memref<?x?xf32>
    %print_mem =  memref.cast %mem1 : memref<?x?xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [0,   1,   2,   3],
    // CHECK-NEXT: [4,   5,   6,   7],
    // CHECK-NEXT: [8,   9,   10,   12],
    // CHECK-NEXT: [13,   1,   15,   16]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()
    return
  }
}
