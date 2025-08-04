// RUN: buddy-opt %s \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
module {
  memref.global "private" @gv : memref<5x5xf32> = dense<[[0. , 1. , 2. , 3. , 4.],
                                                         [10., 11., 12., 13. , 14.],
                                                         [20., 21., 22., 23. , 24.],
                                                         [30., 31., 32., 33. , 34.],
                                                         [40., 41., 42., 43. , 44.]]>
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %mem = memref.get_global @gv : memref<5x5xf32>

    %offset = arith.constant 0 : index
    %sub_size = arith.constant 3 : index
    %stride = arith.constant 2 : index

    %result = memref.subview %mem[%offset, %offset] [%sub_size, %sub_size] [%stride, %stride]
        : memref<5x5xf32> to memref<?x?xf32, #map0>

    %result_reduce = memref.subview %mem[0, 0][1, 3][1, 1]: memref<5x5xf32> to memref<3xf32, strided<[1]>>
    %print_output = memref.cast %result : memref<?x?xf32, #map0> to memref<*xf32>
    // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [10, 2] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [0,   2,   4],
    // CHECK-NEXT: [20,   22,   24],
    // CHECK-NEXT: [40,   42,   44]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()
    %print_output_reduce = memref.cast %result_reduce : memref<3xf32, strided<[1]>> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [3] strides = [1] data =
    // CHECK-NEXT: [0,  1,  2]
    call @printMemrefF32(%print_output_reduce) : (memref<*xf32>) -> ()
    return
  }
}
