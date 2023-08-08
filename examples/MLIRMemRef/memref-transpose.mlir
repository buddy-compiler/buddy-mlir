// RUN: buddy-opt %s \
// RUN:     -lower-affine -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s


module {
  memref.global "private" @gv : memref<3x4xf32> = dense<[[0., 1., 2., 3.],
                                                          [4., 5., 6., 7.],
                                                          [8., 9., 10., 12.]]>
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %mem = memref.get_global @gv : memref<3x4xf32>

    %new = memref.transpose %mem (i, j) -> (j, i) : memref<3x4xf32> to memref<4x3xf32, strided<[1, 4]>>
    %cast_1 = memref.cast %new : memref<4x3xf32, strided<[1, 4]>> to memref<*xf32>
    func.call @printMemrefF32(%cast_1) : (memref<*xf32>) -> ()
    func.return 
  }

}