// RUN: buddy-opt %s \
// RUN:     -lower-affine -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
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
    %c16 = arith.constant 16 : index
    %src = memref.get_global @gv : memref<4x4xf32>
    
    %dst = memref.alloc() : memref<4x4xf32>
    memref.copy %src, %dst : memref<4x4xf32> to memref<4x4xf32>
    %cast = memref.cast %dst : memref<4x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    func.return 
  }
}
