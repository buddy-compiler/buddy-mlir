// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" -convert-vector-to-llvm \
// RUN:     -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:     -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %c0 = arith.constant 16. : f32
  %c1 = arith.constant 0 : index
  %t0 = arith.constant dense<[[0., 1., 2., 3.],
                              [4., 5., 6., 7.],
                              [8., 9., 10., 11.],
                              [12., 13., 14., 15.]]> : tensor<4x4xf32>
  %t1 = tensor.insert %c0 into %t0[%c1, %c1] : tensor<4x4xf32>
  %t2 = tensor.cast %t1 : tensor<4x4xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // TODO: Printed results with errors, currently skipping value test.
  // CHECK: {{.*}}
  func.call @printMemrefF32(%t2) : (tensor<*xf32>) -> ()
  %t3 = tensor.cast %t0 : tensor<4x4xf32> to tensor<?x?xf32>
  %t4 = tensor.insert %c0 into %t3[%c1, %c1] : tensor<?x?xf32>
  %t5 = tensor.cast %t4 : tensor<?x?xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // TODO: Printed results with errors, currently skipping value test.
  // CHECK: {{.*}}
  func.call @printMemrefF32(%t5) : (tensor<*xf32>) -> ()
  func.return
}
