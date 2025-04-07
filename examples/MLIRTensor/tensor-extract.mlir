// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-linalg-to-loops \
// RUN:     -convert-scf-to-cf -finalize-memref-to-llvm \
// RUN:     -convert-func-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %c0 = arith.constant 0 : index
  %t0 = arith.constant dense<[[0., 1.], [2., 3.]]> : tensor<2x2xf32>
  %print_out0 = tensor.extract %t0[%c0, %c0] : tensor<2x2xf32>
  // CHECK: 0
  vector.print %print_out0 : f32
  %t1 = tensor.cast %t0 : tensor<2x2xf32> to tensor<?x?xf32>
  %print_out1 = tensor.extract %t1[%c0, %c0] : tensor<?x?xf32>
  // CHECK: 0
  vector.print %print_out1 : f32
  return
}
