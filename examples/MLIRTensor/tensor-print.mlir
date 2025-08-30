// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-loops \
// RUN:     -convert-scf-to-cf -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %init = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %tensor_unranked = tensor.cast %init : tensor<3xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [3] strides = [1] data =
  // CHECK-NEXT: [1,  2,  3]
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  return
}
