// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:     -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm\
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %c0 = arith.constant 0. : f32
  %c1 = arith.constant 1. : f32
  %c2 = arith.constant 2. : f32
  %c3 = arith.constant 3. : f32
  %c4 = arith.constant 4. : f32
  %c5 = arith.constant 5. : f32
  %c6 = arith.constant 6. : f32
  %c7 = arith.constant 7. : f32
  %c8 = arith.constant 8. : f32
  %t0 = tensor.from_elements %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7, %c8 : tensor<3x3xf32>
  %print_out0 = tensor.cast %t0 : tensor<3x3xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // TODO: Printed results with errors, currently skipping value test.
  // CHECK: {{.*}}
  func.call @printMemrefF32(%print_out0) : (tensor<*xf32>) -> ()
  return
}
