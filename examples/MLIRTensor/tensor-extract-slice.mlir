// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-linalg-to-loops -convert-scf-to-cf -convert-cf-to-llvm -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %t0 = arith.constant dense<[[[0., 1., 2.],
                              [3., 4., 5.]],
                             [[6., 7., 8.],
                              [9., 10., 11.]]]> : tensor<2x2x3xf32>
  %c0 = arith.constant 2 : index
  %t1 =  tensor.extract_slice %t0[0, 0, 0][1, 2, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x2xf32>
  %print_out1 = tensor.cast %t1 : tensor<1x2x2xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 2, 2] strides = [4, 2, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [0,    1],
  // CHECK-NEXT: [3,    4]
  // CHECK-SAME: ]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out1) : (tensor<*xf32>) -> ()
  %t2 =  tensor.extract_slice %t0[0, 0, 0][1, 1, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<1x1x2xf32>
  %print_out2 = tensor.cast %t2 : tensor<1x1x2xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 1, 2] strides = [2, 2, 1] data =
  // CHECK: [
  // CHECK-SAME: [
  // CHECK-SAME: [0,    1]
  // CHECK-SAME: ]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out2) : (tensor<*xf32>) -> ()
  // Drop unit dimensions.
  %t3 =  tensor.extract_slice %t0[0, 0, 0][1, 2, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<2x2xf32>
  %print_out3 = tensor.cast %t3 : tensor<2x2xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1],
  // CHECK-NEXT: [3,   4]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out3) : (tensor<*xf32>) -> ()
  %t4 =  tensor.extract_slice %t0[0, 0, 0][1, 1, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<2xf32>
  %print_out4 = tensor.cast %t4 : tensor<2xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [0,  1]
  call @printMemrefF32(%print_out4) : (tensor<*xf32>) -> ()
  %t5 =  tensor.extract_slice %t0[0, 0, 0][1, 1, %c0][1, 1, 1] : tensor<2x2x3xf32> to tensor<?xf32>
  %print_out5 = tensor.cast %t5 : tensor<?xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [2] strides = [1] data =
  // CHECK-NEXT: [0,  1]
  call @printMemrefF32(%print_out5) : (tensor<*xf32>) -> ()
  return
}
