// RUN: buddy-opt %s \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:     -expand-strided-metadata -lower-affine \
// RUN:     -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm\
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %t0 = arith.constant dense<[[[0., 1.], [2., 3.], [4., 5.]],
                             [[6., 7.], [8., 9.], [10., 11.]]]>
                             : tensor<2x3x2xf32>

  %t1 = tensor.collapse_shape %t0 [[0, 1], [2]]
  : tensor<2x3x2xf32> into tensor<6x2xf32>
  %print_out0 = tensor.cast %t1 : tensor<6x2xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [6, 2] strides = [2, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1],
  // CHECK-NEXT: [2,   3],
  // CHECK-NEXT: [4,   5],
  // CHECK-NEXT: [6,   7],
  // CHECK-NEXT: [8,   9],
  // CHECK-NEXT: [10,   11]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out0) : (tensor<*xf32>) -> ()
  %t2 = tensor.collapse_shape %t0 [[0], [1, 2]]
  : tensor<2x3x2xf32> into tensor<2x6xf32>
  %print_out1 = tensor.cast %t2 : tensor<2x6xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 6] strides = [6, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3,   4,   5],
  // CHECK-NEXT: [6,   7,   8,   9,   10,   11]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out1) : (tensor<*xf32>) -> ()
  %t3 = tensor.cast %t0 : tensor<2x3x2xf32> to tensor<?x?x?xf32>
  %t4 = tensor.collapse_shape %t3 [[0], [1, 2]]
  :tensor<?x?x?xf32> into tensor<?x?xf32>
  %print_out2 = tensor.cast %t4 : tensor<?x?xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 6] strides = [6, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3,   4,   5],
  // CHECK-NEXT: [6,   7,   8,   9,   10,   11]
  // CHECK-SAME: ]
  call @printMemrefF32(%print_out2) : (tensor<*xf32>) -> ()
  %t5 = tensor.collapse_shape %t0 [[0, 1, 2]]
  :tensor<2x3x2xf32> into tensor<12xf32>
  %print_out3 = tensor.cast %t5 : tensor<12xf32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [12] strides = [1] data =
  // CHECK: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11]
  call @printMemrefF32(%print_out3) : (tensor<*xf32>) -> ()
  %t6 = arith.constant dense<[[[1.]]]> : tensor<1x1x1xf32>
  %t7 = tensor.collapse_shape %t6 [] :tensor<1x1x1xf32> into tensor<f32>
  %print_out4 = tensor.cast %t7 : tensor<f32> to tensor<*xf32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
  // CHECK: [1]
  call @printMemrefF32(%print_out4) : (tensor<*xf32>) -> ()
  return
}
