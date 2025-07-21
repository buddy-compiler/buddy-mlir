// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -lower-affine \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64

func.func @kernel(%t0: tensor<32000x4096xf32>, %t1: tensor<1x40xi64>) {
  %t_start = call @rtclock() : () -> f64

  %0 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
  %2 = tosa.reshape %1 {new_shape = array<i64: 1, 40>} : (tensor<1x40xi64>) -> tensor<1x40xi64>
  %3 = tosa.cast %t1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
  %4 = tosa.reshape %t0 {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
  %5 = tosa.gather %4, %3 : (tensor<1x32000x4096xf32>, tensor<1x40xi32>) -> tensor<1x40x4096xf32>
  %6 = tosa.reshape %5 {new_shape = array<i64: 1, 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %6 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [3{{(, 3)*}}],

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<32000x4096xf32>
  %c1 = arith.constant dense <1> : tensor<1x40xi64>

  call @kernel(%c0, %c1) : (tensor<32000x4096xf32>, tensor<1x40xi64>) -> ()

  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
