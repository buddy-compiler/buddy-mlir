// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
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
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel(%t0: tensor<1x32x40x40xf32>, %t1: tensor<1x32x40x128xf32>) {
  %t_start = call @rtclock() : () -> f64

  %110 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
  %111 = tosa.add %t0, %110 : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %112 = tosa.reshape %111 {new_shape = array<i64: 32, 40, 40>} : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
  %113 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
  %114 = tosa.add %t1, %113 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %115 = tosa.reshape %114 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
  %116 = tosa.matmul %112, %115 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
  %117 = tosa.reshape %116 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %117 : tensor<1x32x40x128xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [
  // CHECK-SAME: [240{{(, 240)*}}],

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  %c0 = arith.constant dense<2.0> : tensor<1x32x40x40xf32>
  %c1 = arith.constant dense<3.0> : tensor<1x32x40x128xf32>

  call @kernel(%c0, %c1) : (tensor<1x32x40x40xf32>, tensor<1x32x40x128xf32>) -> ()

  return
}
