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

func.func @kernel(%t0: tensor<1x40x4096xf32>, %t1: tensor<4096x4096xf32>, %t2: tensor<4096x4096xf32>, %t3: tensor<4096x4096xf32>) {
  %t_start = call @rtclock() : () -> f64

  %42 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %43 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%42, %t1 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %44 = tosa.reshape %43 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %45 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %46 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%45, %t2 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %47 = tosa.reshape %46 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %48 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %49 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%48, %t3 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %50 = tosa.reshape %49 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked_q = tensor.cast %44 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [24576{{(, 24576)*}}],

  %tensor_unranked_k = tensor.cast %47 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [32768{{(, 32768)*}}],

  %tensor_unranked_v = tensor.cast %50 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [40960{{(, 40960)*}}],

  // Print results.
  call @printMemrefF32(%tensor_unranked_q) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%tensor_unranked_k) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%tensor_unranked_v) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  %c0 = arith.constant dense<2.0> : tensor<1x40x4096xf32>
  %c1 = arith.constant dense <3.0> : tensor<4096x4096xf32>
  %c2 = arith.constant dense <4.0> : tensor<4096x4096xf32>
  %c3 = arith.constant dense <5.0> : tensor<4096x4096xf32>

  call @kernel(%c0, %c1, %c2, %c3) : (tensor<1x40x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> ()

  return
}
