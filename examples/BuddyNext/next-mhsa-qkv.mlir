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

  %42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %43 = tosa.transpose %t1, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %44 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_6 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %48 = tosa.transpose %t2, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %49 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_7 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %52 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %53 = tosa.transpose %t3, %52 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
  %54 = tosa.reshape %t0 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %55 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%54, %53 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_8 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %56 = tosa.reshape %55 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked_q = tensor.cast %46 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [24576{{(, 24576)*}}],

  %tensor_unranked_k = tensor.cast %51 : tensor<1x40x4096xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [32768{{(, 32768)*}}],

  %tensor_unranked_v = tensor.cast %56 : tensor<1x40x4096xf32> to tensor<*xf32>

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
