// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -convert-tensor-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(tensor<*xf32>)

func.func @test(%a : tensor<1x40x32x128xf32>, %b : tensor<32x40x40xf32>) -> (tensor<1x40x32x128xf32>) {
    %t_start = call @rtclock() : () -> f64
    %0 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = tosa.transpose %a, %0 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>
    %2 = tosa.reshape %1 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
    %3 = tosa.matmul %b, %2 : (tensor<32x40x40xf32>, tensor<32x40x128xf32>) -> tensor<32x40x128xf32>
    %4 = tosa.reshape %3 {new_shape = array<i64: 1, 32, 40, 128>} : (tensor<32x40x128xf32>) -> tensor<1x32x40x128xf32>
    %5 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %6 = tosa.transpose %4, %5 : (tensor<1x32x40x128xf32>, tensor<4xi32>) -> tensor<1x40x32x128xf32>
    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    // Print timings.
    vector.print %time : f64
    return %6 : tensor<1x40x32x128xf32>
  }

func.func @main(){

  %v2 = arith.constant dense<2.0> : tensor<32x40x40xf32>
  %v3 = arith.constant dense<3.0> : tensor<1x40x32x128xf32>
  // %m0 = tensor.cast %v2 : tensor<32x40x40xf32> to tensor<?x?x?xf32>
  // %m1 = tensor.cast %v3 : tensor<40x32x128xf32> to tensor<?x?x?xf32>

  %m2 = call @test(%v3, %v2) : (tensor<1x40x32x128xf32>, tensor<32x40x40xf32>) -> (tensor<1x40x32x128xf32>)

  %printed_m2 = tensor.cast %m2 : tensor<1x40x32x128xf32> to tensor<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 40, 32, 128] strides = [163840, 4096, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [
  // CHECK: [240{{(, 240)*}}]
  call @printMemrefF32(%printed_m2) : (tensor<*xf32>) -> ()

  return
}
