// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64

func.func @kernel_softmax(%arg0 : tensor<1x32x40x40xf32>, %arg1 : tensor<1x1x40x40xf32>) {
  %t_start = call @rtclock() : () -> f64

  // Softmax operations
  %100 = "tosa.const"() <{value = dense<11.3137083> : tensor<1x32x40x40xf32>}> : () -> tensor<1x32x40x40xf32>
  %101 = tosa.reciprocal %100 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %102 = tosa.mul %arg0, %101 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %103 = tosa.add %102, %arg1 : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
  %104 = tosa.reduce_max %103 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
  %105 = tosa.sub %103, %104 : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>
  %106 = tosa.exp %105 : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
  %107 = tosa.reduce_sum %106 {axis = 3 : i32} : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x1xf32>
  %108 = tosa.reciprocal %107 : (tensor<1x32x40x1xf32>) -> tensor<1x32x40x1xf32>
  %109 = tosa.mul %106, %108 {shift = 0 : i8} : (tensor<1x32x40x40xf32>, tensor<1x32x40x1xf32>) -> tensor<1x32x40x40xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %109 : tensor<1x32x40x40xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return
}

func.func @main() {
  %input_tensor_1 = arith.constant dense<3.0> : tensor<1x32x40x40xf32>
  %input_tensor_2 = arith.constant dense<0.0> : tensor<1x1x40x40xf32>

  call @kernel_softmax(%input_tensor_1, %input_tensor_2) : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)