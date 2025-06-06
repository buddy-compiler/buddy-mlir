// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
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
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @softmax_kernel(%input: tensor<1x40x151936xf32>) {
  %t_start = call @rtclock() : () -> f64

  %max = tosa.reduce_max %input {axis = 2 : i32} : (tensor<1x40x151936xf32>) -> tensor<1x40x1xf32>
  %sub = tosa.sub %input, %max : (tensor<1x40x151936xf32>, tensor<1x40x1xf32>) -> tensor<1x40x151936xf32>
  %exp = tosa.exp %sub : (tensor<1x40x151936xf32>) -> tensor<1x40x151936xf32>
  %sum = tosa.reduce_sum %exp {axis = 2 : i32} : (tensor<1x40x151936xf32>) -> tensor<1x40x1xf32>
  %logsum = tosa.log %sum : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
  %add = tosa.add %max, %logsum : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
  %sub2 = tosa.sub %input, %add : (tensor<1x40x151936xf32>, tensor<1x40x1xf32>) -> tensor<1x40x151936xf32>
  %softmax = tosa.exp %sub2 : (tensor<1x40x151936xf32>) -> tensor<1x40x151936xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %softmax : tensor<1x40x151936xf32> to tensor<*xf32>

  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return
}

func.func @main() {
  %c0 = arith.constant dense<2.0> : tensor<1x40x151936xf32>
  call @softmax_kernel(%c0) : (tensor<1x40x151936xf32>) -> ()
  return
}