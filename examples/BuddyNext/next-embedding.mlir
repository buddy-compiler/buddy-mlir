// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -ownership-based-buffer-deallocation \
// RUN:     -buffer-deallocation-simplification \
// RUN:     -bufferization-lower-deallocations \
// RUN:     -matmul-parallel-vectorization-optimize \
// RUN:     -batchmatmul-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -func-bufferize-dynamic-offset \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64

func.func @kernel(%arg0: tensor<151936x1536xf32>, %arg1: tensor<1x1024xi64>) {
  %t_start = call @rtclock() : () -> f64

    %0 = tosa.cast %arg1 : (tensor<1x1024xi64>) -> tensor<1x1024xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 151936, 1536>} : (tensor<151936x1536xf32>) -> tensor<1x151936x1536xf32>
    %2 = tosa.gather %1, %0 : (tensor<1x151936x1536xf32>, tensor<1x1024xi32>) -> tensor<1x1024x1536xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1x1024x1536xf32>) -> tensor<1x1024x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %3 : tensor<1x1024x1536xf32> to tensor<*xf32>

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return
}

func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<151936x1536xf32>
  %c1 = arith.constant dense <1> : tensor<1x1024xi64>

  call @kernel(%c0, %c1) : (tensor<151936x1536xf32>, tensor<1x1024xi64>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
