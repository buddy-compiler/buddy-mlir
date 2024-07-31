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

func.func @kernel_matmul(%arg0 : tensor<40x4096xf32>, %arg1 : tensor<4096x4096xf32>, %arg2 : tensor<40x4096xf32>) {
  %t_start = call @rtclock() : () -> f64

  %matmul_result = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%arg2 : tensor<40x4096xf32>) -> tensor<40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %matmul_result : tensor<40x4096xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return
}

func.func @main() {
  %input_tensor_1 = arith.constant dense<3.0> : tensor<40x4096xf32>
  %input_tensor_2 = arith.constant dense<2.0> : tensor<4096x4096xf32>
  %output_tensor = arith.constant dense<0.0> : tensor<40x4096xf32>

  call @kernel_matmul(%input_tensor_1, %input_tensor_2, %output_tensor) : (tensor<40x4096xf32>, tensor<4096x4096xf32>, tensor<40x4096xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)