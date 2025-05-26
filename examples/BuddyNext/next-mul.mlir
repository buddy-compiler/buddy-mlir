// RUN: mlir-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | mlir-opt \
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
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel_mul(%arg0 : tensor<1x40x1536xf32>, %arg1 : tensor<1x40x1xf32>) {
  %t_start = call @rtclock() : () -> f64

  // Perform element-wise multiplication with shift=0
  %result = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Cast result to unranked tensor for printing
  %tensor_unranked = tensor.cast %result : tensor<1x40x1536xf32> to tensor<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 1536] strides = [61440, 1536, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [6{{(, 6)*}}],

  // Print results
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print execution time
  vector.print %time : f64

  return
}

func.func @main() {
  // Initialize input tensors with constant values
  %c2 = arith.constant dense<2.0> : tensor<1x40x1536xf32>
  %c3 = arith.constant dense<3.0> : tensor<1x40x1xf32>

  // Call kernel function
  call @kernel_mul(%c2, %c3) : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> ()

  return
}