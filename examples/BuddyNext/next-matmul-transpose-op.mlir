// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -convert-tensor-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -matmul-transpose-b-vectorization \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
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

func.func @kernel(%a : tensor<1024x1536xf32>, %b : tensor<1536x1536xf32>, %c : tensor<1024x1536xf32>) -> (tensor<1024x1536xf32>) {
  %t_start = call @rtclock() : () -> f64
  %137 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>}
    ins(%a, %b : tensor<1024x1536xf32>, tensor<1536x1536xf32>)
    outs(%c : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}
  return %137 : tensor<1024x1536xf32>
}

func.func @main(){

  %v2 = arith.constant dense<2.0> : tensor<1536x1536xf32>
  %v3 = arith.constant dense<3.0> : tensor<1024x1536xf32>
  %v4 = arith.constant dense<4.0> : tensor<1024x1536xf32>

  %m2 = call @kernel(%v3, %v2, %v4) : (tensor<1024x1536xf32>, tensor<1536x1536xf32>, tensor<1024x1536xf32>) -> (tensor<1024x1536xf32>)

  %printed_m2 = tensor.cast %m2 : tensor<1024x1536xf32> to tensor<*xf32>
  // call @printMemrefF32(%printed_m2) : (tensor<*xf32>) -> ()

  return
}
