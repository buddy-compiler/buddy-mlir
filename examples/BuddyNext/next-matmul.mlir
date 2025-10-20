// RUN: mlir-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel(%lhs: tensor<40x3072xf32>, %rhs: tensor<3072x1536xf32>) {
  %t_start = call @rtclock() : () -> f64
  
  // linalg.matmul 
  %output_init = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
  %result = linalg.matmul {cast = #linalg.type_fn<cast_signed>} 
    ins(%lhs, %rhs : tensor<40x3072xf32>, tensor<3072x1536xf32>) 
    outs(%output_init : tensor<40x1536xf32>) -> tensor<40x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %result : tensor<40x1536xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  vector.print %time : f64

  return
}

func.func @main() {

  %c0 = arith.constant dense<2.0> : tensor<40x3072xf32>
  %c1 = arith.constant dense<3.0> : tensor<3072x1536xf32>

  call @kernel(%c0, %c1) : (tensor<40x3072xf32>, tensor<3072x1536xf32>) -> ()

  return
}