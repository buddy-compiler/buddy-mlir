// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -ownership-based-buffer-deallocation \
// RUN:     -buffer-deallocation-simplification \
// RUN:     -bufferization-lower-deallocations \
// RUN:     -convert-bufferization-to-memref \
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
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @kernel(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<1x1x1536xf32>, %arg2: tensor<1536xf32>) -> tensor<1x1x1536xf32> {
  %t_start = call @rtclock() : () -> f64

  %133 = tosa.add %arg0, %arg1 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
  %134 = tensor.empty() : tensor<1x1x1536xf32>
  %c2_i32_26 = arith.constant 2 : i32
  %135 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133 : tensor<1x1x1536xf32>) outs(%134 : tensor<1x1x1536xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3551 = math.fpowi %in, %c2_i32_26 : f32, i32
    linalg.yield %3551 : f32
  } -> tensor<1x1x1536xf32>
  %136 = tosa.reduce_sum %135 {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
  %137 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
  %138 = tosa.reciprocal %137 : (tensor<1xf32>) -> tensor<1xf32>
  %139 = tosa.reshape %138 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
  %140 = tosa.mul %139, %136 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %141 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %142 = tosa.add %140, %141 : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %143 = tosa.rsqrt %142 : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %144 = tosa.mul %133, %143 : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1536xf32>
  %145 = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
  %146 = tosa.mul %145, %144 : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return %146 : tensor<1x1x1536xf32>
}

func.func @main() {

  %cst_3 = arith.constant 3.0 : f32
  %empty_0 = tensor.empty() : tensor<1x1x1536xf32>
  %c0 = linalg.fill ins(%cst_3 : f32) outs(%empty_0 : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  %cst_2 = arith.constant 2.0 : f32
  %empty_1 = tensor.empty() : tensor<1x1x1536xf32>
  %c1 = linalg.fill ins(%cst_2 : f32) outs(%empty_1 : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  %cst_4 = arith.constant 4.0 : f32
  %empty_2 = tensor.empty() : tensor<1536xf32>
  %c2 = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<1536xf32>) -> tensor<1536xf32>

  %res = call @kernel(%c0, %c1, %c2) : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1536xf32>) -> tensor<1x1x1536xf32>

  %tensor_unranked = tensor.cast %res : tensor<1x1x1536xf32> to tensor<*xf32>
  // Print results.
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
