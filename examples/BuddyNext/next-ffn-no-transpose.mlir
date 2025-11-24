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
// RUN:     -matmul-vectorization \
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

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>

func.func @kernel(%arg0: tensor<1536x8960xf32>, %arg1: tensor<1024x1536xf32>, %arg2: tensor<1536x8960xf32>, %arg3: tensor<8960x1536xf32>, %arg4: tensor<1536xf32>, %arg5: tensor<1536x1536xf32>) -> tensor<1024x1536xf32> {
  %t_start = call @rtclock() : () -> f64

  %cst_52 = arith.constant dense<0.000000e+00> : tensor<1024x8960xf32>
  %177 = linalg.matmul ins(%arg1, %arg0 : tensor<1024x1536xf32>, tensor<1536x8960xf32>) outs(%cst_52 : tensor<1024x8960xf32>) -> tensor<1024x8960xf32>
  %179 = tosa.sigmoid %177 : (tensor<1024x8960xf32>) -> tensor<1024x8960xf32>
  %180 = tosa.mul %177, %179 : (tensor<1024x8960xf32>, tensor<1024x8960xf32>) -> tensor<1024x8960xf32>
  %181 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_53 = arith.constant dense<0.000000e+00> : tensor<1024x8960xf32>
  %184 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg1, %arg2 : tensor<1024x1536xf32>, tensor<1536x8960xf32>) outs(%cst_53 : tensor<1024x8960xf32>) -> tensor<1024x8960xf32>
  %186 = tosa.mul %180, %184 : (tensor<1024x8960xf32>, tensor<1024x8960xf32>) -> tensor<1024x8960xf32>
  %187 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_54 = arith.constant dense<0.000000e+00> : tensor<1024x1536xf32>
  %190 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%186, %arg3 : tensor<1024x8960xf32>, tensor<8960x1536xf32>) outs(%cst_54 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
  %192 = tosa.add %arg1, %190 : (tensor<1024x1536xf32>, tensor<1024x1536xf32>) -> tensor<1024x1536xf32>
  %193 = tensor.empty() : tensor<1024x1536xf32>
  %c2_i32_55 = arith.constant 2 : i32
  %194 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%192 : tensor<1024x1536xf32>) outs(%193 : tensor<1024x1536xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3964 = math.fpowi %in, %c2_i32_55 : f32, i32
    linalg.yield %3964 : f32
  } -> tensor<1024x1536xf32>
  %195 = tosa.reduce_sum %194 {axis = 1 : i32} : (tensor<1024x1536xf32>) -> tensor<1024x1xf32>
  %196 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
  %197 = tosa.reciprocal %196 : (tensor<1xf32>) -> tensor<1xf32>
  %198 = tosa.reshape %197 {new_shape = array<i64: 1, 1>} : (tensor<1xf32>) -> tensor<1x1xf32>
  %199 = tosa.mul %198, %195 : (tensor<1x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
  %200 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1024x1xf32>}> : () -> tensor<1024x1xf32>
  %201 = tosa.add %199, %200 : (tensor<1024x1xf32>, tensor<1024x1xf32>) -> tensor<1024x1xf32>
  %202 = tosa.rsqrt %201 : (tensor<1024x1xf32>) -> tensor<1024x1xf32>
  %203 = tosa.mul %192, %202 : (tensor<1024x1536xf32>, tensor<1024x1xf32>) -> tensor<1024x1536xf32>
  %204 = tosa.reshape %arg4 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
  %205 = tosa.mul %204, %203 : (tensor<1x1536xf32>, tensor<1024x1536xf32>) -> tensor<1024x1536xf32>

  %cst_55 = arith.constant dense<0.000000e+00> : tensor<1024x1536xf32>
  %211 = linalg.matmul ins(%205, %arg5 : tensor<1024x1536xf32>, tensor<1536x1536xf32>) outs(%cst_55 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return %211 : tensor<1024x1536xf32>
}

func.func @main() {

  %cst_2 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<1536x8960xf32>
  %c0 = linalg.fill ins(%cst_2 : f32) outs(%empty_0 : tensor<1536x8960xf32>) -> tensor<1536x8960xf32>

  %cst_3 = arith.constant 3.0 : f32
  %empty_1 = tensor.empty() : tensor<1024x1536xf32>
  %c1 = linalg.fill ins(%cst_3 : f32) outs(%empty_1 : tensor<1024x1536xf32>) -> tensor<1024x1536xf32>

  %cst_4 = arith.constant 4.0 : f32
  %empty_2 = tensor.empty() : tensor<1536x8960xf32>
  %c2 = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<1536x8960xf32>) -> tensor<1536x8960xf32>

  %cst_5 = arith.constant 5.0 : f32
  %empty_3 = tensor.empty() : tensor<8960x1536xf32>
  %c3 = linalg.fill ins(%cst_5 : f32) outs(%empty_3 : tensor<8960x1536xf32>) -> tensor<8960x1536xf32>

  %cst_6 = arith.constant 6.0 : f32
  %empty_4 = tensor.empty() : tensor<1536xf32>
  %c4 = linalg.fill ins(%cst_6 : f32) outs(%empty_4 : tensor<1536xf32>) -> tensor<1536xf32>

  %cst_7 = arith.constant 7.0 : f32
  %empty_5 = tensor.empty() : tensor<1536x1536xf32>
  %c5 = linalg.fill ins(%cst_7 : f32) outs(%empty_5 : tensor<1536x1536xf32>) -> tensor<1536x1536xf32>

  %res = call @kernel(%c0, %c1, %c2, %c3, %c4, %c5) : (tensor<1536x8960xf32>, tensor<1024x1536xf32>, tensor<1536x8960xf32>, tensor<8960x1536xf32>, tensor<1536xf32>, tensor<1536x1536xf32>) -> tensor<1024x1536xf32>

  %tensor_unranked = tensor.cast %res : tensor<1024x1536xf32> to tensor<*xf32>
  // Print results.
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
