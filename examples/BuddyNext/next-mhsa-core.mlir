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

func.func @kernel(%arg0: tensor<1x1x1024x1024xf32>, %arg1: tensor<1x1x1024x1024xf32>, %arg2: tensor<1x1x1024x1024xf32>, %arg3: tensor<1x12x1024x128xf32>, %arg4: tensor<1x12x1024x128xf32>) -> tensor<1x1024x1536xf32> {
  %t_start = call @rtclock() : () -> f64

  %extracted_slice_43 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 1024, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xf32> to tensor<1x1x1024x1024xf32>
  %extracted_slice_44 = tensor.extract_slice %arg1[0, 0, 0, 0] [1, 1, 1024, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xf32> to tensor<1x1x1024x1024xf32>
  %extracted_slice_45 = tensor.extract_slice %arg2[0, 0, 0, 0] [1, 1, 1024, 1024] [1, 1, 1, 1] : tensor<1x1x1024x1024xf32> to tensor<1x1x1024x1024xf32>
  %cst_46 = arith.constant 0.000000e+00 : f32
  %splat_47 = tensor.splat %cst_46 : tensor<1024x1024xf32>
  %130 = tosa.reshape %extracted_slice_45 {new_shape = array<i64: 1024, 1024>} : (tensor<1x1x1024x1024xf32>) -> tensor<1024x1024xf32>
  %131 = tosa.add %splat_47, %130 : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %132 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %133 = tosa.transpose %arg3, %132 : (tensor<1x12x1024x128xf32>, tensor<4xi32>) -> tensor<1x12x128x1024xf32>
  %134 = tosa.reshape %arg3 {new_shape = array<i64: 12, 1024, 128>} : (tensor<1x12x1024x128xf32>) -> tensor<12x1024x128xf32>
  %135 = tosa.reshape %133 {new_shape = array<i64: 12, 128, 1024>} : (tensor<1x12x128x1024xf32>) -> tensor<12x128x1024xf32>
  %136 = tosa.matmul %134, %135 : (tensor<12x1024x128xf32>, tensor<12x128x1024xf32>) -> tensor<12x1024x1024xf32>
  %cst_48 = arith.constant 0.0883883461 : f32
  %splat_49 = tensor.splat %cst_48 : tensor<12x1024x1024xf32>
  %137 = tosa.mul %136, %splat_49 : (tensor<12x1024x1024xf32>, tensor<12x1024x1024xf32>) -> tensor<12x1024x1024xf32>
  %138 = tosa.reshape %131 {new_shape = array<i64: 1, 1024, 1024>} : (tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
  %139 = tosa.add %137, %138 : (tensor<12x1024x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<12x1024x1024xf32>
  %140 = tosa.reduce_max %139 {axis = 2 : i32} : (tensor<12x1024x1024xf32>) -> tensor<12x1024x1xf32>
  %141 = tosa.sub %139, %140 : (tensor<12x1024x1024xf32>, tensor<12x1024x1xf32>) -> tensor<12x1024x1024xf32>
  %142 = math.exp %141 : tensor<12x1024x1024xf32>
  %143 = tosa.reduce_sum %142 {axis = 2 : i32} : (tensor<12x1024x1024xf32>) -> tensor<12x1024x1xf32>
  %144 = tosa.log %143 : (tensor<12x1024x1xf32>) -> tensor<12x1024x1xf32>
  %145 = tosa.add %140, %144 : (tensor<12x1024x1xf32>, tensor<12x1024x1xf32>) -> tensor<12x1024x1xf32>
  %146 = tosa.sub %139, %145 : (tensor<12x1024x1024xf32>, tensor<12x1024x1xf32>) -> tensor<12x1024x1024xf32>
  %147 = math.exp %146 : tensor<12x1024x1024xf32>
  %148 = tosa.reshape %145 {new_shape = array<i64: 1, 12, 1024>} : (tensor<12x1024x1xf32>) -> tensor<1x12x1024xf32>
  %149 = tosa.reshape %arg3 {new_shape = array<i64: 12, 1024, 128>} : (tensor<1x12x1024x128xf32>) -> tensor<12x1024x128xf32>
  %150 = tosa.matmul %147, %149 : (tensor<12x1024x1024xf32>, tensor<12x1024x128xf32>) -> tensor<12x1024x128xf32>
  %151 = tosa.reshape %150 {new_shape = array<i64: 1, 12, 1024, 128>} : (tensor<12x1024x128xf32>) -> tensor<1x12x1024x128xf32>
  %152 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %153 = tosa.transpose %151, %152 : (tensor<1x12x1024x128xf32>, tensor<4xi32>) -> tensor<1x1024x12x128xf32>
  %154 = tosa.reshape %153 {new_shape = array<i64: 1, 1024, 1536>} : (tensor<1x1024x12x128xf32>) -> tensor<1x1024x1536xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return %154 : tensor<1x1024x1536xf32>
}

func.func @main() {

  %cst_2 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<1x1x1024x1024xf32>
  %c0 = linalg.fill ins(%cst_2 : f32) outs(%empty_0 : tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x1024xf32>

  %cst_3 = arith.constant 3.0 : f32
  %empty_1 = tensor.empty() : tensor<1x1x1024x1024xf32>
  %c1 = linalg.fill ins(%cst_3 : f32) outs(%empty_1 : tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x1024xf32>

  %cst_4 = arith.constant 4.0 : f32
  %empty_2 = tensor.empty() : tensor<1x1x1024x1024xf32>
  %c2 = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x1024xf32>

  %cst_5 = arith.constant 5.0 : f32
  %empty_3 = tensor.empty() : tensor<1x12x1024x128xf32>
  %c3 = linalg.fill ins(%cst_5 : f32) outs(%empty_3 : tensor<1x12x1024x128xf32>) -> tensor<1x12x1024x128xf32>

  %cst_6 = arith.constant 6.0 : f32
  %empty_4 = tensor.empty() : tensor<1x12x1024x128xf32>
  %c4 = linalg.fill ins(%cst_6 : f32) outs(%empty_4 : tensor<1x12x1024x128xf32>) -> tensor<1x12x1024x128xf32>

  %res = call @kernel(%c0, %c1, %c2, %c3, %c4) : (tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>, tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>) -> tensor<1x1024x1536xf32>

  %tensor_unranked = tensor.cast %res : tensor<1x1024x1536xf32> to tensor<*xf32>
  // Print results.
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
