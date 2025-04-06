// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith),func.func(lower-affine))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -lower-affine \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>

func.func @kernel(%arg0 : tensor<1x40x4096xf32>, %arg1 : tensor<1x40x4096xf32>, %arg2 : tensor<1x40x4096xf32>, %arg3 : tensor<1x1x2048x128xf32>, %arg4 : tensor<1x1x2048x128xf32>, %arg5 : tensor<1x40xi64>) {
  %t_start = call @rtclock() : () -> f64

  %57 = tosa.reshape %arg0 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %58 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %59 = tosa.transpose %57, %58 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  %60 = tosa.reshape %arg1 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %61 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %62 = tosa.transpose %60, %61 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  %63 = tosa.reshape %arg2 {new_shape = array<i64: 1, 40, 32, 128>} : (tensor<1x40x4096xf32>) -> tensor<1x40x32x128xf32>
  %64 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %65 = tosa.transpose %63, %64 : (tensor<1x40x32x128xf32>, tensor<4xi32>) -> tensor<1x32x40x128xf32>

  %extracted_slice_9 = tensor.extract_slice %arg3[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_10 = tensor.extract_slice %extracted_slice_9[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_11 = tensor.extract_slice %extracted_slice_10[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
  %extracted_slice_12 = tensor.extract_slice %arg4[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
  %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
  %66 = tensor.empty() : tensor<1x40x128xf32>
  %67 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_11 : tensor<1x1x40x128xf32>) outs(%66 : tensor<1x40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x40x128xf32>
  %68 = tensor.empty() : tensor<40x128xf32>
  %69 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<1x40x128xf32>) outs(%68 : tensor<40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<40x128xf32>
  %70 = tensor.empty() : tensor<1x40x128xf32>
  %71 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x1x40x128xf32>) outs(%70 : tensor<1x40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x40x128xf32>
  %72 = tensor.empty() : tensor<40x128xf32>
  %73 = linalg.generic {indexing_maps = [#map7, #map3], iterator_types = ["parallel", "parallel"]} ins(%71 : tensor<1x40x128xf32>) outs(%72 : tensor<40x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<40x128xf32>
  // precompute_theta_pos_frequencies function, which is used to calculating special values ​​of RoPE according to: https://hyper.ai/wiki/29220
  %74 = tensor.empty() : tensor<1x40x128xf32>
  %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<1x40xi64>) outs(%74 : tensor<1x40x128xf32>) {
  ^bb0(%in: i64, %out: f32):
    %4175 = arith.index_cast %in : i64 to index
    %4176 = linalg.index 2 : index
    %extracted = tensor.extract %69[%4175, %4176] : tensor<40x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x40x128xf32>
  %76 = tosa.reshape %75 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
  %77 = tensor.empty() : tensor<1x40x128xf32>
  %78 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<1x40xi64>) outs(%77 : tensor<1x40x128xf32>) {
  ^bb0(%in: i64, %out: f32):
    %4175 = arith.index_cast %in : i64 to index
    %4176 = linalg.index 2 : index
    %extracted = tensor.extract %73[%4175, %4176] : tensor<40x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x40x128xf32>
  %79 = tosa.reshape %78 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
  %80 = tosa.mul %59, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %extracted_slice_15 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %extracted_slice_16 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %81 = tosa.negate %extracted_slice_16 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
  %82 = tensor.empty() : tensor<1x32x40x128xf32>
  %inserted_slice = tensor.insert_slice %81 into %82[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %inserted_slice_17 = tensor.insert_slice %extracted_slice_15 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %83 = tosa.mul %inserted_slice_17, %79 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %84 = tosa.add %80, %83 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %85 = tosa.mul %62, %76 {shift = 0 : i8} : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x32x40x128xf32>
  %extracted_slice_18 = tensor.extract_slice %62[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %extracted_slice_19 = tensor.extract_slice %62[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
  %86 = tosa.negate %extracted_slice_19 : (tensor<1x32x40x64xf32>) -> tensor<1x32x40x64xf32>
  %87 = tensor.empty() : tensor<1x32x40x128xf32>
  %inserted_slice_20 = tensor.insert_slice %86 into %87[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
  %inserted_slice_21 = tensor.insert_slice %extracted_slice_18 into %inserted_slice_20[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %inserted_slice_21 : tensor<1x32x40x128xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [
  // CHECK-SAME: [-3{{(, [-]?3)*}}],

  %tensor_unranked_1 = tensor.cast %84 : tensor<1x32x40x128xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [
  // CHECK-SAME: [-2{{(, -2)*}}{{(, 22)*}}],

  %tensor_unranked_2 = tensor.cast %85 : tensor<1x32x40x128xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [
  // CHECK-SAME: [15{{(, 15)*}}],

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%tensor_unranked_1) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%tensor_unranked_2) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  %c2 = arith.constant dense<2.0> : tensor<1x40x4096xf32>
  %c3 = arith.constant dense<3.0> : tensor<1x40x4096xf32>
  %c4 = arith.constant dense<4.0> : tensor<1x40x4096xf32>
  %c5 = arith.constant dense<5.0> : tensor<1x1x2048x128xf32>
  %c6 = arith.constant dense<6.0> : tensor<1x1x2048x128xf32>
  %c7 = arith.constant dense<7> : tensor<1x40xi64>

  call @kernel(%c2, %c3, %c4, %c5, %c6, %c7) : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>, tensor<1x40x4096xf32>, tensor<1x1x2048x128xf32>, tensor<1x1x2048x128xf32>, tensor<1x40xi64>) -> ()

  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
