// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
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
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>

func.func @kernel() {
  %t_start = call @rtclock() : () -> f64

  %cst = arith.constant dense<true> : tensor<1x40xi1>
  %cst_0 = arith.constant dense<-3.40282347E+38> : tensor<40x40xf32>
  %7 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
  %8 = "tosa.const"() <{value = dense<1> : tensor<40xi64>}> : () -> tensor<40xi64>
  %9 = tosa.add %7, %8 : (tensor<40xi64>, tensor<40xi64>) -> tensor<40xi64>
  %10 = tosa.reshape %9 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
  %11 = tensor.empty() : tensor<40x40xi1>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %10 : tensor<40xi64>, tensor<40x1xi64>) outs(%11 : tensor<40x40xi1>) {
  ^bb0(%in: i64, %in_742: i64, %out: i1):
    %4175 = arith.cmpi slt, %in, %in_742 : i64
    linalg.yield %4175 : i1
  } -> tensor<40x40xi1>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %13 = tensor.empty() : tensor<40x40xf32>
  %14 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_0 : tensor<40x40xi1>, tensor<40x40xf32>) outs(%13 : tensor<40x40xf32>) {
  ^bb0(%in: i1, %in_742: f32, %out: f32):
    %4175 = arith.select %in, %cst_1, %in_742 : f32
    linalg.yield %4175 : f32
  } -> tensor<40x40xf32>
  %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 40] [1, 1] : tensor<1x40xi1> to tensor<1x40xi1>
  %15 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi1>) -> tensor<1x1x40xi1>
  %16 = tosa.reshape %15 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi1>) -> tensor<1x1x1x40xi1>
  %extracted_slice_2 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi1> to tensor<1x1x1x40xi1>
  %17 = "tosa.const"() <{value = dense<false> : tensor<1x1x40x40xi1>}> : () -> tensor<1x1x40x40xi1>
  %18 = tosa.add %extracted_slice_2, %17 : (tensor<1x1x1x40xi1>, tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xi1>
  %19 = tosa.cast %18 : (tensor<1x1x40x40xi1>) -> tensor<1x1x40x40xf32>
  %20 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
  %21 = tosa.sub %20, %19 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %22 = tosa.cast %21 : (tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xi1>
  %cst_3 = arith.constant -3.40282347E+38 : f32
  %23 = tensor.empty() : tensor<1x1x40x40xf32>
  %24 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %21 : tensor<1x1x40x40xi1>, tensor<1x1x40x40xf32>) outs(%23 : tensor<1x1x40x40xf32>) {
  ^bb0(%in: i1, %in_742: f32, %out: f32):
    %4175 = arith.select %in, %cst_3, %in_742 : f32
    linalg.yield %4175 : f32
  } -> tensor<1x1x40x40xf32>
  %25 = tosa.reshape %14 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
  %26 = tosa.reshape %25 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %extracted_slice_4 = tensor.extract_slice %26[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %27 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
  %28 = tosa.add %extracted_slice_5, %27 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %29 = tosa.add %24, %28 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %29 : tensor<1x1x40x40xf32> to tensor<*xf32>

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 1, 40, 40] strides = [1600, 1600, 40, 1] data =

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64

  return
}

func.func @main() {

  call @kernel() : () -> ()

  return
}
