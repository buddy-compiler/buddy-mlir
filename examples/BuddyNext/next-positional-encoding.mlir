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

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @kernel(%t0 : tensor<1x40xi64>, %t1 : tensor<64xf32>) {
  %t_start = call @rtclock() : () -> f64

  %4 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
  %5 = tosa.reshape %4 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
  %cst = arith.constant dense<-3.40282347E+38> : tensor<40x40xf32>

  %6 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
  %7 = tosa.reshape %4 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
  %8 = tensor.empty() : tensor<40x40xi1>
  %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<40xi64>, tensor<40x1xi64>) outs(%8 : tensor<40x40xi1>) {
  ^bb0(%in: i64, %in_843: i64, %out: i1):
      %3964 = arith.cmpi sgt, %in, %in_843 : i64
      linalg.yield %3964 : i1
  } -> tensor<40x40xi1>
  %10 = tosa.cast %9 : (tensor<40x40xi1>) -> tensor<40x40xf32>
  %11 = tosa.mul %cst, %10 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
  %12 = tosa.reshape %11 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
  %13 = tosa.reshape %12 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %extracted_slice = tensor.extract_slice %13[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %14 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
  %15 = tosa.add %extracted_slice_0, %14 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %16 = tosa.identity %15 : (tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
  %extracted_slice_1 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_2 = tensor.extract_slice %extracted_slice_1[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_3 = tensor.extract_slice %extracted_slice_2[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_4 = tensor.extract_slice %t0[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
  %17 = tosa.reshape %extracted_slice_4 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
  %18 = tosa.reshape %17 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi64>) -> tensor<1x1x1x40xi64>
  %extracted_slice_5 = tensor.extract_slice %18[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi64> to tensor<1x1x1x40xi64>
  %19 = tosa.cast %extracted_slice_5 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
  %20 = tosa.add %extracted_slice_3, %19 : (tensor<1x1x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x40x40xf32>
  %cst_6 = arith.constant 0.000000e+00 : f32
  %splat = tensor.splat %cst_6 : tensor<1x1x40x40xf32>
  %21 = arith.cmpf oeq, %20, %splat : tensor<1x1x40x40xf32>
  %extracted_slice_7 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_8 = tensor.extract_slice %extracted_slice_7[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_9 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %cst_10 = arith.constant -3.40282347E+38 : f32
  %22 = tensor.empty() : tensor<1x1x40x40xf32>
  %splat_11 = tensor.splat %cst_10 : tensor<1x1x40x40xf32>
  %23 = linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %splat_11, %extracted_slice_9 : tensor<1x1x40x40xi1>, tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) outs(%22 : tensor<1x1x40x40xf32>) {
  ^bb0(%in: i1, %in_843: f32, %in_844: f32, %out: f32):
      %3964 = arith.select %in, %in_843, %in_844 : f32
      linalg.yield %3964 : f32
  } -> tensor<1x1x40x40xf32>
  %extracted_slice_12 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %24 = tensor.empty() : tensor<1x1x40x40xf32>
  %25 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x1x40x40xf32>) outs(%extracted_slice_14 : tensor<1x1x40x40xf32>) {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x1x40x40xf32>
  %extracted_slice_15 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_16 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %extracted_slice_17 = tensor.extract_slice %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %inserted_slice = tensor.insert_slice %25 into %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>
  %extracted_slice_18 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %inserted_slice_19 = tensor.insert_slice %inserted_slice into %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>
  %extracted_slice_20 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
  %inserted_slice_21 = tensor.insert_slice %inserted_slice_19 into %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>

  // Generate Rotary Position Embedding (RoPE)
  // %arg3 contains the rotary embedding frequencies (64 dimensions)
  %26 = tosa.reshape %t1 {new_shape = array<i64: 1, 64>} : (tensor<64xf32>) -> tensor<1x64xf32>
  %extracted_slice_22 = tensor.extract_slice %26[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> to tensor<1x64xf32>
  %27 = tosa.reshape %extracted_slice_22 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64xf32>) -> tensor<1x64x1xf32>
  %28 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
  %29 = tosa.add %27, %28 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %extracted_slice_23 = tensor.extract_slice %5[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
  %30 = tosa.reshape %extracted_slice_23 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
  %extracted_slice_24 = tensor.extract_slice %30[0, 0, 0] [1, 1, 40] [1, 1, 1] : tensor<1x1x40xi64> to tensor<1x1x40xi64>
  %31 = tosa.cast %extracted_slice_24 : (tensor<1x1x40xi64>) -> tensor<1x1x40xf32>
  %32 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
  %33 = tosa.add %29, %32 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %34 = tosa.reshape %33 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %35 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40xf32>}> : () -> tensor<1x1x40xf32>
  %36 = tosa.add %31, %35 : (tensor<1x1x40xf32>, tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
  %37 = tosa.reshape %36 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
  %38 = tosa.matmul %34, %37 : (tensor<1x64x1xf32>, tensor<1x1x40xf32>) -> tensor<1x64x40xf32>
  %39 = tosa.reshape %38 {new_shape = array<i64: 1, 64, 40>} : (tensor<1x64x40xf32>) -> tensor<1x64x40xf32>
  %40 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %41 = tosa.transpose %39, %40 : (tensor<1x64x40xf32>, tensor<3xi32>) -> tensor<1x40x64xf32>
  %42 = tosa.reshape %41 {new_shape = array<i64: 1, 40, 1, 64>} : (tensor<1x40x64xf32>) -> tensor<1x40x1x64xf32>
  %43 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x40x2x64xf32>}> : () -> tensor<1x40x2x64xf32>
  %44 = tosa.add %42, %43 : (tensor<1x40x1x64xf32>, tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
  %45 = tosa.identity %44 : (tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
  %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 128>} : (tensor<1x40x2x64xf32>) -> tensor<1x40x128xf32>
  %47 = tosa.identity %46 : (tensor<1x40x128xf32>) -> tensor<1x40x128xf32>
  %48 = math.cos %47 : tensor<1x40x128xf32>
  %49 = math.sin %47 : tensor<1x40x128xf32>

  %cst_25 = arith.constant dense<1.000000e+00> : tensor<1xf32>
  %50 = tosa.reshape %cst_25 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
  %51 = tosa.mul %48, %50 : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>

  %cst_26 = arith.constant dense<1.000000e+00> : tensor<1xf32>
  %52 = tosa.reshape %cst_26 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
  %53 = tosa.mul %49, %52 : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %51 : tensor<1x40x128xf32> to tensor<*xf32>
  %tensor_unranked_1 = tensor.cast %53 : tensor<1x40x128xf32> to tensor<*xf32>

  // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%tensor_unranked_1) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return
}

func.func @main() {

  %c2 = arith.constant dense<2> : tensor<1x40xi64>
  %c3 = arith.constant dense<3.0> : tensor<64xf32>

  call @kernel(%c2, %c3) : (tensor<1x40xi64>, tensor<64xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
