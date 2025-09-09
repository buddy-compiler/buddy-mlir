#map = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1) -> (0, d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
module {
  func.func private @rtclock() -> f64
  func.func @kernel(%arg0: tensor<1x40x4096xf32>, %arg1: tensor<1x40x4096xf32>, %arg2: tensor<1x40x4096xf32>, %arg3: tensor<1x1x2048x128xf32>, %arg4: tensor<1x1x2048x128xf32>, %arg5: tensor<1x40xi64>) {
    %0 = call @rtclock() : () -> f64
    %1 = tosa.const_shape  {values = dense<[1, 40, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [1, 40, 32, 128] : tensor<1x40x4096xf32> into tensor<1x40x32x128xf32>
    %2 = tensor.empty() : tensor<1x32x40x128xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<1x40x32x128xf32>) outs(%2 : tensor<1x32x40x128xf32>) permutation = [0, 2, 1, 3] 
    %3 = tosa.const_shape  {values = dense<[1, 40, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %expanded_0 = tensor.expand_shape %arg1 [[0], [1], [2, 3]] output_shape [1, 40, 32, 128] : tensor<1x40x4096xf32> into tensor<1x40x32x128xf32>
    %4 = tensor.empty() : tensor<1x32x40x128xf32>
    %transposed_1 = linalg.transpose ins(%expanded_0 : tensor<1x40x32x128xf32>) outs(%4 : tensor<1x32x40x128xf32>) permutation = [0, 2, 1, 3] 
    %5 = tosa.const_shape  {values = dense<[1, 40, 32, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %expanded_2 = tensor.expand_shape %arg2 [[0], [1], [2, 3]] output_shape [1, 40, 32, 128] : tensor<1x40x4096xf32> into tensor<1x40x32x128xf32>
    %6 = tensor.empty() : tensor<1x32x40x128xf32>
    %transposed_3 = linalg.transpose ins(%expanded_2 : tensor<1x40x32x128xf32>) outs(%6 : tensor<1x32x40x128xf32>) permutation = [0, 2, 1, 3] 
    %extracted_slice = tensor.extract_slice %arg3[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %extracted_slice_4 = tensor.extract_slice %arg4[0, 0, 0, 0] [1, 1, 40, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x40x128xf32>
    %7 = tensor.empty() : tensor<1x40x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x1x40x128xf32>) outs(%7 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %9 = tensor.empty() : tensor<40x128xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x40x128xf32>) outs(%9 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %11 = tensor.empty() : tensor<1x40x128xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_4 : tensor<1x1x40x128xf32>) outs(%11 : tensor<1x40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x40x128xf32>
    %13 = tensor.empty() : tensor<40x128xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1x40x128xf32>) outs(%13 : tensor<40x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<40x128xf32>
    %15 = tensor.empty() : tensor<1x40x128xf32>
    %16 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<1x40xi64>) outs(%15 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %37 = arith.index_cast %in : i64 to index
      %38 = linalg.index 2 : index
      %extracted = tensor.extract %10[%37, %38] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %17 = tosa.const_shape  {values = dense<[1, 1, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %expanded_5 = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [1, 1, 40, 128] : tensor<1x40x128xf32> into tensor<1x1x40x128xf32>
    %18 = tensor.empty() : tensor<1x40x128xf32>
    %19 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<1x40xi64>) outs(%18 : tensor<1x40x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %37 = arith.index_cast %in : i64 to index
      %38 = linalg.index 2 : index
      %extracted = tensor.extract %14[%37, %38] : tensor<40x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x40x128xf32>
    %20 = tosa.const_shape  {values = dense<[1, 1, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %expanded_6 = tensor.expand_shape %19 [[0, 1], [2], [3]] output_shape [1, 1, 40, 128] : tensor<1x40x128xf32> into tensor<1x1x40x128xf32>
    %cst = arith.constant dense<0> : tensor<1xi8>
    %21 = tensor.empty() : tensor<1x32x40x128xf32>
    %22 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed, %expanded_5 : tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) outs(%21 : tensor<1x32x40x128xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %37 = arith.mulf %in, %in_19 : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x128xf32>
    %extracted_slice_7 = tensor.extract_slice %transposed[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_8 = tensor.extract_slice %transposed[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %23 = tensor.empty() : tensor<1x32x40x64xf32>
    %24 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_8 : tensor<1x32x40x64xf32>) outs(%23 : tensor<1x32x40x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.negf %in : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x64xf32>
    %25 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice = tensor.insert_slice %24 into %25[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_11 = tensor.insert_slice %extracted_slice_7 into %inserted_slice[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %cst_12 = arith.constant dense<0> : tensor<1xi8>
    %26 = tensor.empty() : tensor<1x32x40x128xf32>
    %27 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%inserted_slice_11, %expanded_6 : tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) outs(%26 : tensor<1x32x40x128xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %37 = arith.mulf %in, %in_19 : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x128xf32>
    %28 = tensor.empty() : tensor<1x32x40x128xf32>
    %29 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %27 : tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) outs(%28 : tensor<1x32x40x128xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %37 = arith.addf %in, %in_19 : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x128xf32>
    %30 = tensor.empty() : tensor<1x32x40x128xf32>
    %31 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1, %expanded_5 : tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>) outs(%30 : tensor<1x32x40x128xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %37 = arith.mulf %in, %in_19 : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x128xf32>
    %extracted_slice_13 = tensor.extract_slice %transposed_1[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %extracted_slice_14 = tensor.extract_slice %transposed_1[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x128xf32> to tensor<1x32x40x64xf32>
    %32 = tensor.empty() : tensor<1x32x40x64xf32>
    %33 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_14 : tensor<1x32x40x64xf32>) outs(%32 : tensor<1x32x40x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.negf %in : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x40x64xf32>
    %34 = tensor.empty() : tensor<1x32x40x128xf32>
    %inserted_slice_15 = tensor.insert_slice %33 into %34[0, 0, 0, 0] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %inserted_slice_16 = tensor.insert_slice %extracted_slice_13 into %inserted_slice_15[0, 0, 0, 64] [1, 32, 40, 64] [1, 1, 1, 1] : tensor<1x32x40x64xf32> into tensor<1x32x40x128xf32>
    %35 = call @rtclock() : () -> f64
    %36 = arith.subf %35, %0 : f64
    %cast = tensor.cast %inserted_slice_16 : tensor<1x32x40x128xf32> to tensor<*xf32>
    %cast_17 = tensor.cast %29 : tensor<1x32x40x128xf32> to tensor<*xf32>
    %cast_18 = tensor.cast %31 : tensor<1x32x40x128xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    call @printMemrefF32(%cast_17) : (tensor<*xf32>) -> ()
    call @printMemrefF32(%cast_18) : (tensor<*xf32>) -> ()
    vector.print %36 : f64
    return
  }
  func.func @main() {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x40x4096xf32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<1x40x4096xf32>
    %cst_1 = arith.constant dense<4.000000e+00> : tensor<1x40x4096xf32>
    %cst_2 = arith.constant dense<5.000000e+00> : tensor<1x1x2048x128xf32>
    %cst_3 = arith.constant dense<6.000000e+00> : tensor<1x1x2048x128xf32>
    %cst_4 = arith.constant dense<7> : tensor<1x40xi64>
    call @kernel(%cst, %cst_0, %cst_1, %cst_2, %cst_3, %cst_4) : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>, tensor<1x40x4096xf32>, tensor<1x1x2048x128xf32>, tensor<1x1x2048x128xf32>, tensor<1x40xi64>) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>)
}

