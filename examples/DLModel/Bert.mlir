#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> ()>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map9 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map10 = affine_map<(d0, d1, d2) -> (d2)>
#map11 = affine_map<(d0, d1) -> (d1, d0)>
#map12 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map15 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map17 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map18 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "OnlyLogitsHuggingFaceModel"} {
  func.func @forward(%arg0: tensor<1x12xi64>) -> tensor<1x2xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<2x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<128x512xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<512x128xf32>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_9 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_10 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_11 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_12 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_13 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_14 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_15 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_16 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_17 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_18 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_19 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_20 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_21 = arith.constant dense_resource<__elided__> : tensor<128x512xf32>
    %cst_22 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_23 = arith.constant dense_resource<__elided__> : tensor<512x128xf32>
    %cst_24 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_25 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_26 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_27 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_28 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_29 = arith.constant dense<8.000000e+00> : tensor<f64>
    %cst_30 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_31 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_32 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_33 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_34 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_35 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_36 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_37 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_38 = arith.constant dense_resource<__elided__> : tensor<512x128xf32>
    %cst_39 = arith.constant dense_resource<__elided__> : tensor<2x128xf32>
    %cst_40 = arith.constant dense_resource<__elided__> : tensor<30522x128xf32>
    %cst_41 = arith.constant dense_resource<__elided__> : tensor<1x512xi64>
    %cst_42 = arith.constant dense<-3.4028234663852886E+38> : tensor<f64>
    %cst_43 = arith.constant dense<0> : tensor<1x512xi64>
    %cst_44 = arith.constant 1.000000e+00 : f32
    %c512 = arith.constant 512 : index
    %c30522 = arith.constant 30522 : index
    %c2 = arith.constant 2 : index
    %cst_45 = arith.constant 0.000000e+00 : f32
    %cst_46 = arith.constant 0xFF800000 : f32
    %cst_47 = arith.constant 1.41421354 : f32
    %cst_48 = arith.constant 5.000000e-01 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_49 = arith.constant 9.9999999999999998E-13 : f64
    %cst_50 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<1x12xf32>
    %extracted_slice = tensor.extract_slice %cst_43[0, 0] [1, 12] [1, 1] : tensor<1x512xi64> to tensor<1x12xi64>
    %1 = tensor.empty() : tensor<1x12xi64>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x12xi64>) outs(%1 : tensor<1x12xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x12xi64>
    %expanded = tensor.expand_shape %0 [[0], [1, 2, 3]] : tensor<1x12xf32> into tensor<1x1x1x12xf32>
    %3 = linalg.fill ins(%cst_44 : f32) outs(%expanded : tensor<1x1x1x12xf32>) -> tensor<1x1x1x12xf32>
    %4 = tensor.empty() : tensor<1x1x1x12xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x1x1x12xf32>) outs(%4 : tensor<1x1x1x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.subf %cst_44, %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x1x1x12xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_42 : tensor<1x1x1x12xf32>, tensor<f64>) outs(%4 : tensor<1x1x1x12xf32>) {
    ^bb0(%in: f32, %in_73: f64, %out: f32):
      %212 = arith.truncf %in_73 : f64 to f32
      %213 = arith.mulf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x1x1x12xf32>
    %extracted_slice_51 = tensor.extract_slice %cst_41[0, 0] [1, 12] [1, 1] : tensor<1x512xi64> to tensor<1x12xi64>
    %7 = tensor.empty() : tensor<1x12x128xf32>
    %8 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x12xi64>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %212 = arith.index_cast %in : i64 to index
      %213 = linalg.index 2 : index
      %214 = arith.cmpi slt, %212, %c30522 : index
      cf.assert %214, "index must be smaller than dim size"
      %215 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %215, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_40[%212, %213] : tensor<30522x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x128xf32>
    %9 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x12xi64>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %212 = arith.index_cast %in : i64 to index
      %213 = linalg.index 2 : index
      %214 = arith.cmpi slt, %212, %c2 : index
      cf.assert %214, "index must be smaller than dim size"
      %215 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %215, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_39[%212, %213] : tensor<2x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x128xf32>
    %10 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %11 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_51 : tensor<1x12xi64>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %212 = arith.index_cast %in : i64 to index
      %213 = linalg.index 2 : index
      %214 = arith.cmpi slt, %212, %c512 : index
      cf.assert %214, "index must be smaller than dim size"
      %215 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %215, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_38[%212, %213] : tensor<512x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x128xf32>
    %12 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %11 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %13 = tensor.empty() : tensor<1x12x1xf32>
    %14 = linalg.fill ins(%cst_45 : f32) outs(%13 : tensor<1x12x1xf32>) -> tensor<1x12x1xf32>
    %15 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %16 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %17 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %18 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %17 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %19 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %18 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %20 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %21 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %22 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.truncf %cst_49 : f64 to f32
      %213 = arith.addf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x12x1xf32>
    %23 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.rsqrt %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %24 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %25 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %24 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %26 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %cst_36 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %27 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %cst_37 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %28 = tensor.empty() : tensor<128x128xf32>
    %29 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_34 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %30 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %31 = tensor.empty() : tensor<1x128x128xf32>
    %32 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %33 = linalg.fill ins(%cst_45 : f32) outs(%7 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %34 = linalg.batch_matmul ins(%30, %32 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %35 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %cst_35 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_32 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %37 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %38 = linalg.batch_matmul ins(%30, %37 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %39 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %cst_33 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %expanded_52 = tensor.expand_shape %39 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %40 = tensor.empty() : tensor<1x2x12x64xf32>
    %41 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_52 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %42 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_30 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %43 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %44 = linalg.batch_matmul ins(%30, %43 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %45 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %cst_31 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %expanded_53 = tensor.expand_shape %45 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %46 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_53 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %expanded_54 = tensor.expand_shape %35 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %47 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_54 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %48 = tensor.empty() : tensor<1x2x64x12xf32>
    %49 = linalg.generic {indexing_maps = [#map3, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<1x2x12x64xf32>) outs(%48 : tensor<1x2x64x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x64x12xf32>
    %50 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<1x2x12x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %51 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49 : tensor<1x2x64x12xf32>) outs(%48 : tensor<1x2x64x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x64x12xf32>
    %collapsed = tensor.collapse_shape %50 [[0, 1], [2], [3]] : tensor<1x2x12x64xf32> into tensor<2x12x64xf32>
    %collapsed_55 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<1x2x64x12xf32> into tensor<2x64x12xf32>
    %52 = tensor.empty() : tensor<2x12x12xf32>
    %53 = linalg.fill ins(%cst_45 : f32) outs(%52 : tensor<2x12x12xf32>) -> tensor<2x12x12xf32>
    %54 = linalg.batch_matmul ins(%collapsed, %collapsed_55 : tensor<2x12x64xf32>, tensor<2x64x12xf32>) outs(%53 : tensor<2x12x12xf32>) -> tensor<2x12x12xf32>
    %expanded_56 = tensor.expand_shape %54 [[0, 1], [2], [3]] : tensor<2x12x12xf32> into tensor<1x2x12x12xf32>
    %55 = tensor.empty() : tensor<1x2x12x12xf32>
    %56 = linalg.generic {indexing_maps = [#map15, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_56, %cst_29 : tensor<1x2x12x12xf32>, tensor<f64>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f64, %out: f32):
      %212 = arith.truncf %in_73 : f64 to f32
      %213 = arith.divf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x2x12x12xf32>
    %57 = linalg.generic {indexing_maps = [#map15, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %6 : tensor<1x2x12x12xf32>, tensor<1x1x1x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %58 = tensor.empty() : tensor<1x2x12x1xi64>
    %59 = linalg.fill ins(%c0_i64 : i64) outs(%58 : tensor<1x2x12x1xi64>) -> tensor<1x2x12x1xi64>
    %60 = tensor.empty() : tensor<1x2x12x1xf32>
    %61 = linalg.fill ins(%cst_46 : f32) outs(%60 : tensor<1x2x12x1xf32>) -> tensor<1x2x12x1xf32>
    %62:2 = linalg.generic {indexing_maps = [#map3, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%57 : tensor<1x2x12x12xf32>) outs(%61, %59 : tensor<1x2x12x1xf32>, tensor<1x2x12x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_73: i64):
      %212 = linalg.index 3 : index
      %213 = arith.index_cast %212 : index to i64
      %214 = arith.maxf %in, %out : f32
      %215 = arith.cmpf ogt, %in, %out : f32
      %216 = arith.select %215, %213, %out_73 : i64
      linalg.yield %214, %216 : f32, i64
    } -> (tensor<1x2x12x1xf32>, tensor<1x2x12x1xi64>)
    %63 = linalg.generic {indexing_maps = [#map15, #map17, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %62#0 : tensor<1x2x12x12xf32>, tensor<1x2x12x1xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %64 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63 : tensor<1x2x12x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.exp %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %65 = linalg.fill ins(%cst_45 : f32) outs(%60 : tensor<1x2x12x1xf32>) -> tensor<1x2x12x1xf32>
    %66 = linalg.generic {indexing_maps = [#map3, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%64 : tensor<1x2x12x12xf32>) outs(%65 : tensor<1x2x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x1xf32>
    %67 = linalg.generic {indexing_maps = [#map15, #map17, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %66 : tensor<1x2x12x12xf32>, tensor<1x2x12x1xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.divf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %68 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<1x2x12x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x12xf32>
    %69 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<1x2x12x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %collapsed_57 = tensor.collapse_shape %68 [[0, 1], [2], [3]] : tensor<1x2x12x12xf32> into tensor<2x12x12xf32>
    %collapsed_58 = tensor.collapse_shape %69 [[0, 1], [2], [3]] : tensor<1x2x12x64xf32> into tensor<2x12x64xf32>
    %70 = tensor.empty() : tensor<2x12x64xf32>
    %71 = linalg.fill ins(%cst_45 : f32) outs(%70 : tensor<2x12x64xf32>) -> tensor<2x12x64xf32>
    %72 = linalg.batch_matmul ins(%collapsed_57, %collapsed_58 : tensor<2x12x12xf32>, tensor<2x12x64xf32>) outs(%71 : tensor<2x12x64xf32>) -> tensor<2x12x64xf32>
    %expanded_59 = tensor.expand_shape %72 [[0, 1], [2], [3]] : tensor<2x12x64xf32> into tensor<1x2x12x64xf32>
    %73 = tensor.empty() : tensor<1x12x2x64xf32>
    %74 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_59 : tensor<1x2x12x64xf32>) outs(%73 : tensor<1x12x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x2x64xf32>
    %collapsed_60 = tensor.collapse_shape %74 [[0], [1], [2, 3]] : tensor<1x12x2x64xf32> into tensor<1x12x128xf32>
    %75 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_27 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %76 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_60 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %77 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %78 = linalg.batch_matmul ins(%76, %77 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %79 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78, %cst_28 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %80 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %27 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %81 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %82 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %83 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %84 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %83 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %85 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %84 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %86 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %87 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %88 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.truncf %cst_49 : f64 to f32
      %213 = arith.addf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x12x1xf32>
    %89 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.rsqrt %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %90 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %91 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %90 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %92 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %cst_25 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %93 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %cst_26 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %94 = tensor.empty() : tensor<128x512xf32>
    %95 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_23 : tensor<512x128xf32>) outs(%94 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x512xf32>
    %96 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %97 = tensor.empty() : tensor<1x128x512xf32>
    %98 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<128x512xf32>) outs(%97 : tensor<1x128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x512xf32>
    %99 = tensor.empty() : tensor<1x12x512xf32>
    %100 = linalg.fill ins(%cst_45 : f32) outs(%99 : tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
    %101 = linalg.batch_matmul ins(%96, %98 : tensor<1x12x128xf32>, tensor<1x128x512xf32>) outs(%100 : tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
    %102 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %cst_24 : tensor<1x12x512xf32>, tensor<512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x512xf32>
    %103 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x12x512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_47 : f32
      %213 = math.erf %212 : f32
      %214 = arith.addf %213, %cst_44 : f32
      %215 = arith.mulf %214, %cst_48 : f32
      %216 = arith.mulf %in, %215 : f32
      linalg.yield %216 : f32
    } -> tensor<1x12x512xf32>
    %104 = tensor.empty() : tensor<512x128xf32>
    %105 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_21 : tensor<128x512xf32>) outs(%104 : tensor<512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x128xf32>
    %106 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%103 : tensor<1x12x512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x512xf32>
    %107 = tensor.empty() : tensor<1x512x128xf32>
    %108 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105 : tensor<512x128xf32>) outs(%107 : tensor<1x512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x512x128xf32>
    %109 = linalg.batch_matmul ins(%106, %108 : tensor<1x12x512xf32>, tensor<1x512x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %110 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %cst_22 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %111 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%110, %93 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %112 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %113 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %114 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %115 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%111, %114 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %116 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %115 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %117 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %118 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %119 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.truncf %cst_49 : f64 to f32
      %213 = arith.addf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x12x1xf32>
    %120 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.rsqrt %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %121 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%120 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %122 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %121 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %123 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%122, %cst_19 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %124 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %cst_20 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %125 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_17 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %126 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%124 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %127 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %128 = linalg.batch_matmul ins(%126, %127 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %129 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %cst_18 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %130 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_15 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %131 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %132 = linalg.batch_matmul ins(%126, %131 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %133 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%132, %cst_16 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %expanded_61 = tensor.expand_shape %133 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %134 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_61 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %135 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %136 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%135 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %137 = linalg.batch_matmul ins(%126, %136 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %138 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137, %cst_14 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %expanded_62 = tensor.expand_shape %138 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %139 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_62 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %expanded_63 = tensor.expand_shape %129 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x2x64xf32>
    %140 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_63 : tensor<1x12x2x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %141 = linalg.generic {indexing_maps = [#map3, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134 : tensor<1x2x12x64xf32>) outs(%48 : tensor<1x2x64x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x64x12xf32>
    %142 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140 : tensor<1x2x12x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %143 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%141 : tensor<1x2x64x12xf32>) outs(%48 : tensor<1x2x64x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x64x12xf32>
    %collapsed_64 = tensor.collapse_shape %142 [[0, 1], [2], [3]] : tensor<1x2x12x64xf32> into tensor<2x12x64xf32>
    %collapsed_65 = tensor.collapse_shape %143 [[0, 1], [2], [3]] : tensor<1x2x64x12xf32> into tensor<2x64x12xf32>
    %144 = linalg.batch_matmul ins(%collapsed_64, %collapsed_65 : tensor<2x12x64xf32>, tensor<2x64x12xf32>) outs(%53 : tensor<2x12x12xf32>) -> tensor<2x12x12xf32>
    %expanded_66 = tensor.expand_shape %144 [[0, 1], [2], [3]] : tensor<2x12x12xf32> into tensor<1x2x12x12xf32>
    %145 = linalg.generic {indexing_maps = [#map15, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_66, %cst_29 : tensor<1x2x12x12xf32>, tensor<f64>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f64, %out: f32):
      %212 = arith.truncf %in_73 : f64 to f32
      %213 = arith.divf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x2x12x12xf32>
    %146 = linalg.generic {indexing_maps = [#map15, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %6 : tensor<1x2x12x12xf32>, tensor<1x1x1x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %147:2 = linalg.generic {indexing_maps = [#map3, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%146 : tensor<1x2x12x12xf32>) outs(%61, %59 : tensor<1x2x12x1xf32>, tensor<1x2x12x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_73: i64):
      %212 = linalg.index 3 : index
      %213 = arith.index_cast %212 : index to i64
      %214 = arith.maxf %in, %out : f32
      %215 = arith.cmpf ogt, %in, %out : f32
      %216 = arith.select %215, %213, %out_73 : i64
      linalg.yield %214, %216 : f32, i64
    } -> (tensor<1x2x12x1xf32>, tensor<1x2x12x1xi64>)
    %148 = linalg.generic {indexing_maps = [#map15, #map17, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %147#0 : tensor<1x2x12x12xf32>, tensor<1x2x12x1xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %149 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148 : tensor<1x2x12x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.exp %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %150 = linalg.generic {indexing_maps = [#map3, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%149 : tensor<1x2x12x12xf32>) outs(%65 : tensor<1x2x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x1xf32>
    %151 = linalg.generic {indexing_maps = [#map15, #map17, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %150 : tensor<1x2x12x12xf32>, tensor<1x2x12x1xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.divf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2x12x12xf32>
    %152 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151 : tensor<1x2x12x12xf32>) outs(%55 : tensor<1x2x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x12xf32>
    %153 = linalg.generic {indexing_maps = [#map15, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%139 : tensor<1x2x12x64xf32>) outs(%40 : tensor<1x2x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x12x64xf32>
    %collapsed_67 = tensor.collapse_shape %152 [[0, 1], [2], [3]] : tensor<1x2x12x12xf32> into tensor<2x12x12xf32>
    %collapsed_68 = tensor.collapse_shape %153 [[0, 1], [2], [3]] : tensor<1x2x12x64xf32> into tensor<2x12x64xf32>
    %154 = linalg.batch_matmul ins(%collapsed_67, %collapsed_68 : tensor<2x12x12xf32>, tensor<2x12x64xf32>) outs(%71 : tensor<2x12x64xf32>) -> tensor<2x12x64xf32>
    %expanded_69 = tensor.expand_shape %154 [[0, 1], [2], [3]] : tensor<2x12x64xf32> into tensor<1x2x12x64xf32>
    %155 = linalg.generic {indexing_maps = [#map3, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_69 : tensor<1x2x12x64xf32>) outs(%73 : tensor<1x12x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x2x64xf32>
    %collapsed_70 = tensor.collapse_shape %155 [[0], [1], [2, 3]] : tensor<1x12x2x64xf32> into tensor<1x12x128xf32>
    %156 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_11 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %157 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_70 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %158 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156 : tensor<128x128xf32>) outs(%31 : tensor<1x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x128xf32>
    %159 = linalg.batch_matmul ins(%157, %158 : tensor<1x12x128xf32>, tensor<1x128x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %160 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159, %cst_12 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %161 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160, %124 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %162 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%161 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %163 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %164 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%163 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %165 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %164 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %166 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%165, %165 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %167 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%166 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %168 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%167 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %169 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.truncf %cst_49 : f64 to f32
      %213 = arith.addf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x12x1xf32>
    %170 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.rsqrt %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %171 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%170 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %172 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%165, %171 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %173 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %cst_9 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %174 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%173, %cst_10 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %175 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_7 : tensor<512x128xf32>) outs(%94 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x512xf32>
    %176 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174 : tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %177 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%175 : tensor<128x512xf32>) outs(%97 : tensor<1x128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x512xf32>
    %178 = linalg.batch_matmul ins(%176, %177 : tensor<1x12x128xf32>, tensor<1x128x512xf32>) outs(%100 : tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
    %179 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178, %cst_8 : tensor<1x12x512xf32>, tensor<512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x512xf32>
    %180 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%179 : tensor<1x12x512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_47 : f32
      %213 = math.erf %212 : f32
      %214 = arith.addf %213, %cst_44 : f32
      %215 = arith.mulf %214, %cst_48 : f32
      %216 = arith.mulf %in, %215 : f32
      linalg.yield %216 : f32
    } -> tensor<1x12x512xf32>
    %181 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_5 : tensor<128x512xf32>) outs(%104 : tensor<512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x128xf32>
    %182 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180 : tensor<1x12x512xf32>) outs(%99 : tensor<1x12x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x512xf32>
    %183 = linalg.generic {indexing_maps = [#map12, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181 : tensor<512x128xf32>) outs(%107 : tensor<1x512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x512x128xf32>
    %184 = linalg.batch_matmul ins(%182, %183 : tensor<1x12x512xf32>, tensor<1x512x128xf32>) outs(%33 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %185 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%184, %cst_6 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %186 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%185, %174 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %187 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%186 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %188 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %189 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %190 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%186, %189 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.subf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %191 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %190 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %192 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%191 : tensor<1x12x128xf32>) outs(%14 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.addf %in, %out : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %193 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.divf %in, %cst_50 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %194 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%193 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = arith.truncf %cst_49 : f64 to f32
      %213 = arith.addf %in, %212 : f32
      linalg.yield %213 : f32
    } -> tensor<1x12x1xf32>
    %195 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%194 : tensor<1x12x1xf32>) outs(%13 : tensor<1x12x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.rsqrt %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x1xf32>
    %196 = linalg.generic {indexing_maps = [#map9, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195 : tensor<1x12x1xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128xf32>
    %197 = linalg.generic {indexing_maps = [#map7, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %196 : tensor<1x12x128xf32>, tensor<1x12x128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %198 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197, %cst_3 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.mulf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %199 = linalg.generic {indexing_maps = [#map7, #map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198, %cst_4 : tensor<1x12x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x12x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x12x128xf32>
    %extracted_slice_71 = tensor.extract_slice %199[0, 0, 0] [1, 1, 128] [1, 1, 1] : tensor<1x12x128xf32> to tensor<1x1x128xf32>
    %collapsed_72 = tensor.collapse_shape %extracted_slice_71 [[0, 1], [2]] : tensor<1x1x128xf32> into tensor<1x128xf32>
    %200 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<128x128xf32>) outs(%28 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %201 = tensor.empty() : tensor<1x128xf32>
    %202 = linalg.fill ins(%cst_45 : f32) outs(%201 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %203 = linalg.matmul ins(%collapsed_72, %200 : tensor<1x128xf32>, tensor<128x128xf32>) outs(%202 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %204 = linalg.generic {indexing_maps = [#map, #map18, #map1], iterator_types = ["parallel", "parallel"]} ins(%203, %cst_2 : tensor<1x128xf32>, tensor<128xf32>) outs(%201 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x128xf32>
    %205 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%204 : tensor<1x128xf32>) outs(%201 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %212 = math.tanh %in : f32
      linalg.yield %212 : f32
    } -> tensor<1x128xf32>
    %206 = tensor.empty() : tensor<128x2xf32>
    %207 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<2x128xf32>) outs(%206 : tensor<128x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x2xf32>
    %208 = tensor.empty() : tensor<1x2xf32>
    %209 = linalg.fill ins(%cst_45 : f32) outs(%208 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %210 = linalg.matmul ins(%205, %207 : tensor<1x128xf32>, tensor<128x2xf32>) outs(%209 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %211 = linalg.generic {indexing_maps = [#map, #map18, #map1], iterator_types = ["parallel", "parallel"]} ins(%210, %cst_0 : tensor<1x2xf32>, tensor<2xf32>) outs(%208 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_73: f32, %out: f32):
      %212 = arith.addf %in, %in_73 : f32
      linalg.yield %212 : f32
    } -> tensor<1x2xf32>
    return %211 : tensor<1x2xf32>
  }
}
