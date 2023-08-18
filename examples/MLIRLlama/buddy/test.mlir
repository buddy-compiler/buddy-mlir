#map = affine_map<(d0, d1) -> (d0, d1)>
module{
  func.func @forward() -> tensor<1x13xf32> {
    %0 = tensor.empty() : tensor<1x13xi1>
    %1 = tensor.empty() : tensor<1x13xf32>
    %2 = arith.extui %0 : tensor<1x13xi1> to tensor<1x13xi32>
    %3 = arith.bitcast %2 : tensor<1x13xi32> to tensor<1x13xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %3 : tensor<1x13xf32>, tensor<1x13xf32>) outs(%3 : tensor<1x13xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<1x13xf32>
    %5 = math.rsqrt %4 : tensor<1x13xf32>
    return %5 : tensor<1x13xf32>
  }
}
