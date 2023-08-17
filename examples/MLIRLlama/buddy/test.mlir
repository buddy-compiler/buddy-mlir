#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module{
  func.func @forward() -> tensor<1x16x4xf32> {
    %0 = tensor.empty() : tensor<8x16x4xf32>
    %1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] : tensor<8x16x4xf32> to tensor<1x16x4xf32>
    return %1 : tensor<1x16x4xf32>
  }
}
