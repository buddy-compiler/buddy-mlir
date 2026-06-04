// LinalgBench: linalg.contract.
module {
  func.func @contract(%a: tensor<4x8x16xf32>, %b: tensor<4x16x8xf32>,
                      %init: tensor<4x8x8xf32>) -> tensor<4x8x8xf32> {
    %0 = linalg.contract
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ]
        ins(%a, %b : tensor<4x8x16xf32>, tensor<4x16x8xf32>)
        outs(%init : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
    return %0 : tensor<4x8x8xf32>
  }
}
