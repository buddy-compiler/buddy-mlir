// LinalgBench: elementwise add via linalg.generic on tensors.
module {
  func.func @elementwise_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>, %init: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      }
      ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
      outs(%init : tensor<1024xf32>) {
      ^bb0(%x: f32, %y: f32, %out: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
