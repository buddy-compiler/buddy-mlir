// LinalgBench: linalg.index inside linalg.generic.
module {
  func.func @generic_index(%init: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      outs(%init : tensor<1024xf32>) {
      ^bb0(%out: f32):
        %idx = linalg.index 0 : index
        %idx_i32 = arith.index_cast %idx : index to i32
        %idx_f32 = arith.sitofp %idx_i32 : i32 to f32
        linalg.yield %idx_f32 : f32
      } -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
