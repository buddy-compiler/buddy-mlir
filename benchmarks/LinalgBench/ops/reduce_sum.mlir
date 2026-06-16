// LinalgBench: row-wise sum reduction via linalg.reduce on tensors.
module {
  func.func @reduce_sum(%input: tensor<256x256xf32>, %init: tensor<256xf32>) -> tensor<256xf32> {
    %0 = linalg.reduce
        ins(%input : tensor<256x256xf32>)
        outs(%init : tensor<256xf32>)
        dimensions = [1]
        (%x: f32, %acc: f32) {
        %sum = arith.addf %acc, %x : f32
        linalg.yield %sum : f32
      }
    return %0 : tensor<256xf32>
  }
}
