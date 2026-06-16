// LinalgBench: core structured ops that are not named kernels.
module {
  func.func @core_structured_ops(%a: tensor<16x32xf32>, %b: tensor<16x32xf32>,
                                 %map_init: tensor<16x32xf32>,
                                 %reduce_init: tensor<16xf32>,
                                 %broadcast_init: tensor<16x32xf32>,
                                 %elementwise_init: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %mapped = linalg.map
        ins(%a, %b : tensor<16x32xf32>, tensor<16x32xf32>)
        outs(%map_init : tensor<16x32xf32>)
        (%x: f32, %y: f32) {
          %sum = arith.addf %x, %y : f32
          linalg.yield %sum : f32
        }
    %reduced = linalg.reduce
        ins(%mapped : tensor<16x32xf32>)
        outs(%reduce_init : tensor<16xf32>)
        dimensions = [1]
        (%x: f32, %acc: f32) {
          %sum = arith.addf %x, %acc : f32
          linalg.yield %sum : f32
        }
    %broadcast = linalg.broadcast
        ins(%reduced : tensor<16xf32>)
        outs(%broadcast_init : tensor<16x32xf32>)
        dimensions = [1]
    %result = linalg.elementwise
        kind=#linalg.elementwise_kind<add>
        ins(%mapped, %broadcast : tensor<16x32xf32>, tensor<16x32xf32>)
        outs(%elementwise_init : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %result : tensor<16x32xf32>
  }
}
