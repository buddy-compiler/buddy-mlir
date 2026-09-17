// RUN: buddy-opt %s -o %t.base
// RUN: buddy-opt %s --fuse-qwen-gate-up-w8a8-linear -o %t.fused
// RUN: cmp %t.base %t.fused

// A lone T=1 projection has the decode Gate/Up shape but no legal partner.
// The fusion pass must be a true no-op, including avoiding the general folds
// normally performed by the greedy rewrite driver.
module {
  func.func @single_projection(
      %xq: tensor<1x1024xi8>, %xs: tensor<1x1xf32>,
      %wq: tensor<48x1x64x1024xi8>, %ws: tensor<1x3072xf32>)
      -> (tensor<1x1x3072xf32>, i32) {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %foldable = arith.addi %c1, %c2 : i32
    %init = tensor.empty() : tensor<1x3072xf32>
    %result = "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %init)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (tensor<1x1024xi8>, tensor<1x1xf32>,
         tensor<48x1x64x1024xi8>, tensor<1x3072xf32>,
         tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %shape = tosa.const_shape {values = dense<[1, 1, 3072]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %view = tosa.reshape %result, %shape :
        (tensor<1x3072xf32>, !tosa.shape<3>) -> tensor<1x1x3072xf32>
    // Keep the otherwise foldable arithmetic live so an accidental greedy
    // invocation is observable in the byte-for-byte output comparison.
    return %view, %foldable : tensor<1x1x3072xf32>, i32
  }
}
