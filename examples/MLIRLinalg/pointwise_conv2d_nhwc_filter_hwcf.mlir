// Generated from Mobilenet.mlir file
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 256)>
module  {
  // func private @print_memref_f32(memref<*xf32>)
  func private @print_memref_f32(tensor<*xf32>) -> ()

  // func @conv_2d_nhwc_hwcf(
  //   %input:  tensor<?x?x?x?xf32>,
  //   %filter: tensor<?x?x?x?xf32>,
  //   %output: tensor<?x?x?x?xf32>
  //   ) -> tensor<?x?x?x?xf32> {
  //   %res = linalg.init_tensor [1, 4, 5, 7] : tensor<1x4x5x7xf32>
  //   linalg.conv_2d_nhwc_hwcf 
  //   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  //     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  //     outs (%res: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  //   return %res : tensor<?x?x?x?xf32>
  // }

  func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %f10 = arith.constant 10.00000e+00 : f32
    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32

    %const_in2D_tensor = arith.constant dense<1.> : tensor<1x4x5x2xf32>
    %const_filter2D_tensor = arith.constant dense<2.> : tensor<1x1x2x7xf32>
    %const_out2D_tensor = arith.constant dense<0.> : tensor<1x4x5x7xf32>

    // %dynamic_const_in2D_tensor = tensor.cast %const_in2D_tensor: tensor<1x4x5x2xf32> to tensor<?x?x?x?xf32>
    // %dynamic_const_filter2D_tensor = tensor.cast %const_filter2D_tensor: tensor<1x1x2x7xf32> to tensor<?x?x?x?xf32>
    // %dynamic_const_out2D_tensor = tensor.cast %const_out2D_tensor: tensor<1x4x5x7xf32> to tensor<?x?x?x?xf32>

    
    // normal_conv2d_test
    // filter: 1,1,1,3 
    // in    : 1,2,2,1
    // out   : 1,2,2,3
    %conv = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%const_in2D_tensor, %const_filter2D_tensor : tensor<1x4x5x2xf32>, tensor<1x1x2x7xf32>)
        outs(%const_out2D_tensor : tensor<1x4x5x7xf32>) -> tensor<1x4x5x7xf32>
    
    // %dynamic_const_out2D_tensor_ = call @conv_2d_nhwc_hwcf(%dynamic_const_in2D_tensor, %dynamic_const_filter2D_tensor, %dynamic_const_out2D_tensor) : (tensor<?x?x?x?xf32>,
    //       tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    
    %out2D_nhwc_ = tensor.cast %conv : tensor<1x4x5x7xf32> to tensor<*xf32>
    call @print_memref_f32(%out2D_nhwc_): (tensor<*xf32>) -> ()
    return
  }
}
