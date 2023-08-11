func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x58x58x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>)
      outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    return %0 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}
