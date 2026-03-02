// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s

// Test Conv2D NHWC layout to IME lowering
// Input:  [1, 12, 12, 16] - batch=1, H=12, W=12, IC=16
// Filter: [3, 3, 16, 8]  - FH=3, FW=3, IC=16, OC=8
// Output: [1, 10, 10, 8] - batch=1, OH=10, OW=10, OC=8

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf
func.func @conv2d_nhwc_hwcf(%input: memref<1x12x12x16xi8>, 
                            %filter: memref<3x3x16x8xi8>,
                            %output: memref<1x10x10x8xi32>) {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: ime.vmadot
  linalg.conv_2d_nhwc_hwcf ins(%input, %filter : memref<1x12x12x16xi8>, memref<3x3x16x8xi8>)
                           outs(%output : memref<1x10x10x8xi32>)
  return
}

// Test smaller convolution with exact tile sizes
// Input:  [1, 6, 6, 8] - batch=1, H=6, W=6, IC=8
// Filter: [3, 3, 8, 4]  - FH=3, FW=3, IC=8, OC=4
// Output: [1, 4, 4, 4] - batch=1, OH=4, OW=4, OC=4

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_small
func.func @conv2d_nhwc_hwcf_small(%input: memref<1x6x6x8xi8>, 
                                  %filter: memref<3x3x8x4xi8>,
                                  %output: memref<1x4x4x4xi32>) {
  // CHECK: scf.for
  // CHECK: ime.vmadot
  linalg.conv_2d_nhwc_hwcf ins(%input, %filter : memref<1x6x6x8xi8>, memref<3x3x8x4xi8>)
                           outs(%output : memref<1x4x4x4xi32>)
  return
}

// Test Conv2D NCHW layout to IME lowering
// Input:  [1, 16, 12, 12] - batch=1, IC=16, H=12, W=12
// Filter: [8, 16, 3, 3]  - OC=8, IC=16, FH=3, FW=3
// Output: [1, 8, 10, 10] - batch=1, OC=8, OH=10, OW=10

// CHECK-LABEL: func.func @conv2d_nchw_fchw
func.func @conv2d_nchw_fchw(%input: memref<1x16x12x12xi8>, 
                            %filter: memref<8x16x3x3xi8>,
                            %output: memref<1x8x10x10xi32>) {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: ime.vmadot
  linalg.conv_2d_nchw_fchw ins(%input, %filter : memref<1x16x12x12xi8>, memref<8x16x3x3xi8>)
                           outs(%output : memref<1x8x10x10xi32>)
  return
}

// Test with stride > 1
// Input:  [1, 14, 14, 16] - batch=1, H=14, W=14, IC=16
// Filter: [3, 3, 16, 8]  - FH=3, FW=3, IC=16, OC=8
// Output: [1, 6, 6, 8]   - with stride=2: OH=(14-3)/2+1=6

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_stride2
func.func @conv2d_nhwc_hwcf_stride2(%input: memref<1x14x14x16xi8>, 
                                    %filter: memref<3x3x16x8xi8>,
                                    %output: memref<1x6x6x8xi32>) {
  // CHECK: scf.for
  // CHECK: ime.vmadot2
  linalg.conv_2d_nhwc_hwcf {strides = dense<2> : tensor<2xi64>}
                           ins(%input, %filter : memref<1x14x14x16xi8>, memref<3x3x16x8xi8>)
                           outs(%output : memref<1x6x6x8xi32>)
  return
}

// Test with stride=3
// Input:  [1, 16, 16, 8] - batch=1, H=16, W=16, IC=8
// Filter: [3, 3, 8, 4]  - FH=3, FW=3, IC=8, OC=4
// Output: [1, 5, 5, 4]   - with stride=3: OH=(16-3)/3+1=5

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_stride3
func.func @conv2d_nhwc_hwcf_stride3(%input: memref<1x16x16x8xi8>, 
                                    %filter: memref<3x3x8x4xi8>,
                                    %output: memref<1x5x5x4xi32>) {
  // CHECK: scf.for
  // CHECK: ime.vmadot3
  linalg.conv_2d_nhwc_hwcf {strides = dense<3> : tensor<2xi64>}
                           ins(%input, %filter : memref<1x16x16x8xi8>, memref<3x3x8x4xi8>)
                           outs(%output : memref<1x5x5x4xi32>)
  return
}

// Test with stride=4 (uses vmadotn with dynamic slide)
// Input:  [1, 20, 20, 8] - batch=1, H=20, W=20, IC=8
// Filter: [3, 3, 8, 4]  - FH=3, FW=3, IC=8, OC=4
// Output: [1, 5, 5, 4]   - with stride=4: OH=(20-3)/4+1=5

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_stride4
func.func @conv2d_nhwc_hwcf_stride4(%input: memref<1x20x20x8xi8>, 
                                    %filter: memref<3x3x8x4xi8>,
                                    %output: memref<1x5x5x4xi32>) {
  // CHECK: scf.for
  // CHECK: ime.vmadotn
  linalg.conv_2d_nhwc_hwcf {strides = dense<4> : tensor<2xi64>}
                           ins(%input, %filter : memref<1x20x20x8xi8>, memref<3x3x8x4xi8>)
                           outs(%output : memref<1x5x5x4xi32>)
  return
}
