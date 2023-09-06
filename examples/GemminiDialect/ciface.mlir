// RUN: buddy-opt %s --llvm-request-c-wrappers \
// RUN:    --convert-linalg-to-gemmini --convert-linalg-to-loops \
// RUN:    --convert-func-to-llvm | \ 
// RUN: FileCheck %s

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul1
func.func @gemmini_matmul1(%arg0 : memref<32x32xi8>, %arg1 : memref<32x32xi8>, %arg2 : memref<32x32xi8>, %arg3 : memref<32x32xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<32x32xi8> memref<32x32xi8> memref<32x32xi8> memref<32x32xi32>
  return
}

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul2
func.func @gemmini_matmul2(%arg0 : memref<64x64xi8>, %arg1 : memref<64x64xi8>, %arg2 : memref<64x64xi8>, %arg3 : memref<64x64xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<64x64xi8> memref<64x64xi8> memref<64x64xi8> memref<64x64xi32>
  return
}

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul5
func.func @gemmini_matmul5(%arg0 : memref<512x512xi8>, %arg1 : memref<512x512xi8>, %arg2 : memref<512x512xi8>, %arg3 : memref<512x512xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<512x512xi8> memref<512x512xi8> memref<512x512xi8> memref<512x512xi32>
  return
}

// CHECk: llvm.func @_mlir_ciface_linalg_matmul1
func.func @linalg_matmul1(%arg0 : memref<32x32xi8>, %arg1 : memref<32x32xi8>, %arg2 : memref<32x32xi8>) {
  linalg.matmul
    ins(%arg0, %arg1: memref<32x32xi8>, memref<32x32xi8>) 
  outs(%arg2 : memref<32x32xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_matmul2
func.func @linalg_matmul2(%arg0 : memref<64x64xi8>, %arg1:memref<64x64xi8>, %arg2 : memref<64x64xi8>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<64x64xi8>, memref<64x64xi8>)
  outs(%arg2 : memref<64x64xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_matmul3
func.func @linalg_matmul3(%arg0 : memref<128x128xi8>, %arg1:memref<128x128xi8>, %arg2 : memref<128x128xi8>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<128x128xi8>, memref<128x128xi8>)
  outs(%arg2 : memref<128x128xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_matmul4
func.func @linalg_matmul4(%arg0 : memref<256x256xi8>, %arg1:memref<256x256xi8>, %arg2 : memref<256x256xi8>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<256x256xi8>, memref<256x256xi8>)
  outs(%arg2 : memref<256x256xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_matmul5
func.func @linalg_matmul5(%arg0 : memref<512x512xi8>, %arg1:memref<512x512xi8>, %arg2 : memref<512x512xi8>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<512x512xi8>, memref<512x512xi8>)
  outs(%arg2 : memref<512x512xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_matmul6
func.func @linalg_matmul6(%arg0 : memref<1024x1024xi8>, %arg1:memref<1024x1024xi8>, %arg2 : memref<1024x1024xi8>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<1024x1024xi8>, memref<1024x1024xi8>)
  outs(%arg2 : memref<1024x1024xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul6
func.func @gemmini_matmul6(%arg0 : memref<1024x1024xi8>, %arg1 : memref<1024x1024xi8>, %arg2 : memref<1024x1024xi8>, %arg3 : memref<1024x1024xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<1024x1024xi8> memref<1024x1024xi8> memref<1024x1024xi8> memref<1024x1024xi32>
  return
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv1
func.func @linalg_conv1(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x3x3xi8>, %arg2 : memref<1x1x254x254xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x3x3xi8>)
  outs(%arg2 : memref<1x1x254x254xi8>)
  return
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv2
func.func @linalg_conv2(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x5x5xi8>, %arg2 : memref<1x1x252x252xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x5x5xi8>)
  outs(%arg2 : memref<1x1x252x252xi8>)
  return
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv3
func.func @linalg_conv3(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x7x7xi8>, %arg2 : memref<1x1x250x250xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x7x7xi8>)
  outs(%arg2 : memref<1x1x250x250xi8>)
  return 
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv4
func.func @linalg_conv4(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x9x9xi8>, %arg2 : memref<1x1x248x248xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x9x9xi8>)
  outs(%arg2 : memref<1x1x248x248xi8>)
  return
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv5
func.func @linalg_conv5(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x11x11xi8>, %arg2 : memref<1x1x246x246xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x11x11xi8>)
  outs(%arg2 : memref<1x1x246x246xi8>)
  return
}

// CHECK: llvm.func @_mlir_ciface_linalg_conv6
func.func @linalg_conv6(%arg0 : memref<1x1x256x256xi8>, %arg1 : memref<1x1x13x13xi8>, %arg2 : memref<1x1x244x244xi8>) {
  linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : memref<1x1x256x256xi8>, memref<1x1x13x13xi8>)
  outs(%arg2 : memref<1x1x244x244xi8>)
  return
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv1
func.func @gemmini_conv1(%input: memref<1x256x256x1xi8>, %weights: memref<9x1xi8>, %bias: memref<1xi32>, %output: memref<64516x1xi8>) {
  %outdim = arith.constant 254 : i64 
  %kernelDim = arith.constant 3 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<9x1xi8> memref<1xi32> memref<64516x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv2
func.func @gemmini_conv2(%input: memref<1x256x256x1xi8>, %weights: memref<25x1xi8>, %bias: memref<1xi32>, %output: memref<63504x1xi8>) {
  %outdim = arith.constant 252 : i64 
  %kernelDim = arith.constant 5 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<25x1xi8> memref<1xi32> memref<63504x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv3
func.func @gemmini_conv3(%input: memref<1x256x256x1xi8>, %weights: memref<49x1xi8>, %bias: memref<1xi32>, %output: memref<62500x1xi8>) {
  %outdim = arith.constant 250 : i64 
  %kernelDim = arith.constant 7 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<49x1xi8> memref<1xi32> memref<62500x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv4
func.func @gemmini_conv4(%input: memref<1x256x256x1xi8>, %weights: memref<81x1xi8>, %bias: memref<1xi32>, %output: memref<61504x1xi8>) {
  %outdim = arith.constant 248 : i64 
  %kernelDim = arith.constant 9 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<81x1xi8> memref<1xi32> memref<61504x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv5
func.func @gemmini_conv5(%input: memref<1x256x256x1xi8>, %weights: memref<121x1xi8>, %bias: memref<1xi32>, %output: memref<60516x1xi8>) {
  %outdim = arith.constant 246 : i64 
  %kernelDim = arith.constant 11 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<121x1xi8> memref<1xi32> memref<60516x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_conv6
func.func @gemmini_conv6(%input: memref<1x256x256x1xi8>, %weights: memref<169x1xi8>, %bias: memref<1xi32>, %output: memref<59536x1xi8>) {
  %outdim = arith.constant 244 : i64 
  %kernelDim = arith.constant 13 : i64
  gemmini.tile_conv %input %weights %bias %output %outdim %outdim %kernelDim {stride = 1} :
  memref<1x256x256x1xi8> memref<169x1xi8> memref<1xi32> memref<59536x1xi8> i64 i64 i64
  return 
}

// CHECK: llvm.func @_mlir_ciface_gemmini_matmul3
func.func @gemmini_matmul3(%arg0 : memref<128x128xi8>, %arg1 : memref<128x128xi8>, %arg2 : memref<128x128xi8>, %arg3 : memref<128x128xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<128x128xi8> memref<128x128xi8> memref<128x128xi8> memref<128x128xi32>
  return
}

// CHECK:llvm.func @_mlir_ciface_gemmini_matmul4
func.func @gemmini_matmul4(%arg0 : memref<256x256xi8>, %arg1 : memref<256x256xi8>, %arg2 : memref<256x256xi8>, %arg3 : memref<256x256xi32>) {
  gemmini.tile_matmul %arg0 %arg1 %arg2 %arg3 : memref<256x256xi8> memref<256x256xi8> memref<256x256xi8> memref<256x256xi32>
  return
}
