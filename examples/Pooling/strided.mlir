memref.global "private" @input : memref<1x4x6x1xf32> = 
   dense<[[[[0.],  [1.],  [2.],  [3.],  [4.],  [5.]],
      [[6.], [7.],  [8.],  [9.],  [10.], [11.]],
      [[12.], [13.], [14.], [15.], [16.], [17.]],
      [[18.], [19.], [20.], [21.], [22.], [23.]]]]>
memref.global "private" @kernel : memref<3x3xf32> = dense<0.>
memref.global "private" @output : memref<1x2x2x1xf32> = dense<0.>

func @main() {
   // input.
   %arg0 = memref.get_global @input : memref<1x4x6x1xf32>
   // kernel.
   %arg1 = memref.get_global @kernel : memref<3x3xf32>
   // output.
   %arg2 = memref.get_global @output : memref<1x2x2x1xf32>

   // print input.
   %cst1 = arith.constant 0.000000e+00 : f32
   %cst0 = arith.constant 0 : index
   %arg0_vec = vector.transfer_read %arg0[%cst0, %cst0, %cst0, %cst0], %cst1 : memref<1x4x6x1xf32>, vector<1x4x6x1xf32>
    vector.print %arg0_vec : vector<1x4x6x1xf32>

   // sum pooling.
   linalg.pooling_nhwc_sum
      {strides = dense<[1,3]>: tensor<2xi64>, dilations = dense<[1,1]>: tensor<2xi64>}
      ins(%arg0, %arg1: memref<1x4x6x1xf32>, memref<3x3xf32>)
      outs(%arg2: memref<1x2x2x1xf32>)

   // print output.
   %arg2_vec = vector.transfer_read %arg2[%cst0, %cst0, %cst0, %cst0], %cst1 : memref<1x2x2x1xf32>, vector<1x2x2x1xf32>
   vector.print %arg2_vec : vector<1x2x2x1xf32>

    return
  }

