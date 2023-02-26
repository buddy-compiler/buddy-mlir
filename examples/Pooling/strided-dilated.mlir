memref.global "private" @input : memref<1x4x6x1xf32> = 
  dense<[[[[0.],  [1.],  [2.],  [3.],  [4.],  [5.]],
    [[6.], [7.],  [8.],  [9.],  [10.], [11.]],
    [[12.], [13.], [14.], [15.], [16.], [17.]],
    [[18.], [19.], [20.], [21.], [22.], [23.]]]]>
memref.global "private" @kernel : memref<2x2xf32> = dense<0.0>
memref.global "private" @output : memref<1x2x2x1xf32> = dense<0.0>

func.func @main() {
  // Input.
  %arg0 = memref.get_global @input : memref<1x4x6x1xf32>
  // Kernel.
  %arg1 = memref.get_global @kernel : memref<2x2xf32>
  // Output.
  %arg2 = memref.get_global @output : memref<1x2x2x1xf32>

  // Print input.
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 0.000000e+00 : f32
  %arg0_vec = vector.transfer_read %arg0[%cst0, %cst0, %cst0, %cst0], %cst1 : memref<1x4x6x1xf32>, vector<1x4x6x1xf32>
  vector.print %arg0_vec : vector<1x4x6x1xf32>

  // Sum pooling.
  linalg.pooling_nhwc_sum
    {strides = dense<[1,3]>: tensor<2xi64>, dilations = dense<[2,2]>: tensor<2xi64>}
    ins(%arg0, %arg1: memref<1x4x6x1xf32>, memref<2x2xf32>)
    outs(%arg2: memref<1x2x2x1xf32>)

  // Print output.
  %arg2_vec = vector.transfer_read %arg2[%cst0, %cst0, %cst0, %cst0], %cst1 : memref<1x2x2x1xf32>, vector<1x2x2x1xf32>
  vector.print %arg2_vec : vector<1x2x2x1xf32>

    return
  }

