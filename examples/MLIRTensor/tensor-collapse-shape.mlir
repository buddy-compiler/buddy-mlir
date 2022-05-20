func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %t0 = arith.constant dense<[[[0., 1.], [2., 3.], [4., 5.]],
                             [[6., 7.], [8., 9.], [10., 11.]]]>
                             : tensor<2x3x2xf32>
    
  %t1 = tensor.collapse_shape %t0 [[0, 1], [2]] 
  : tensor<2x3x2xf32> into tensor<6x2xf32>   
  %print_out0 = tensor.cast %t1 : tensor<6x2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out0) : (tensor<*xf32>) -> ()

  %t2 = tensor.collapse_shape %t0 [[0], [1, 2]] 
  : tensor<2x3x2xf32> into tensor<2x6xf32>
  %print_out1 = tensor.cast %t2 : tensor<2x6xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out1) : (tensor<*xf32>) -> ()

  %t3 = tensor.cast %t0 : tensor<2x3x2xf32> to tensor<?x?x?xf32>
  %t4 = tensor.collapse_shape %t3 [[0], [1, 2]] 
  :tensor<?x?x?xf32> into tensor<?x?xf32>
  %print_out2 = tensor.cast %t4 : tensor<?x?xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out2) : (tensor<*xf32>) -> ()
  %t5 = tensor.collapse_shape %t0 [[0, 1, 2]]
  :tensor<2x3x2xf32> into tensor<12xf32>
  %print_out3 = tensor.cast %t5 : tensor<12xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out3) : (tensor<*xf32>) -> ()
  %t6 = arith.constant dense<[[[1.]]]> : tensor<1x1x1xf32>
  %t7 = tensor.collapse_shape %t6 []
  :tensor<1x1x1xf32> into tensor<f32>
  %rank = tensor.rank %t7 : tensor<f32>
  vector.print %rank : index
  %print_out4 = tensor.cast %t7 : tensor<f32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out4) : (tensor<*xf32>) -> ()
  func.return 

}

