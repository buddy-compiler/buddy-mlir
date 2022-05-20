func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %t0 = arith.constant dense<[[[0., 1., 2.], 
                              [3., 4., 5.]],
                             [[6., 7., 8.], 
                              [9., 10., 11.]]]> : tensor<2x2x3xf32>
  %c0 = arith.constant 2 : index
  %print_out0 = tensor.cast %t0 : tensor<2x2x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out0) : (tensor<*xf32>) -> ()    
  %t1 =  tensor.extract_slice %t0[0, 0, 0][1, 2, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x2xf32>
  %print_out1 = tensor.cast %t1 : tensor<1x2x2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out1) : (tensor<*xf32>) -> ()
  %t2 =  tensor.extract_slice %t0[0, 0, 0][1, 1, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<1x1x2xf32>
  %print_out2 = tensor.cast %t2 : tensor<1x1x2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out2) : (tensor<*xf32>) -> ()
  // Drop unit dimensions.
  %t3 =  tensor.extract_slice %t0[0, 0, 0][1, 2, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<2x2xf32>
  %print_out3 = tensor.cast %t3 : tensor<2x2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out3) : (tensor<*xf32>) -> ()
  %t4 =  tensor.extract_slice %t0[0, 0, 0][1, 1, 2][1, 1, 1] : tensor<2x2x3xf32> to tensor<2xf32>
  %print_out4 = tensor.cast %t4 : tensor<2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out4) : (tensor<*xf32>) -> ()
  %t5 =  tensor.extract_slice %t0[0, 0, 0][1, 1, %c0][1, 1, 1] : tensor<2x2x3xf32> to tensor<?xf32>
  %print_out5 = tensor.cast %t5 : tensor<?xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out5) : (tensor<*xf32>) -> ()
  func.return

}
