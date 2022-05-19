func.func private @printMemrefF32(%ptr : tensor<*xf32>) 

func.func @main() {
  %t0 = arith.constant dense<[[[0., 1., 2.],
                               [3., 4., 5.],
                               [6., 7., 8.]],
                              [[9., 10., 11.],
                               [12., 13., 14.],
                               [15., 16., 17.]],
                              [[18., 19., 20.],
                               [21., 22., 23.],
                               [24., 25., 26.]]]> : tensor<3x3x3xf32>
  %t1 = arith.constant dense<[[27., 28., 29.],[30., 31., 32.],[33., 34., 35.]]> : tensor<3x3xf32>
  %t2 = tensor.insert_slice %t1 into %t0[0, 0, 0][1, 3, 3][1, 1, 1] : 
  tensor<3x3xf32> into tensor<3x3x3xf32>
  %print_out0 = tensor.cast %t2 : tensor<3x3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out0) : (tensor<*xf32>) -> ()
  %t3 = tensor.insert_slice %t1 into %t0[1, 0, 0][1, 3, 3][1, 1, 1] : 
  tensor<3x3xf32> into tensor<3x3x3xf32>
  %print_out1 = tensor.cast %t3 : tensor<3x3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out1) : (tensor<*xf32>) -> ()

  %t4 = tensor.insert_slice %t1 into %t0[0, 1, 0][1, 3, 3][1, 1, 1] : 
  tensor<3x3xf32> into tensor<3x3x3xf32>
  %print_out2 = tensor.cast %t4 : tensor<3x3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out2) : (tensor<*xf32>) -> ()

  %t5 = tensor.insert_slice %t1 into %t0[0, 0, 0][1, 3, 3][1, 2, 1] : 
  tensor<3x3xf32> into tensor<3x3x3xf32>
  %print_out3 = tensor.cast %t5 : tensor<3x3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out3) : (tensor<*xf32>) -> ()

  %t6 = tensor.insert_slice %t1 into %t0[0, 0, 0][1, 3, 3][1, 1, 3] : 
  tensor<3x3xf32> into tensor<3x3x3xf32>
  %print_out4 = tensor.cast %t6 : tensor<3x3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%print_out4) : (tensor<*xf32>) -> ()
  func.return 

}
