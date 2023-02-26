func.func @main() {
  %c0 = arith.constant 0 : index
  %t0 = arith.constant dense<[[0., 1.], [2., 3.]]> : tensor<2x2xf32>
  %print_out0 = tensor.extract %t0[%c0, %c0] : tensor<2x2xf32>
  vector.print %print_out0 : f32
  %t1 = tensor.cast %t0 : tensor<2x2xf32> to tensor<?x?xf32>
  %print_out1 = tensor.extract %t1[%c0, %c0] : tensor<?x?xf32>
  vector.print %print_out1 : f32
  func.return

}
