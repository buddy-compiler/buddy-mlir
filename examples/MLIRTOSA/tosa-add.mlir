func.func @main() {
    
  %0 = arith.constant dense<[[11.,12.],[30.,40.]]> : tensor<2x2xf32>
  %1 = arith.constant dense <[[1.],[23.]]> : tensor<2x1xf32>

  %2 = "tosa.add"(%0,%1) {} : (tensor<2x2xf32>,tensor<2x1xf32>) -> tensor<2x2xf32>
  %tensor_unranked = tensor.cast %2 : tensor<2x2xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
