func.func @main() {
    
  %input = arith.constant dense<[[11.0,12.0],[30.0,40.0]]> : tensor<2x2xf32>
  %output = "tosa.log"(%input) {} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)

  %tensor_unranked = tensor.cast %output : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
