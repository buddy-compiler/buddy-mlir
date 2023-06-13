func.func @main() {
  %input = arith.constant dense<[[[[1.0], [2.0]],[[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
  %output = "tosa.resize"(%input) {
    scale = array<i64 : 4, 2, 4, 2>, offset = array<i64 : 0, 0>, 
    border = array<i64 : 0, 0>, mode = "NEAREST_NEIGHBOR"} : (tensor<1x2x2x1xf32>) -> (tensor<1x4x4x1xf32>) 
  %tensor_unranked = tensor.cast %output : tensor<1x4x4x1xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
