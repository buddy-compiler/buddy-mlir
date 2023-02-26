func.func @main() {
  %input = arith.constant dense<[[[[1.0], [2.0]],[[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
  %output = "tosa.resize"(%input) {
    scale = [4, 2, 4, 2], output_size = [4, 4], stride = [0, 0], offset = [0, 0],
    stride_fp = [0.5 : f32, 0.5 : f32], offset_fp = [0.1 : f32, 0.2 : f32], shift = 0 : i32,
    mode = "NEAREST_NEIGHBOR", border = [0, 0]} : (tensor<1x2x2x1xf32>)  -> (tensor<1x4x4x1xf32>)
  %tensor_unranked = tensor.cast %output : tensor<1x4x4x1xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
