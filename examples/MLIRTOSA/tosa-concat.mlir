func.func @main() {

  %0 = arith.constant dense<[[11.,12.],[30.,40.]]> : tensor<2x2xf32>
  %1 = arith.constant dense<[[12.,13.],[23.,45.],[11.,89.]]> : tensor<3x2xf32>

  %output = "tosa.concat"(%0,%1) {axis=0 : i32} : (tensor<2x2xf32>,tensor<3x2xf32>) -> tensor<5x2xf32>
  %tensor_unranked = tensor.cast %output : tensor<5x2xf32> to tensor<*xf32>
  
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
