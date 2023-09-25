 func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %A = arith.constant dense<1.000000e-01> : tensor<16x32x64xf32>

  %B = "tosa.reduce_sum"(%A) 
          {axis = 1, sche.axis = 1:i32, 
          sche.devices = [{targetId = "cpu", targetConfig = "", duty_ratio = 0.2:f32},
                          {targetId = "gpu", targetConfig = "", duty_ratio = 0.8:f32}]}
          : (tensor<16x32x64xf32>) -> tensor<16x1x64xf32>
  %res = tensor.extract %B[%c1,%c0,%c2] : tensor<16x1x64xf32>
  // CHECK: 10
  vector.print %res : f32

  return
}
 