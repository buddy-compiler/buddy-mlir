func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel_mul(%arg0 : tensor<1x40x1536xf32>, %arg1 : tensor<1x40x1xf32>) -> tensor<1x40x1536xf32> {
  // Perform element-wise multiplication with shift=0
  %result = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
  
  return %result : tensor<1x40x1536xf32>
}

func.func @main() {
  // Initialize input tensors with constant values
  %c2 = arith.constant dense<2.0> : tensor<1x40x1536xf32>
  %c3 = arith.constant dense<3.0> : tensor<1x40x1xf32>
  
  // Start timing
  %t_start = call @rtclock() : () -> f64
  
  // Call kernel function
  %result = call @kernel_mul(%c2, %c3) : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
  
  // End timing
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  // Cast result to unranked tensor for printing
  %tensor_unranked = tensor.cast %result : tensor<1x40x1536xf32> to tensor<*xf32>
  
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 1536] strides = [61440, 1536, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [
  // CHECK-SAME: [6{{(, 6)*}}],
  
  // Print results
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print execution time
  vector.print %time : f64
  
  return
}