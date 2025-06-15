// next-silu.mlir
//
// Implements the SiLU (Sigmoid Linear Unit) activation function.
// SiLU(x) = x * sigmoid(x)
// This is realized by composing a tosa.sigmoid and a tosa.mul operation.
//
module {
  // Declare external utility functions for timing and printing.
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)

  //
  // The kernel function that performs the SiLU calculation.
  //
  func.func @kernel_silu(%arg0: tensor<1x40x8960xf32>) {
    %t_start = call @rtclock() : () -> f64

    // Step 1: Calculate the sigmoid of the input.
    %sigmoid_x = tosa.sigmoid %arg0 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>

    // Step 2: Multiply the original input with its sigmoid.
    // This is the SiLU operation. {shift=0} is standard for float multiplication.
    %silu_result = tosa.mul %arg0, %sigmoid_x {shift = 0 : i8} : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    // Cast the result to an unranked tensor for the print function.
    %unranked_result = tensor.cast %silu_result : tensor<1x40x8960xf32> to tensor<*xf32>
    
    // Print the result tensor to verify correctness.
    call @printMemrefF32(%unranked_result) : (tensor<*xf32>) -> ()
    
    // Print the execution time.
    vector.print %time : f64

    return
  }

  //
  // Main function to drive the test.
  //
  func.func @main() {
    // Initialize input with a constant value (e.g., 3.0).
    // The SiLU of 3.0 is 3.0 * sigmoid(3.0) = 3.0 * 0.952574 = 2.857722
    %input_tensor = arith.constant dense<3.0> : tensor<1x40x8960xf32>

    // Call the SiLU kernel.
    call @kernel_silu(%input_tensor) : (tensor<1x40x8960xf32>) -> ()

    return
  }
} 
