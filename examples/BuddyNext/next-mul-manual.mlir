func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel_mul(%arg0 : tensor<1x40x1536xf32>, %arg1 : tensor<1x40x1xf32>) {
  %t_start = call @rtclock() : () -> f64

  // Convert input tensors to memrefs
  %memref_arg0 = bufferization.to_memref %arg0 : memref<1x40x1536xf32>
  %memref_arg1 = bufferization.to_memref %arg1 : memref<1x40x1xf32>

  // Allocate output memref
  %memref.result = memref.alloc() : memref<1x40x1536xf32>

  // Constants for loop bounds and vector size
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %c1536 = arith.constant 1536 : index
  %c16 = arith.constant 16 : index // Vector size for SIMD

  // Outer loops over batch and rows
  scf.for %i = %c0 to %c1 step %c1 {
    scf.for %j = %c0 to %c40 step %c1 {
      // Load scalar value from %arg1 for broadcasting
      %arg1_val = memref.load %memref_arg1[%c0, %j, %c0] : memref<1x40x1xf32>
      %arg1_vec = vector.broadcast %arg1_val : f32 to vector<16xf32>

      // Inner loop processes 1536 elements in chunks of 16
      scf.for %k = %c0 to %c1536 step %c16 {
        // Load vector from %arg0
        %arg0_vec = vector.load %memref_arg0[%c0, %j, %k] : memref<1x40x1536xf32>, vector<16xf32>
        // Perform vectorized multiplication
        %mul_vec = arith.mulf %arg0_vec, %arg1_vec : vector<16xf32>
        // Store result back to output memref
        vector.store %mul_vec, %memref.result[%c0, %j, %k] : memref<1x40x1536xf32>, vector<16xf32>
      }
    }
  }

  // Convert output memref back to tensor for compatibility
  %result = bufferization.to_tensor %memref.result : memref<1x40x1536xf32>

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

  // Deallocate memref
  memref.dealloc %memref.result : memref<1x40x1536xf32>

  return
}

func.func @main() {
  // Initialize input tensors with constant values
  %c2 = arith.constant dense<2.0> : tensor<1x40x1536xf32>
  %c3 = arith.constant dense<3.0> : tensor<1x40x1xf32>

  // Call kernel function
  call @kernel_mul(%c2, %c3) : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> ()

  return
}