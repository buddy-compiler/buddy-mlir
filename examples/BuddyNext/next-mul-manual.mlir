// RUN: buddy-opt %s \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @kernel_mul_optimized(%arg0 : tensor<1x40x1536xf32>, %arg1 : tensor<1x40x1xf32>) -> tensor<1x40x1536xf32> {
  // Convert input tensors to memrefs for better performance
  %memref_arg0 = bufferization.to_memref %arg0 : memref<1x40x1536xf32>
  %memref_arg1 = bufferization.to_memref %arg1 : memref<1x40x1xf32>
  
  // Allocate output memref
  %memref_result = memref.alloc() : memref<1x40x1536xf32>
  
  // Constants for vectorization
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c40 = arith.constant 40 : index
  %c1536 = arith.constant 1536 : index
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  
  // Parallel outer loop over rows (OpenMP parallelization)
  scf.parallel (%i) = (%c0) to (%c40) step (%c1) {
    // Load and broadcast scalar value from %arg1 once per row
    %arg1_scalar = memref.load %memref_arg1[%c0, %i, %c0] : memref<1x40x1xf32>
    %arg1_vec32 = vector.broadcast %arg1_scalar : f32 to vector<32xf32>
    
    // Process columns in chunks for better cache locality
    scf.for %chunk_start = %c0 to %c1536 step %c128 {
      %chunk_end = arith.addi %chunk_start, %c128 : index
      %actual_chunk_end = arith.minsi %chunk_end, %c1536 : index
      
      // Vectorized inner loop with unrolling
      scf.for %j = %chunk_start to %actual_chunk_end step %c128 {
        // Calculate remaining elements
        %remaining = arith.subi %actual_chunk_end, %j : index
        %process_full_vectors = arith.cmpi sge, %remaining, %c32 : index
        
        scf.if %process_full_vectors {
          // Process 4 vectors in parallel (loop unrolling)
          %j1 = arith.addi %j, %c32 : index
          %j2 = arith.addi %j, %c32 : index
          %j2_actual = arith.addi %j2, %c32 : index
          %j3 = arith.addi %j2_actual, %c32 : index
          
          // Check bounds for unrolled iterations
          %can_unroll = arith.addi %j, %c128 : index
          %unroll_safe = arith.cmpi sle, %can_unroll, %actual_chunk_end : index
          
          scf.if %unroll_safe {
            // Load 4 vectors simultaneously
            %vec0 = vector.load %memref_arg0[%c0, %i, %j] : memref<1x40x1536xf32>, vector<32xf32>
            %vec1 = vector.load %memref_arg0[%c0, %i, %j1] : memref<1x40x1536xf32>, vector<32xf32>
            %vec2 = vector.load %memref_arg0[%c0, %i, %j2_actual] : memref<1x40x1536xf32>, vector<32xf32>
            %vec3 = vector.load %memref_arg0[%c0, %i, %j3] : memref<1x40x1536xf32>, vector<32xf32>
            
            // Perform vectorized multiplication
            %mul0 = arith.mulf %vec0, %arg1_vec32 : vector<32xf32>
            %mul1 = arith.mulf %vec1, %arg1_vec32 : vector<32xf32>
            %mul2 = arith.mulf %vec2, %arg1_vec32 : vector<32xf32>
            %mul3 = arith.mulf %vec3, %arg1_vec32 : vector<32xf32>
            
            // Store results
            vector.store %mul0, %memref_result[%c0, %i, %j] : memref<1x40x1536xf32>, vector<32xf32>
            vector.store %mul1, %memref_result[%c0, %i, %j1] : memref<1x40x1536xf32>, vector<32xf32>
            vector.store %mul2, %memref_result[%c0, %i, %j2_actual] : memref<1x40x1536xf32>, vector<32xf32>
            vector.store %mul3, %memref_result[%c0, %i, %j3] : memref<1x40x1536xf32>, vector<32xf32>
          } else {
            // Fallback to single vector processing
            %vec = vector.load %memref_arg0[%c0, %i, %j] : memref<1x40x1536xf32>, vector<32xf32>
            %mul = arith.mulf %vec, %arg1_vec32 : vector<32xf32>
            vector.store %mul, %memref_result[%c0, %i, %j] : memref<1x40x1536xf32>, vector<32xf32>
          }
        } else {
          // Handle tail elements with masking
          %tail_size = arith.subi %actual_chunk_end, %j : index
          %mask = vector.create_mask %tail_size : vector<32xi1>
          
          %vec_masked = vector.maskedload %memref_arg0[%c0, %i, %j], %mask, %arg1_vec32 : 
                        memref<1x40x1536xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %mul_masked = arith.mulf %vec_masked, %arg1_vec32 : vector<32xf32>
          vector.maskedstore %memref_result[%c0, %i, %j], %mask, %mul_masked : 
                            memref<1x40x1536xf32>, vector<32xi1>, vector<32xf32>
        }
      }
    }
  }
  
  // Convert output memref back to tensor
  %result = bufferization.to_tensor %memref_result : memref<1x40x1536xf32>
  
  // Deallocate memref before return
  memref.dealloc %memref_result : memref<1x40x1536xf32>
  
  // Return result as the last operation
  return %result : tensor<1x40x1536xf32>
}

func.func @main() {
  // Initialize input tensors with constant values
  %c2 = arith.constant dense<2.0> : tensor<1x40x1536xf32>
  %c3 = arith.constant dense<3.0> : tensor<1x40x1xf32>
  
  // Start timing
  %t_start = call @rtclock() : () -> f64
  
  // Call optimized kernel function
  %result = call @kernel_mul_optimized(%c2, %c3) : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
  
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