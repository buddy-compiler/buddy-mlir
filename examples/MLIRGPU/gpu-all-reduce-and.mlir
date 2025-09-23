// =============================================================================
// MLIR GPU All-Reduce AND Operation Example
// This example demonstrates how to perform all-reduce AND operation on GPU
// all-reduce AND: Performs bitwise AND operation on data from all threads
// within each block.
// =============================================================================

func.func @main() {
  // Allocate memory buffers
  // %data: 2x6 2D array to store input data
  // 2 rows represent 2 different data sets
  // 6 columns represent 6 elements per set
  %data = memref.alloc() : memref<2x6xi32>

  // %sum: 1D array of length 2 to store AND reduction results for each row
  %sum = memref.alloc() : memref<2xi32>

  // Define constant values for the first row data (0, 1, 2, 4, 8, 16)
  // These values are chosen to demonstrate the effect of AND operation
  %cst0 = arith.constant 0 : i32   // Binary: 0000
  %cst1 = arith.constant 1 : i32   // Binary: 0001
  %cst2 = arith.constant 2 : i32   // Binary: 0010
  %cst4 = arith.constant 4 : i32   // Binary: 0100
  %cst8 = arith.constant 8 : i32   // Binary: 1000
  %cst16 = arith.constant 16 : i32 // Binary: 10000

  // Define constant values for the second row data (2, 3, 6, 7, 10, 11)
  // These values are also carefully chosen to demonstrate AND operation
  %cst3 = arith.constant 3 : i32   // Binary: 0011
  %cst6 = arith.constant 6 : i32   // Binary: 0110
  %cst7 = arith.constant 7 : i32   // Binary: 0111
  %cst10 = arith.constant 10 : i32 // Binary: 1010
  %cst11 = arith.constant 11 : i32 // Binary: 1011

  // Define index constants for array access
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // ===========================================================================
  // GPU Memory Management: Unified Memory
  // ===========================================================================
  // gpu.host_register implements unified memory management with the following
  // features:
  // 1. Registered memory can be accessed by both GPU and CPU
  // 2. No explicit CPU-GPU data transfer (copy) required
  // 3. After GPU modification, CPU can directly read the latest values
  // 4. This is a simplified mode of modern GPU programming,
  //    similar to CUDA Unified Memory

  %cast_data = memref.cast %data : memref<2x6xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>  // Register input data buffer

  %cast_sum = memref.cast %sum : memref<2xi32> to memref<*xi32>
  gpu.host_register %cast_sum : memref<*xi32>   // Register result buffer

  // Note: After GPU kernel execution, CPU can directly access the updated data
  // without additional synchronization or copy operations

  // Initialize first row data: [0, 1, 2, 4, 8, 16]
  // Expected AND result: 0 & 1 & 2 & 4 & 8 & 16 = 0
  // (because 0 AND anything is 0)
  memref.store %cst0, %data[%c0, %c0] : memref<2x6xi32>   // data[0,0] = 0
  memref.store %cst1, %data[%c0, %c1] : memref<2x6xi32>   // data[0,1] = 1
  memref.store %cst2, %data[%c0, %c2] : memref<2x6xi32>   // data[0,2] = 2
  memref.store %cst4, %data[%c0, %c3] : memref<2x6xi32>   // data[0,3] = 4
  memref.store %cst8, %data[%c0, %c4] : memref<2x6xi32>   // data[0,4] = 8
  memref.store %cst16, %data[%c0, %c5] : memref<2x6xi32>  // data[0,5] = 16

  // Initialize second row data: [2, 3, 6, 7, 10, 11]
  // Expected AND result: 2 & 3 & 6 & 7 & 10 & 11 = 2
  // Detailed calculation: 0010 & 0011 & 0110 & 0111 & 1010 & 1011 = 0010 = 2
  memref.store %cst2, %data[%c1, %c0] : memref<2x6xi32>   // data[1,0] = 2
  memref.store %cst3, %data[%c1, %c1] : memref<2x6xi32>   // data[1,1] = 3
  memref.store %cst6, %data[%c1, %c2] : memref<2x6xi32>   // data[1,2] = 6
  memref.store %cst7, %data[%c1, %c3] : memref<2x6xi32>   // data[1,3] = 7
  memref.store %cst10, %data[%c1, %c4] : memref<2x6xi32>  // data[1,4] = 10
  memref.store %cst11, %data[%c1, %c5] : memref<2x6xi32>  // data[1,5] = 11

  // ===========================================================================
  // GPU Kernel Launch - All-Reduce AND Operation
  // ===========================================================================

  // gpu.launch starts GPU kernel and defines GPU execution configuration
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {

    // GPU execution configuration explanation:
    // - Grid dimension: (2, 1, 1) - Total 2 blocks, each block processes one
    //   row of data
    // - Block dimension: (6, 1, 1) - Each block has 6 threads, corresponding
    //   to 6 elements per row
    // - %bx: block index in x dimension (0, 1) - corresponds to row index
    // - %tx: thread index in x dimension within block (0, 1, 2, 3, 4, 5)
    //   - corresponds to column index

    // Each thread loads the corresponding data element
    // thread(bx, tx) loads data[bx, tx]
    %val = memref.load %data[%bx, %tx] : memref<2x6xi32>

    // =========================================================================
    // All-Reduce AND Operation
    // =========================================================================
    // gpu.all_reduce performs reduction operation among all threads within
    // the same block
    // - 'and': reduction operation type, performs bitwise AND operation
    // - %val: input value from each thread
    // - uniform {}: indicates all threads use the same reduction function
    //   (empty here, uses default AND)
    // - Result: all threads get the same reduction result
    %reduced = gpu.all_reduce and %val uniform {} : (i32) -> (i32)

    // Store reduction result to result array
    // Note: all threads write to the same position, but values are the same
    // so it's fine
    memref.store %reduced, %sum[%bx] : memref<2xi32>

    // GPU kernel termination marker
    gpu.terminator
  }

  // ===========================================================================
  // Result Reading: No Additional Data Transfer Required
  // ===========================================================================
  // Due to using gpu.host_register, we can directly read the GPU computation
  // results here
  // No explicit GPU→CPU data copy needed, MLIR runtime automatically handles
  // memory synchronization

  // Expected output: [0, 2]
  // - sum[0] = 0 & 1 & 2 & 4 & 8 & 16 = 0 (AND result of first row)
  // - sum[1] = 2 & 3 & 6 & 7 & 10 & 11 = 2 (AND result of second row)
  call @printMemrefI32(%cast_sum) : (memref<*xi32>) -> ()

  return
}

// External function declaration: used to print memref content
// This function needs to be implemented in the runtime environment
func.func private @printMemrefI32(memref<*xi32>)

// =============================================================================
// Execution Flow Summary:
// 1. Allocate and initialize data buffers (2x6 input data, 2 results)
// 2. Register host memory to GPU
// 3. Launch GPU kernel with 2 blocks, each block has 6 threads
// 4. Each block processes one row of data, performs all-reduce AND operation
// 5. Store results and print
//
// GPU Execution Pattern:
// Block 0: threads(0-5) process data[0, 0-5] → sum[0] = 0
// Block 1: threads(0-5) process data[1, 0-5] → sum[1] = 2
// =============================================================================
