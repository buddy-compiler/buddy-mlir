// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s


// External functions for utilities
// printMemrefF32: Prints contents of a float32 memref buffer
// rtclock: Returns current time for performance measurement
func.func private @printMemrefF32(memref<*xf32>)
func.func private @rtclock() -> f64

// Main SpMM computation kernel
// Parameters:
// - values: Non-zero values of sparse matrix A in CSR format
// - col_indices: Column indices for each non-zero in A
// - row_pointers: Start/end indices for each row of A in values array
// - dense: Dense input matrix B
// - result: Output dense matrix C = A * B
func.func @spmm(%values: memref<?xf32>, %col_indices: memref<?xi32>, 
                %row_pointers: memref<?xi32>, %dense: memref<?x?xf32>, 
                %result: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // Get dimensions for iteration bounds
  // num_rows: Number of rows in sparse matrix A and result C
  // num_cols: Number of columns in result C
  // dense_cols: Number of columns in dense matrix B
  %num_rows = memref.dim %result, %c0 : memref<?x?xf32>
  %num_cols = memref.dim %result, %c1 : memref<?x?xf32>
  %dense_cols = memref.dim %dense, %c1 : memref<?x?xf32>
  
  // Start timing the computation
  %t_start = call @rtclock() : () -> f64
  
  // Main computation loops:
  // 1. Outer loop: Iterate over each row of sparse matrix A
  // 2. Middle loop: Iterate over each column of dense matrix B
  // 3. Inner loop: Process non-zeros in current row of A
  scf.for %i = %c0 to %num_rows step %c1 {
    // Get start and end indices for current row in CSR format
    // These indices mark the range of non-zeros in the current row
    %row_start_ptr = memref.load %row_pointers[%i] : memref<?xi32>
    %row_start = arith.index_cast %row_start_ptr : i32 to index
    
    %i_plus_1 = arith.addi %i, %c1 : index
    %row_end_ptr = memref.load %row_pointers[%i_plus_1] : memref<?xi32>
    %row_end = arith.index_cast %row_end_ptr : i32 to index
    
    // Process each column in result matrix C
    scf.for %j = %c0 to %dense_cols step %c1 {
      // Initialize accumulator for dot product computation
      %sum = arith.constant 0.0 : f32
      
      // Compute dot product of current row of A with column j of B
      %result_sum = scf.for %k = %row_start to %row_end step %c1 iter_args(%current_sum = %sum) -> (f32) {
        // Load non-zero value from A and its column index
        %val = memref.load %values[%k] : memref<?xf32>
        %col_ptr = memref.load %col_indices[%k] : memref<?xi32>
        %col = arith.index_cast %col_ptr : i32 to index
        
        // Load corresponding value from dense matrix B
        %dense_val = memref.load %dense[%col, %j] : memref<?x?xf32>
        
        // Multiply and accumulate into partial sum
        %prod = arith.mulf %val, %dense_val : f32
        %new_sum = arith.addf %current_sum, %prod : f32
        
        scf.yield %new_sum : f32
      }
      
      // Store computed result in output matrix C
      memref.store %result_sum, %result[%i, %j] : memref<?x?xf32>
    }
  }
  

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  
  %printed_result = memref.cast %result : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed_result) : (memref<*xf32>) -> ()

  vector.print %time : f64

  return
}

// Helper function to allocate and initialize values array
// Allocates memory for given size and fills with specified value
func.func @alloc_values(%size: index, %val: f32) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %values = memref.alloc(%size) : memref<?xf32>
  
  scf.for %i = %c0 to %size step %c1 {
    memref.store %val, %values[%i] : memref<?xf32>
  }
  
  return %values : memref<?xf32>
}

// Helper function to allocate and initialize column indices array
// Creates cyclic pattern of indices modulo pattern size
func.func @alloc_col_indices(%size: index, %pattern: index) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %indices = memref.alloc(%size) : memref<?xi32>
  
  scf.for %i = %c0 to %size step %c1 {
    %col = arith.remui %i, %pattern : index
    %col_i32 = arith.index_cast %col : index to i32
    memref.store %col_i32, %indices[%i] : memref<?xi32>
  }
  
  return %indices : memref<?xi32>
}

// Helper function to allocate and initialize row pointers array
// Creates regular pattern with fixed number of non-zeros per row
func.func @alloc_row_pointers(%rows: index, %nnz_per_row: index) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows_plus_1 = arith.addi %rows, %c1 : index
  %pointers = memref.alloc(%rows_plus_1) : memref<?xi32>
  
  scf.for %i = %c0 to %rows_plus_1 step %c1 {
    %val = arith.muli %i, %nnz_per_row : index
    %val_i32 = arith.index_cast %val : index to i32
    memref.store %val_i32, %pointers[%i] : memref<?xi32>
  }
  
  return %pointers : memref<?xi32>
}

// Helper function to allocate and initialize 2D float array
// Allocates memory for given dimensions and fills with specified value
func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  
  scf.for %idx0 = %c0 to %arg0 step %c1 {
    scf.for %idx1 = %c0 to %arg1 step %c1 {
      memref.store %arg2, %0[%idx0, %idx1] : memref<?x?xf32>
    }
  }
  
  return %0 : memref<?x?xf32>
}

// Main function that sets up test case and runs SpMM
// Creates a test case with:
// - 4x4 sparse matrix with 2 non-zeros per row (values = 2.0)
// - 4x3 dense matrix (values = 3.0)
// - 4x3 result matrix (initialized to 0.0)
func.func @main() {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  
  // Test parameters:
  // nnz: Total number of non-zeros in sparse matrix (8)
  // nnz_per_row: Number of non-zeros per row (2)
  %nnz = arith.constant 8 : index  
  %nnz_per_row = arith.constant 2 : index  
  
  // Allocate and initialize sparse matrix components
  %values = call @alloc_values(%nnz, %f2) : (index, f32) -> memref<?xf32>
  %col_indices = call @alloc_col_indices(%nnz, %c4) : (index, index) -> memref<?xi32>
  %row_pointers = call @alloc_row_pointers(%c4, %nnz_per_row) : (index, index) -> memref<?xi32>

  // Allocate and initialize dense input and result matrices
  %dense = call @alloc_f32(%c4, %c3, %f3) : (index, index, f32) -> memref<?x?xf32>
  //4x3
  %result = call @alloc_f32(%c4, %c3, %f0) : (index, index, f32) -> memref<?x?xf32>
  
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 3] strides = [3, 1] data =
  // CHECK-NEXT: [[12,   12,   12],
  // CHECK-NEXT:  [12,   12,   12],
  // CHECK-NEXT:  [12,   12,   12],
  // CHECK-NEXT:  [12,   12,   12]]
  call @spmm(%values, %col_indices, %row_pointers, %dense, %result) : 
    (memref<?xf32>, memref<?xi32>, memref<?xi32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  
  return
}
