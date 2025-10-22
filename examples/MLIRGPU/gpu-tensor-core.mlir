// =============================================================================
// MLIR GPU Tensor Core Example - Using WMMA (Warp Matrix Multiply-Accumulate)
// =============================================================================
//
// This example demonstrates how to directly use Tensor Core instructions for
// matrix multiplication.
//
// Tensor Core Architecture Mapping:
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Tensor Core Hardware Hierarchy               │
// ├─────────────────────────────────────────────────────────────────┤
// │ Warp (32 threads)  ──→  Using Tensor Core for Computation       │
// │   └─ WMMA Instructions  ──→  16x16x16 Matrix Multiplication     │
// │       └─ Data Types: f16, bf16, tf32, int8, etc.                │
// └─────────────────────────────────────────────────────────────────┘
//
// ============================================================================

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @tensor_core_matmul(%arg0: memref<16x16xf16>,
                                 %arg1: memref<16x16xf16>,
                                 %arg2: memref<16x16xf32>) kernel {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16

      // =======================================================================
      // Using Tensor Core for 16x16x16 Matrix Multiplication
      // =======================================================================
      // Load matrix A to Tensor Core (16x16 f16)
      %A = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index, transpose}
          : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">

      // Load matrix B to Tensor Core (16x16 f16)
      %B = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index}
          : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">

      // Load accumulator matrix C to Tensor Core (16x16 f32)
      %C = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 16 : index}
          : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

      // Tensor Core computation: C = A * B + C
      %D = gpu.subgroup_mma_compute %A, %B, %C {a_transpose}
          : !gpu.mma_matrix<16x16xf16, "AOp">,
            !gpu.mma_matrix<16x16xf16, "BOp">
          -> !gpu.mma_matrix<16x16xf32, "COp">

      // Store result matrix D
      gpu.subgroup_mma_store_matrix %D, %arg2[%c0, %c0] {leadDimension = 16 : index}
          : !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>

      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %cst_f16 = arith.constant 0.000000e+00 : f16
    %cst_f32 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.0 : f16
    %c2 = arith.constant 2.0 : f16

    // =========================================================================
    // 16x16 Matrix Allocation - Tensor Core Optimized Size
    // =========================================================================
    %A = memref.alloc() : memref<16x16xf16>
    %B = memref.alloc() : memref<16x16xf16>
    %C = memref.alloc() : memref<16x16xf32>

    // Initialize matrix A (all 1.0)
    %cst_A = arith.constant dense<1.0> : vector<16x16xf16>
    vector.transfer_write %cst_A, %A[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>

    // Initialize matrix B (all 2.0)
    %cst_B = arith.constant dense<2.0> : vector<16x16xf16>
    vector.transfer_write %cst_B, %B[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>

    // Initialize result matrix C to 0
    %cst_C = arith.constant dense<0.0> : vector<16x16xf32>
    vector.transfer_write %cst_C, %C[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>

    // GPU memory registration
    %A_cast = memref.cast %A : memref<16x16xf16> to memref<*xf16>
    gpu.host_register %A_cast : memref<*xf16>
    %B_cast = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    gpu.host_register %B_cast : memref<*xf16>
    %C_cast = memref.cast %C : memref<16x16xf32> to memref<*xf32>
    gpu.host_register %C_cast : memref<*xf32>

    // =========================================================================
    // Tensor Core GPU Launch Configuration
    // =========================================================================
    // Use 32 threads per warp (Tensor Core requirement)
    %c32 = arith.constant 32 : index
    gpu.launch_func @kernels::@tensor_core_matmul
        blocks in (%c1_idx, %c1_idx, %c1_idx)    // 1x1x1 = 1 block
        threads in (%c32, %c1_idx, %c1_idx)      // 32x1x1 = 32 threads (1 warp)
        args(%A:memref<16x16xf16>, %B:memref<16x16xf16>, %C:memref<16x16xf32>)

    // Print result matrix
    %cast_sum = memref.cast %C : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast_sum) : (memref<*xf32>) -> ()

    // Clean up memory
    memref.dealloc %A : memref<16x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    memref.dealloc %C : memref<16x16xf32>
    return
  }

  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
