// NOTE: AMX testing is disabled for automated test suites due to system requirements.
// AMX requires arch_prctl system calls for permission setup which cannot be
// performed in JIT environments and may not be available in CI/testing systems.
//
// To test manually, use: make amx-bf16-matmul-aot
//
// REQUIRES: has_amx
// RUN: make -C %S amx-bf16-matmul-aot | FileCheck %s
//
// AMX BF16 MatMul (No-Transpose Interface)
//
// Expected output verification:
// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 512] strides = [512, 1] data =
// CHECK-NEXT: [
// CHECK: [11439{{(, [0-9]+)*}}
// Requirements:
// - M, N are multiples of 16; K is a multiple of 32.
// - A, B are bf16; C is f32.
// - B must be pre-packed into an "AMX-friendly" layout so that each logical
//   block B[k0:k0+32, n0:n0+16] can be loaded by a single amx.tile_load into a
//   !amx.tile<16x32xbf16> (i.e., stored in memory as 16 rows x 32 columns bf16).
//   This avoids runtime transposes/gathers and ensures optimal AMX loads.
//
// Note:
// The AMX dialect abstracts the hardware orientation; both lhs and rhs tiles for
// amx.tile_mulf use the same tile type !amx.tile<16x32xbf16>, and the reduction
// dimension is K=32 under the hood.

module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefBF16(memref<*xbf16>)

  // Initialize matrix A with cyclic values 1-9
  // A[i,j] = ((i*K + j) % 9) + 1
  // Pattern: 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, ...
  func.func @init_matrix_A(%A: memref<?x?xbf16>, %M: index, %K: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index

    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %K step %c1 {
        // Compute linear index: i*K + j
        %i_times_K = arith.muli %i, %K : index
        %linear_idx = arith.addi %i_times_K, %j : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to float
        %idx_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %idx_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16

        memref.store %val_bf16, %A[%i, %j] : memref<?x?xbf16>
      }
    }
    return
  }

  // Initialize B matrix in standard K×N layout with cyclic values 1-9
  // B[k, n] = ((k*N + n) % 9) + 1
  // Pattern: 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, ...
  func.func @init_matrix_B(%B: memref<?x?xbf16>, %K: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index

    scf.for %k = %c0 to %K step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        // Compute linear index: k*N + n
        %k_times_N = arith.muli %k, %N : index
        %linear_idx = arith.addi %k_times_N, %n : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to float
        %idx_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %idx_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16

        memref.store %val_bf16, %B[%k, %n] : memref<?x?xbf16>
      }
    }
    return
  }

  // Pack B matrix from standard K×N layout into VNNI format
  // Input: B[K x N] in standard row-major layout
  // Output: B_packed[K/2 x N*2] in VNNI format
  // VNNI format: each row contains interleaved elements from 2 consecutive rows of B
  //   B_packed[row_pair, 2*n]   = B[2*row_pair, n]
  //   B_packed[row_pair, 2*n+1] = B[2*row_pair+1, n]
  func.func @pack_B_to_VNNI(%B: memref<?x?xbf16>, %B_packed: memref<?x?xbf16>,
                             %K: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // K must be even (multiple of 2)
    // Iterate over row pairs: row_pair = 0, 1, 2, ..., K/2-1
    %K_div_2 = arith.divui %K, %c2 : index

    scf.for %row_pair = %c0 to %K_div_2 step %c1 {
      // k0 and k1 are the two consecutive rows in original B matrix
      %k0 = arith.muli %row_pair, %c2 : index
      %k1 = arith.addi %k0, %c1 : index

      scf.for %n = %c0 to %N step %c1 {
        // Load values from B[k0, n] and B[k1, n]
        %val0 = memref.load %B[%k0, %n] : memref<?x?xbf16>
        %val1 = memref.load %B[%k1, %n] : memref<?x?xbf16>

        // Store in VNNI format: interleaved
        %col0 = arith.muli %n, %c2 : index
        %col1 = arith.addi %col0, %c1 : index

        memref.store %val0, %B_packed[%row_pair, %col0] : memref<?x?xbf16>
        memref.store %val1, %B_packed[%row_pair, %col1] : memref<?x?xbf16>
      }
    }
    return
  }

  // Real AMX BF16 kernel
  // A: [M x K] in row-major
  // B_packed: [K/2 x N*2] in VNNI format (16 rows x 32 cols for 32x16 block)
  // C: [M x N] in row-major
  func.func @amx_bf16_matmul(
      %A: memref<?x?xbf16>,     // [M x K], row-major
      %B_packed: memref<?x?xbf16>,  // VNNI packed format
      %C: memref<?x?xf32>,      // [M x N], row-major
      %M: index, %N: index, %K: index) {
    %c0  = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index

    scf.for %m = %c0 to %M step %c16 {
      scf.for %n = %c0 to %N step %c16 {
        // Initialize C tile to zero once per (m,n) tile.
        %zero_tile = amx.tile_zero : !amx.tile<16x16xf32>
        amx.tile_store %C[%m, %n], %zero_tile
          : memref<?x?xf32>, !amx.tile<16x16xf32>

        // Reduce across K in chunks of 32 (VNNI requirement for BF16)
        // Each iteration processes A[m:m+16, k:k+32] * B_packed[k/2:k/2+16, n*2:n*2+32]
        scf.for %k = %c0 to %K step %c32 {
          // Load A tile: 16x32 bf16 from [m, k]
          %tA = amx.tile_load %A[%m, %k]
               : memref<?x?xbf16> into !amx.tile<16x32xbf16>

          // Load B_packed tile: 16x32 bf16 in VNNI format
          // B_packed is [K/2 x N*2] in VNNI format
          // For tile at (m, n, k), we need B[k:k+32, n:n+16]
          // In B_packed, this is at [k/2:k/2+16, n*2:n*2+32]
          %c2 = arith.constant 2 : index
          %k_div_2 = arith.divui %k, %c2 : index
          %n_mul_2 = arith.muli %n, %c2 : index
          %tB = amx.tile_load %B_packed[%k_div_2, %n_mul_2]
               : memref<?x?xbf16> into !amx.tile<16x32xbf16>

          // Load current accumulator from C, perform FMA, then store back.
          %tAcc = amx.tile_load %C[%m, %n]
                 : memref<?x?xf32> into !amx.tile<16x16xf32>
          %tAcc2 = amx.tile_mulf %tA, %tB, %tAcc
                  : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
          amx.tile_store %C[%m, %n], %tAcc2
            : memref<?x?xf32>, !amx.tile<16x16xf32>
        }
      }
    }
    return
  }

  // Test with M=512, K=512, N=512
  func.func @amx_main() {
    %c0    = arith.constant 0 : index
    %c256  = arith.constant 256 : index
    %c512  = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index

    // Allocate matrices
    // A: [512 x 512] (M x K)
    // B: [512 x 512] (K x N) - standard layout
    // B_packed: [256 x 1024] (K/2 x N*2) - VNNI format
    // C: [512 x 512] (M x N)
    %A = memref.alloc(%c512, %c512) : memref<?x?xbf16>
    %B = memref.alloc(%c512, %c512) : memref<?x?xbf16>
    %B_packed = memref.alloc(%c256, %c1024) : memref<?x?xbf16>
    %C = memref.alloc(%c512, %c512) : memref<?x?xf32>

    // Initialize A and B with cyclic values 1-9
    // A[i,j] = ((i*K + j) % 9) + 1  (pattern: 1,2,3,4,5,6,7,8,9,1,2,3,...)
    // B[k,n] = ((k*N + n) % 9) + 1  (pattern: 1,2,3,4,5,6,7,8,9,1,2,3,...)
    call @init_matrix_A(%A, %c512, %c512) : (memref<?x?xbf16>, index, index) -> ()
    call @init_matrix_B(%B, %c512, %c512) : (memref<?x?xbf16>, index, index) -> ()

    // Pack B into VNNI format
    call @pack_B_to_VNNI(%B, %B_packed, %c512, %c512)
      : (memref<?x?xbf16>, memref<?x?xbf16>, index, index) -> ()

    // Initialize C = 0.0
    %zero_f32 = arith.constant 0.0 : f32
    linalg.fill ins(%zero_f32 : f32) outs(%C : memref<?x?xf32>)

    // Start timing
    %t_start = call @rtclock() : () -> f64

    // Call AMX kernel
    // C[512×512] = A[512×512] × B_packed[256×1024 in VNNI format]
    call @amx_bf16_matmul(%A, %B_packed, %C, %c512, %c512, %c512)
      : (memref<?x?xbf16>, memref<?x?xbf16>, memref<?x?xf32>, index, index, index) -> ()

    %t_end = call @rtclock() : () -> f64
    %computation_time = arith.subf %t_end, %t_start : f64

    // Print the entire output matrix C[512×512]
    %Cu = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%Cu) : (memref<*xf32>) -> ()

    // Print timing result
    // CHECK: {{[0-9]+\.[0-9]+}}
    vector.print %computation_time : f64

    // Cleanup
    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B_packed : memref<?x?xbf16>
    memref.dealloc %B : memref<?x?xbf16>
    memref.dealloc %A : memref<?x?xbf16>
    return
  }
}
