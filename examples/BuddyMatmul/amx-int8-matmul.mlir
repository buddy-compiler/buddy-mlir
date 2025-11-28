// NOTE: AMX testing is disabled for automated test suites due to system requirements.
// AMX requires arch_prctl system calls for permission setup which cannot be
// performed in JIT environments and may not be available in CI/testing systems.
//
// To test manually, use: make amx-int8-matmul-aot
//
// REQUIRES: has_amx
// RUN: make -C %S amx-int8-matmul-aot | FileCheck %s
//
// AMX INT8 MatMul (No-Transpose Interface)
//
// Expected output verification:
// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 512] strides = [512, 1] data =
// CHECK-NEXT: [
// CHECK: [11439{{(, [0-9]+)*}}
//
// Requirements:
// - M, N are multiples of 16; K is a multiple of 64.
// - A, B are i8; C is i32.
// - B must be pre-packed into an "AMX-friendly" VNNI layout so that each logical
//   block B[k0:k0+64, n0:n0+16] can be loaded by a single amx.tile_load into a
//   !amx.tile<16x64xi8> (i.e., stored in memory as 16 rows x 64 columns i8).
//   This avoids runtime transposes/gathers and ensures optimal AMX loads.
//
// Note:
// The AMX dialect abstracts the hardware orientation; both lhs and rhs tiles for
// amx.tile_muli use the same tile type !amx.tile<16x64xi8>, and the reduction
// dimension is K=64 under the hood (4 bytes per column, 4-way dot product).
//
// INT8 VNNI format:
// For INT8, AMX requires VNNI format where 4 consecutive elements along K are
// interleaved. Each tile load of !amx.tile<16x64xi8> processes 64 K elements
// by loading 16 rows of 64 bytes each.

module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI8(memref<*xi8>)

  // Initialize matrix A with cyclic values 1-9 (as signed i8)
  // A[i,j] = ((i*K + j) % 9) + 1
  // Pattern: 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, ...
  func.func @init_matrix_A(%A: memref<?x?xi8>, %M: index, %K: index) {
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

        // Convert to i8
        %val_i8 = arith.index_cast %cyclic_val : index to i8

        memref.store %val_i8, %A[%i, %j] : memref<?x?xi8>
      }
    }
    return
  }

  // Initialize B matrix in standard K×N layout with cyclic values 1-9 (as signed i8)
  // B[k, n] = ((k*N + n) % 9) + 1
  // Pattern: 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, ...
  func.func @init_matrix_B(%B: memref<?x?xi8>, %K: index, %N: index) {
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

        // Convert to i8
        %val_i8 = arith.index_cast %cyclic_val : index to i8

        memref.store %val_i8, %B[%k, %n] : memref<?x?xi8>
      }
    }
    return
  }

  // Pack B matrix from standard K×N layout into VNNI format for INT8
  // Input: B[K x N] in standard row-major layout
  // Output: B_packed[K/4 x N*4] in VNNI format
  // VNNI format for INT8: each row contains interleaved elements from 4 consecutive rows of B
  //   B_packed[row_quad, 4*n+0] = B[4*row_quad+0, n]
  //   B_packed[row_quad, 4*n+1] = B[4*row_quad+1, n]
  //   B_packed[row_quad, 4*n+2] = B[4*row_quad+2, n]
  //   B_packed[row_quad, 4*n+3] = B[4*row_quad+3, n]
  func.func @pack_B_to_VNNI(%B: memref<?x?xi8>, %B_packed: memref<?x?xi8>,
                             %K: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // K must be multiple of 4
    // Iterate over row quads: row_quad = 0, 1, 2, ..., K/4-1
    %K_div_4 = arith.divui %K, %c4 : index

    scf.for %row_quad = %c0 to %K_div_4 step %c1 {
      // k0, k1, k2, k3 are the four consecutive rows in original B matrix
      %k0 = arith.muli %row_quad, %c4 : index
      %k1 = arith.addi %k0, %c1 : index
      %k2 = arith.addi %k0, %c2 : index
      %k3 = arith.addi %k0, %c3 : index

      scf.for %n = %c0 to %N step %c1 {
        // Load values from B[k0, n], B[k1, n], B[k2, n], B[k3, n]
        %val0 = memref.load %B[%k0, %n] : memref<?x?xi8>
        %val1 = memref.load %B[%k1, %n] : memref<?x?xi8>
        %val2 = memref.load %B[%k2, %n] : memref<?x?xi8>
        %val3 = memref.load %B[%k3, %n] : memref<?x?xi8>

        // Store in VNNI format: interleaved
        %col0 = arith.muli %n, %c4 : index
        %col1 = arith.addi %col0, %c1 : index
        %col2 = arith.addi %col0, %c2 : index
        %col3 = arith.addi %col0, %c3 : index

        memref.store %val0, %B_packed[%row_quad, %col0] : memref<?x?xi8>
        memref.store %val1, %B_packed[%row_quad, %col1] : memref<?x?xi8>
        memref.store %val2, %B_packed[%row_quad, %col2] : memref<?x?xi8>
        memref.store %val3, %B_packed[%row_quad, %col3] : memref<?x?xi8>
      }
    }
    return
  }

  // Real AMX INT8 kernel (signed x signed = signed)
  // A: [M x K] in row-major (i8)
  // B_packed: [K/4 x N*4] in VNNI format (16 rows x 64 cols for 64x16 block)
  // C: [M x N] in row-major (i32)
  //
  // AMX INT8 tile configuration:
  // - !amx.tile<16x64xi8> for both A and B tiles
  // - !amx.tile<16x16xi32> for accumulator C tile
  // - Each tile_muli processes K=64 elements (4-way i8 dot product)
  func.func @amx_int8_matmul(
      %A: memref<?x?xi8>,           // [M x K], row-major
      %B_packed: memref<?x?xi8>,    // VNNI packed format [K/4 x N*4]
      %C: memref<?x?xi32>,          // [M x N], row-major
      %M: index, %N: index, %K: index) {
    %c0  = arith.constant 0 : index
    %c4  = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    scf.for %m = %c0 to %M step %c16 {
      scf.for %n = %c0 to %N step %c16 {
        // Initialize C tile to zero once per (m,n) tile.
        %zero_tile = amx.tile_zero : !amx.tile<16x16xi32>
        amx.tile_store %C[%m, %n], %zero_tile
          : memref<?x?xi32>, !amx.tile<16x16xi32>

        // Reduce across K in chunks of 64 (VNNI requirement for INT8)
        // Each iteration processes A[m:m+16, k:k+64] * B_packed[k/4:k/4+16, n*4:n*4+64]
        scf.for %k = %c0 to %K step %c64 {
          // Load A tile: 16x64 i8 from [m, k]
          %tA = amx.tile_load %A[%m, %k]
               : memref<?x?xi8> into !amx.tile<16x64xi8>

          // Load B_packed tile: 16x64 i8 in VNNI format
          // B_packed is [K/4 x N*4] in VNNI format
          // For tile at (m, n, k), we need B[k:k+64, n:n+16]
          // In B_packed, this is at [k/4:k/4+16, n*4:n*4+64]
          %k_div_4 = arith.divui %k, %c4 : index
          %n_mul_4 = arith.muli %n, %c4 : index
          %tB = amx.tile_load %B_packed[%k_div_4, %n_mul_4]
               : memref<?x?xi8> into !amx.tile<16x64xi8>

          // Load current accumulator from C, perform integer matmul, then store back.
          // Using signed x signed (default, no zext attributes)
          %tAcc = amx.tile_load %C[%m, %n]
                 : memref<?x?xi32> into !amx.tile<16x16xi32>
          %tAcc2 = amx.tile_muli %tA, %tB, %tAcc
                  : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
          amx.tile_store %C[%m, %n], %tAcc2
            : memref<?x?xi32>, !amx.tile<16x16xi32>
        }
      }
    }
    return
  }

  // Test with M=512, K=512, N=512
  // Note: K must be multiple of 64 for INT8 AMX
  func.func @amx_main() {
    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c128  = arith.constant 128 : index
    %c512  = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index

    // Allocate matrices
    // A: [512 x 512] (M x K)
    // B: [512 x 512] (K x N) - standard layout
    // B_packed: [128 x 2048] (K/4 x N*4) - VNNI format for INT8
    // C: [512 x 512] (M x N)
    %A = memref.alloc(%c512, %c512) : memref<?x?xi8>
    %B = memref.alloc(%c512, %c512) : memref<?x?xi8>
    %B_packed = memref.alloc(%c128, %c2048) : memref<?x?xi8>
    %C = memref.alloc(%c512, %c512) : memref<?x?xi32>

    // Initialize A and B with cyclic values 1-9
    // A[i,j] = ((i*K + j) % 9) + 1  (pattern: 1,2,3,4,5,6,7,8,9,1,2,3,...)
    // B[k,n] = ((k*N + n) % 9) + 1  (pattern: 1,2,3,4,5,6,7,8,9,1,2,3,...)
    call @init_matrix_A(%A, %c512, %c512) : (memref<?x?xi8>, index, index) -> ()
    call @init_matrix_B(%B, %c512, %c512) : (memref<?x?xi8>, index, index) -> ()

    // Pack B into VNNI format for INT8
    call @pack_B_to_VNNI(%B, %B_packed, %c512, %c512)
      : (memref<?x?xi8>, memref<?x?xi8>, index, index) -> ()

    // Initialize C = 0
    %zero_i32 = arith.constant 0 : i32
    linalg.fill ins(%zero_i32 : i32) outs(%C : memref<?x?xi32>)

    // Start timing
    %t_start = call @rtclock() : () -> f64

    // Call AMX kernel
    // C[512×512] = A[512×512] × B_packed[128×2048 in VNNI format]
    call @amx_int8_matmul(%A, %B_packed, %C, %c512, %c512, %c512)
      : (memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, index, index, index) -> ()

    %t_end = call @rtclock() : () -> f64
    %computation_time = arith.subf %t_end, %t_start : f64

    // Print I32 result
    %Cu_i32 = memref.cast %C : memref<?x?xi32> to memref<*xi32>
    call @printMemrefI32(%Cu_i32) : (memref<*xi32>) -> ()

    // Print timing result
    // CHECK: {{[0-9]+\.[0-9]+}}
    vector.print %computation_time : f64

    // Cleanup
    memref.dealloc %C : memref<?x?xi32>
    memref.dealloc %B_packed : memref<?x?xi8>
    memref.dealloc %B : memref<?x?xi8>
    memref.dealloc %A : memref<?x?xi8>
    return
  }
}
