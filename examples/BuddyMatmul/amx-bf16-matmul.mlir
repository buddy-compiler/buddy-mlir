// NOTE: AMX testing is disabled for automated test suites due to system requirements.
// AMX requires arch_prctl system calls for permission setup which cannot be
// performed in JIT environments and may not be available in CI/testing systems.
//
// To test manually, use: make amx-bf16-matmul-aot
//
// RUN: 
// 
// AMX BF16 MatMul (No-Transpose Interface)
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

  // Real AMX BF16 kernel
  func.func @amx_bf16_matmul(
      %A: memref<?x?xbf16>,     // [M x K], row-major
      %Bpack: memref<?x?xbf16>, // B pre-packed for AMX-friendly tile loads
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

        // Reduce across K in chunks of 32. Accumulator is stored/loaded via C tile.
        scf.for %k0 = %c0 to %K step %c32 {
          // Load A sub-block: 16x32xbf16 from [%m, %k0]
          %tA = amx.tile_load %A[%m, %k0]
               : memref<?x?xbf16> into !amx.tile<16x32xbf16>

          // Load B sub-block (pre-packed): 16x32xbf16 from [%k0, %n]
          %tB = amx.tile_load %Bpack[%k0, %n]
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

  // Performance test with MLIR-level timing: larger matrices for meaningful benchmarks.
  func.func @amx_main() {
    %c0  = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    // Test with 512x2048x1024 matrices: A[512x1024] × B[1024x2048] = C[512x2048]
    // Allocate matrices
    %A = memref.alloc(%c512, %c1024) : memref<?x?xbf16>      // 512x1024
    %Bpack = memref.alloc(%c1024, %c2048) : memref<?x?xbf16>  // 1024x2048 (pre-packed)
    %C = memref.alloc(%c512, %c2048) : memref<?x?xf32>       // 512x2048

    // Initialize A = 1.0bf16, Bpack = 1.0bf16, C = 0.0f32
    %one_bf16 = arith.constant 1.0 : bf16
    %zero_f32 = arith.constant 0.0 : f32

    linalg.fill ins(%one_bf16 : bf16) outs(%A : memref<?x?xbf16>)
    linalg.fill ins(%one_bf16 : bf16) outs(%Bpack : memref<?x?xbf16>)
    linalg.fill ins(%zero_f32 : f32) outs(%C : memref<?x?xf32>)

    // Start timing
    %t_start = call @rtclock() : () -> f64

    // Call AMX kernel
    call @amx_bf16_matmul(%A, %Bpack, %C, %c512, %c2048, %c1024)
      : (memref<?x?xbf16>, memref<?x?xbf16>, memref<?x?xf32>, index, index, index) -> ()

    // End timing (only measure computation, not printing)
    %t_end = call @rtclock() : () -> f64
    %computation_time = arith.subf %t_end, %t_start : f64

    // Print the entire output matrix
    // All elements should be ~1024.0f (since A=1.0, B=1.0, K=1024)
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 2048] strides = [2048, 1] data =
    %Cu = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%Cu) : (memref<*xf32>) -> ()

    // Print timing result (computation only, excluding printing time)
    // CHECK: {{[0-9]+\.[0-9]+}}
    vector.print %computation_time : f64

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %Bpack : memref<?x?xbf16>
    memref.dealloc %A : memref<?x?xbf16>
    return
  }
}

