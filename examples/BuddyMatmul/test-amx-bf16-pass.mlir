// REQUIRES: has_amx
// RUN: buddy-opt %s -matmul-amx \
// RUN:     --llvm-request-c-wrappers \
// RUN:     -convert-linalg-to-loops \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm="enable-amx" \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN:   mlir-translate --mlir-to-llvmir > %t.ll
// RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+amx-bf16,+amx-tile,+amx-int8 %t.ll -filetype=obj -o %t.o
// RUN: clang -c %S/amx-wrapper.c -o %t-wrapper.o
// RUN: clang++ %t-wrapper.o %t.o -o %t.exe \
// RUN:     -L%mlir_runner_utils_dir -lmlir_runner_utils -lmlir_c_runner_utils -lmlir_float16_utils -lpthread \
// RUN:     -Wl,-rpath,%mlir_runner_utils_dir
// RUN: %t.exe | FileCheck %s
//
// AMX BF16 Matrix Multiplication Example (using -matmul-amx pass)
// ===============================================================
// This file demonstrates automatic conversion from linalg.matmul (bf16×bf16→f32)
// to Intel AMX BF16 tile operations using the unified -matmul-amx pass.
// The pass automatically detects data types (BF16 or INT8) and applies
// the appropriate AMX conversion.
//
// Matrix dimensions: A[512×512] × B[512×512] = C[512×512]
// (Same as amx-bf16-matmul.mlir for comparison)
//
// Data types: A=bf16, B=bf16, C=f32
//
// Initialization pattern (cyclic 1-9):
//   A[m,k] = ((m*K + k) % 9) + 1  → values in range [1, 9]
//   B[k,n] = ((k*N + n) % 9) + 1  → values in range [1, 9]
//
// Expected result:
//   C[0,0] = sum_{k=0}^{511} A[0,k] * B[k,0]
//   With cyclic 1-9 pattern, C[0,0] should be 11439 (same as amx-bf16-matmul.mlir)
//
// Expected output verification (512×512×512 matmul):
// Output order: matrix C first, then timing
// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 512] strides = [512, 1] data =
// CHECK-NEXT: [
// CHECK: [11439{{(, [0-9]+)*}}
// CHECK: {{[0-9]+\.[0-9]+}}

module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  // Initialize A matrix (512×512) with cyclic values 1-9
  // A[m, k] = ((m*K + k) % 9) + 1
  func.func @init_matrix_A_bf16(%A: memref<512x512xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c512 = arith.constant 512 : index

    scf.for %m = %c0 to %c512 step %c1 {
      scf.for %k = %c0 to %c512 step %c1 {
        // Compute linear index: m*512 + k
        %m_times_K = arith.muli %m, %c512 : index
        %linear_idx = arith.addi %m_times_K, %k : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to bf16: index -> i32 -> f32 -> bf16
        %val_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %val_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16
        memref.store %val_bf16, %A[%m, %k] : memref<512x512xbf16>
      }
    }
    return
  }

  // Initialize B matrix (512×512) with cyclic values 1-9
  // B[k, n] = ((k*N + n) % 9) + 1
  func.func @init_matrix_B_bf16(%B: memref<512x512xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c512 = arith.constant 512 : index

    scf.for %k = %c0 to %c512 step %c1 {
      scf.for %n = %c0 to %c512 step %c1 {
        // Compute linear index: k*512 + n
        %k_times_N = arith.muli %k, %c512 : index
        %linear_idx = arith.addi %k_times_N, %n : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to bf16: index -> i32 -> f32 -> bf16
        %val_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %val_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16
        memref.store %val_bf16, %B[%k, %n] : memref<512x512xbf16>
      }
    }
    return
  }

  // Main entry point for testing LinalgToAMX BF16 conversion
  func.func @amx_main() {
    %c0 = arith.constant 0 : index
    %cst_f32 = arith.constant 0.0 : f32

    // Allocate matrices with AMX-compatible dimensions (same as amx-bf16-matmul.mlir)
    // A: 512×512 (M must be multiple of 16, K must be multiple of 32 for BF16)
    // B: 512×512 (K must be multiple of 32, N must be multiple of 16)
    // C: 512×512 (M×N result)
    %A = memref.alloc() : memref<512x512xbf16>
    %B = memref.alloc() : memref<512x512xbf16>
    %C = memref.alloc() : memref<512x512xf32>

    // Initialize matrices with cyclic 1-9 values (same pattern as amx-bf16-matmul.mlir)
    func.call @init_matrix_A_bf16(%A) : (memref<512x512xbf16>) -> ()
    func.call @init_matrix_B_bf16(%B) : (memref<512x512xbf16>) -> ()
    linalg.fill ins(%cst_f32 : f32) outs(%C : memref<512x512xf32>)

    // Get timing
    %start_time = func.call @rtclock() : () -> f64

    // This linalg.matmul will be converted by the unified -matmul-amx pass
    // to use Intel AMX BF16 tile operations (amx.tile_mulf)
    // The pass automatically detects BF16 data types and handles VNNI packing
    linalg.matmul ins(%A, %B : memref<512x512xbf16>, memref<512x512xbf16>)
                  outs(%C : memref<512x512xf32>)

    %end_time = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %end_time, %start_time : f64

    // Print result matrix C (same order as amx-bf16-matmul.mlir)
    %C_unranked = memref.cast %C : memref<512x512xf32> to memref<*xf32>
    func.call @printMemrefF32(%C_unranked) : (memref<*xf32>) -> ()

    // Print timing
    vector.print %elapsed : f64

    // Clean up
    memref.dealloc %A : memref<512x512xbf16>
    memref.dealloc %B : memref<512x512xbf16>
    memref.dealloc %C : memref<512x512xf32>

    return
  }
}
