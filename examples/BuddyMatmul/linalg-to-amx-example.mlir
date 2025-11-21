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
// RUN:     -L%mlir_runner_utils_dir -lmlir_runner_utils -lmlir_c_runner_utils -lpthread \
// RUN:     -Wl,-rpath,%mlir_runner_utils_dir
// RUN: %t.exe | FileCheck %s
//
// Expected output verification (512×1024×2048 matmul):
// CHECK: {{[0-9]+\.[0-9]+}}
// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [512, 2048] strides = [2048, 1] data =
// CHECK-NEXT: [
// CHECK: [28953{{(, [0-9]+)*}}
// CHECK: [25903{{(, [0-9]+)*}}


// Example demonstrating automatic conversion from linalg.matmul to AMX operations
// This file shows how the LinalgToAMX pass converts high-level linalg operations
// to low-level AMX tile operations for optimal performance on Intel AMX hardware.

module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  // Initialize A matrix with cyclic values 1-9
  // A[m, k] = ((m*K + k) % 9) + 1
  func.func @init_matrix_A(%A: memref<512x1024xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index

    scf.for %m = %c0 to %c512 step %c1 {
      scf.for %k = %c0 to %c1024 step %c1 {
        // Compute linear index: m*1024 + k
        %m_times_K = arith.muli %m, %c1024 : index
        %linear_idx = arith.addi %m_times_K, %k : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to bf16
        %idx_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %idx_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16

        memref.store %val_bf16, %A[%m, %k] : memref<512x1024xbf16>
      }
    }
    return
  }

  // Initialize B matrix with cyclic values 1-9
  // B[k, n] = ((k*N + n) % 9) + 1
  func.func @init_matrix_B(%B: memref<1024x2048xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    scf.for %k = %c0 to %c1024 step %c1 {
      scf.for %n = %c0 to %c2048 step %c1 {
        // Compute linear index: k*2048 + n
        %k_times_N = arith.muli %k, %c2048 : index
        %linear_idx = arith.addi %k_times_N, %n : index

        // Compute (linear_idx % 9) + 1
        %mod_val = arith.remui %linear_idx, %c9 : index
        %cyclic_val = arith.addi %mod_val, %c1 : index

        // Convert to bf16
        %idx_i32 = arith.index_cast %cyclic_val : index to i32
        %val_f32 = arith.sitofp %idx_i32 : i32 to f32
        %val_bf16 = arith.truncf %val_f32 : f32 to bf16

        memref.store %val_bf16, %B[%k, %n] : memref<1024x2048xbf16>
      }
    }
    return
  }

  // Main entry point for testing LinalgToAMX conversion
  func.func @amx_main() {
    %c0 = arith.constant 0 : index
    %cst_f32 = arith.constant 0.0 : f32

    // Allocate matrices with AMX-compatible dimensions
    %A = memref.alloc() : memref<512x1024xbf16>
    %B = memref.alloc() : memref<1024x2048xbf16>
    %C = memref.alloc() : memref<512x2048xf32>

    // Initialize matrices with cyclic 1-9 values
    func.call @init_matrix_A(%A) : (memref<512x1024xbf16>) -> ()
    func.call @init_matrix_B(%B) : (memref<1024x2048xbf16>) -> ()
    linalg.fill ins(%cst_f32 : f32) outs(%C : memref<512x2048xf32>)

    // Get timing
    %start_time = func.call @rtclock() : () -> f64

    // this will be converted by LinalgToAMX pass
    linalg.matmul ins(%A, %B : memref<512x1024xbf16>, memref<1024x2048xbf16>)
                  outs(%C : memref<512x2048xf32>)

    %end_time = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %end_time, %start_time : f64

    // Print timing
    // CHECK: {{[0-9]+\.[0-9]+}}
    vector.print %elapsed : f64

    // Print result matrix C
    %C_unranked = memref.cast %C : memref<512x2048xf32> to memref<*xf32>
    func.call @printMemrefF32(%C_unranked) : (memref<*xf32>) -> ()

    // Clean up
    memref.dealloc %A : memref<512x1024xbf16>
    memref.dealloc %B : memref<1024x2048xbf16>
    memref.dealloc %C : memref<512x2048xf32>

    return
  }
}
