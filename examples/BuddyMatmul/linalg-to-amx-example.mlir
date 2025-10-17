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
// RUN: %t.exe


// Example demonstrating automatic conversion from linalg.matmul to AMX operations
// This file shows how the LinalgToAMX pass converts high-level linalg operations
// to low-level AMX tile operations for optimal performance on Intel AMX hardware.

module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  // Main entry point for testing LinalgToAMX conversion
  func.func @amx_main() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    // Constants for initialization
    %cst_bf16 = arith.constant 1.0 : bf16
    %cst_f32 = arith.constant 0.0 : f32

    // Allocate matrices with AMX-compatible dimensions
    %A = memref.alloc() : memref<512x1024xbf16>
    %B = memref.alloc() : memref<1024x2048xbf16>
    %C = memref.alloc() : memref<512x2048xf32>

    // Initialize matrices
    linalg.fill ins(%cst_bf16 : bf16) outs(%A : memref<512x1024xbf16>)
    linalg.fill ins(%cst_bf16 : bf16) outs(%B : memref<1024x2048xbf16>)
    linalg.fill ins(%cst_f32 : f32) outs(%C : memref<512x2048xf32>)

    // Get timing
    %start_time = func.call @rtclock() : () -> f64

    // this will be converted by LinalgToAMX pass
    linalg.matmul ins(%A, %B : memref<512x1024xbf16>, memref<1024x2048xbf16>)
                  outs(%C : memref<512x2048xf32>)

    %end_time = func.call @rtclock() : () -> f64
    %elapsed = arith.subf %end_time, %start_time : f64

    // Print results
    vector.print %elapsed : f64

    // Print a sample of the result matrix
    %sample = memref.load %C[%c0, %c0] : memref<512x2048xf32>
    vector.print %sample : f32

    // Clean up
    memref.dealloc %A : memref<512x1024xbf16>
    memref.dealloc %B : memref<1024x2048xbf16>
    memref.dealloc %C : memref<512x2048xf32>

    return
  }
}
