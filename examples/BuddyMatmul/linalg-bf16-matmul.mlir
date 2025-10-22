// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e linalg_main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// Regular linalg.matmul for performance comparison
module {
  // External functions for timing and printing
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  // Regular linalg matmul kernel for comparison
  // linalg.matmul automatically promotes bf16 inputs to f32 for computation
  func.func @linalg_bf16_matmul(
      %A: memref<?x?xbf16>,     // [M x K], row-major
      %B: memref<?x?xbf16>,     // [K x N], row-major
      %C: memref<?x?xf32>,      // [M x N], row-major
      %M: index, %N: index, %K: index) {

    // Direct bf16 to f32 matmul: linalg.matmul performs numeric casting
    // on the operands to the inner multiply, promoting them to the same
    // data type as the accumulator/output (f32 in this case)
    linalg.matmul ins(%A, %B : memref<?x?xbf16>, memref<?x?xbf16>)
                  outs(%C : memref<?x?xf32>)

    return
  }

  // Performance test with MLIR-level timing: same size as AMX version for comparison.
  func.func @linalg_main() {
    %c0  = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index

    // Test with 512x2048x1024 matrices: A[512x1024] × B[1024x2048] = C[512x2048] (same as AMX version)
    // Allocate matrices
    %A = memref.alloc(%c512, %c1024) : memref<?x?xbf16>      // 512x1024
    %B = memref.alloc(%c1024, %c2048) : memref<?x?xbf16>      // 1024x2048
    %C = memref.alloc(%c512, %c2048) : memref<?x?xf32>       // 512x2048

    // Initialize A = 1.0bf16, B = 1.0bf16, C = 0.0f32
    %one_bf16 = arith.constant 1.0 : bf16
    %zero_f32 = arith.constant 0.0 : f32

    linalg.fill ins(%one_bf16 : bf16) outs(%A : memref<?x?xbf16>)
    linalg.fill ins(%one_bf16 : bf16) outs(%B : memref<?x?xbf16>)
    linalg.fill ins(%zero_f32 : f32) outs(%C : memref<?x?xf32>)

    // Start timing
    %t_start = call @rtclock() : () -> f64

    // Call linalg kernel
    call @linalg_bf16_matmul(%A, %B, %C, %c512, %c2048, %c1024)
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
    memref.dealloc %B : memref<?x?xbf16>
    memref.dealloc %A : memref<?x?xbf16>
    return
  }
}
