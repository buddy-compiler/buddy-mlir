// RUN: buddy-opt %s \
// RUN:     -matmul-parallel-vectorization-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-parallelize \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @rtclock() -> f64
  func.func private @printMemrefF64(memref<*xf64>)

  func.func private @report_case(%m: index, %n: index, %k: index, %time: f64) {
    %buffer = memref.alloca() : memref<4xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %m_i64 = arith.index_cast %m : index to i64
    %m_f64 = arith.sitofp %m_i64 : i64 to f64
    %n_i64 = arith.index_cast %n : index to i64
    %n_f64 = arith.sitofp %n_i64 : i64 to f64
    %k_i64 = arith.index_cast %k : index to i64
    %k_f64 = arith.sitofp %k_i64 : i64 to f64
    memref.store %m_f64, %buffer[%c0] : memref<4xf64>
    memref.store %n_f64, %buffer[%c1] : memref<4xf64>
    memref.store %k_f64, %buffer[%c2] : memref<4xf64>
    memref.store %time, %buffer[%c3] : memref<4xf64>
    %cast = memref.cast %buffer : memref<4xf64> to memref<*xf64>
    //[M, N, K, avg_seconds]
    call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
    return
  }

  func.func private @run_matmul(%m: index, %n: index, %k: index) -> f64 {
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %three = arith.constant 3.0 : f32

    %A = memref.alloc(%m, %k) : memref<?x?xf32>
    %B = memref.alloc(%k, %n) : memref<?x?xf32>
    %C = memref.alloc(%m, %n) : memref<?x?xf32>

    linalg.fill ins(%one : f32) outs(%A : memref<?x?xf32>)
    linalg.fill ins(%two : f32) outs(%B : memref<?x?xf32>)
    linalg.fill ins(%three : f32) outs(%C : memref<?x?xf32>)

    %start = call @rtclock() : () -> f64
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
    %end = call @rtclock() : () -> f64

    %elapsed = arith.subf %end, %start : f64

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>

    return %elapsed : f64
  }

  func.func private @perf_case(%m: index, %n: index, %k: index) {
    %time = call @run_matmul(%m, %n, %k) : (index, index, index) -> f64
    // print formart: [M, N, K, seconds]
    call @report_case(%m, %n, %k, %time) : (index, index, index, f64) -> ()
    return
  }

  func.func @main() {
    %m_prefill = arith.constant 1024 : index
    %n_prefill_0 = arith.constant 256 : index
    %n_prefill_1 = arith.constant 1536 : index
    %n_prefill_2 = arith.constant 8960 : index
    %n_prefill_3 = arith.constant 151936 : index
    %k_prefill_0 = arith.constant 1536 : index
    %k_prefill_1 = arith.constant 8960 : index

    %m_decode = arith.constant 1 : index
    %n_decode_0 = arith.constant 256 : index
    %n_decode_1 = arith.constant 1536 : index
    %n_decode_2 = arith.constant 8960 : index
    %n_decode_3 = arith.constant 151936 : index

    // Prefill cases
    call @perf_case(%m_prefill, %n_prefill_0, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: Unranked Memref base@
    // CHECK-NEXT: [1024,  256,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_prefill, %n_prefill_1, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1024,  1536,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_prefill, %n_prefill_1, %k_prefill_1) : (index, index, index) -> ()
    // CHECK: [1024,  1536,  8960,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_prefill, %n_prefill_2, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1024,  8960,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_prefill, %n_prefill_3, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1024,  151936,  1536,  {{[0-9]+\.[0-9]+}}]

    // Decode cases
    call @perf_case(%m_decode, %n_decode_0, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1,  256,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_decode, %n_decode_1, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1,  1536,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_decode, %n_decode_1, %k_prefill_1) : (index, index, index) -> ()
    // CHECK: [1,  1536,  8960,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_decode, %n_decode_2, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1,  8960,  1536,  {{[0-9]+\.[0-9]+}}]
    call @perf_case(%m_decode, %n_decode_3, %k_prefill_0) : (index, index, index) -> ()
    // CHECK: [1,  151936,  1536,  {{[0-9]+\.[0-9]+}}]

    return
  }
}
