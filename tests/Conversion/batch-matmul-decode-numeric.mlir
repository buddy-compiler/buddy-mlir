// RUN: buddy-opt %s \
// RUN:     -batch-matmul-vectorization-decode="vector-size=4" \
// RUN:     -batchmatmul-transpose-b-vectorization="vector-size=4" \
// RUN:     -convert-linalg-to-loops -scf-parallel-for-to-nested-fors \
// RUN:     -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// Numeric regression test for the decode batch-matmul pipeline, i.e. pass 1
// (-batch-matmul-vectorization-decode, which only rewrites *default* indexing
// maps) followed by pass 2 (-batchmatmul-transpose-b-vectorization, which
// handles the transposed-B layout GPT-2's Q@K^T uses).  Instead of checking
// pass order or IR structure, we check the numbers the lowered code computes,
// so the test still fails if either pass regresses the result.
//
// The two functions take the *same* A (all ones) and the *same* B (row i holds
// the constant i+1) and differ only in how the batch_matmul indexes B:
//
//   @non_transposed:  C[b,m,n] = sum_k A[b,m,k] * B[b,k,n]   -> [10, 10, 10, 10]
//   @transposed_b:    C[b,m,n] = sum_k A[b,m,k] * B[b,n,k]   -> [4, 8, 12, 16]
//
// Reading the transposed B along the wrong axis (the original bug) turns the
// second case into the first, so the [4, 8, 12, 16] check catches it.

module {
  func.func private @printMemrefF32(memref<*xf32>)

  // B (batch=1, k=4, n=4): default indexing maps.
  func.func @non_transposed(%A: memref<1x1x4xf32>,
                            %B: memref<1x4x4xf32>,
                            %C: memref<1x1x4xf32>) {
    linalg.batch_matmul
      ins(%A, %B : memref<1x1x4xf32>, memref<1x4x4xf32>)
      outs(%C : memref<1x1x4xf32>)
    return
  }

  // B (batch=1, n=4, k=4): transposed B, i.e. the GPT-2 Q@K^T layout.
  func.func @transposed_b(%A: memref<1x1x4xf32>,
                          %B: memref<1x4x4xf32>,
                          %C: memref<1x1x4xf32>) {
    linalg.batch_matmul indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
      ins(%A, %B : memref<1x1x4xf32>, memref<1x4x4xf32>)
      outs(%C : memref<1x1x4xf32>)
    return
  }

  // Fill B[0, i, j] = i + 1 so every row is a distinct constant.
  func.func @fill_b(%B: memref<1x4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1 {
      %i_i32 = arith.index_cast %i : index to i32
      %row = arith.addi %i_i32, %c1_i32 : i32
      %row_f = arith.sitofp %row : i32 to f32
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %row_f, %B[%c0, %i, %j] : memref<1x4x4xf32>
      }
    }
    return
  }

  func.func @main() {
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32

    // --- Non-transposed B ---
    %A_nt = memref.alloc() : memref<1x1x4xf32>
    %B_nt = memref.alloc() : memref<1x4x4xf32>
    %C_nt = memref.alloc() : memref<1x1x4xf32>
    linalg.fill ins(%f1 : f32) outs(%A_nt : memref<1x1x4xf32>)
    call @fill_b(%B_nt) : (memref<1x4x4xf32>) -> ()
    linalg.fill ins(%f0 : f32) outs(%C_nt : memref<1x1x4xf32>)
    call @non_transposed(%A_nt, %B_nt, %C_nt)
      : (memref<1x1x4xf32>, memref<1x4x4xf32>, memref<1x1x4xf32>) -> ()
    %print_nt = memref.cast %C_nt : memref<1x1x4xf32> to memref<*xf32>
    call @printMemrefF32(%print_nt) : (memref<*xf32>) -> ()

    // --- Transposed B ---
    %A_t = memref.alloc() : memref<1x1x4xf32>
    %B_t = memref.alloc() : memref<1x4x4xf32>
    %C_t = memref.alloc() : memref<1x1x4xf32>
    linalg.fill ins(%f1 : f32) outs(%A_t : memref<1x1x4xf32>)
    call @fill_b(%B_t) : (memref<1x4x4xf32>) -> ()
    linalg.fill ins(%f0 : f32) outs(%C_t : memref<1x1x4xf32>)
    call @transposed_b(%A_t, %B_t, %C_t)
      : (memref<1x1x4xf32>, memref<1x4x4xf32>, memref<1x1x4xf32>) -> ()
    %print_t = memref.cast %C_t : memref<1x1x4xf32> to memref<*xf32>
    call @printMemrefF32(%print_t) : (memref<*xf32>) -> ()

    memref.dealloc %C_t : memref<1x1x4xf32>
    memref.dealloc %B_t : memref<1x4x4xf32>
    memref.dealloc %A_t : memref<1x1x4xf32>
    memref.dealloc %C_nt : memref<1x1x4xf32>
    memref.dealloc %B_nt : memref<1x4x4xf32>
    memref.dealloc %A_nt : memref<1x1x4xf32>
    return
  }
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 1, 4] strides = [4, 4, 1] data =
// CHECK: 10,    10,    10,    10
// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 1, 4] strides = [4, 4, 1] data =
// CHECK: 4,{{.*}}8,{{.*}}12,{{.*}}16
