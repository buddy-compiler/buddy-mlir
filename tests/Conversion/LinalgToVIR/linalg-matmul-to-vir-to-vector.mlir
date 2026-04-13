// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// Test: f32 matmul basic case.
func.func @case_linalg_matmul_to_vir_to_vector_f32() -> i32 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 8 : index

  %A = memref.alloc() : memref<2x4xf32>
  %B = memref.alloc(%n) : memref<4x?xf32>
  %C = memref.alloc(%n) : memref<2x?xf32>

  // Fill A with A[i,k] = (i*10 + k)
  affine.for %i = 0 to 2 {
    affine.for %k = 0 to 4 {
      %ii = arith.index_cast %i : index to i64
      %kk = arith.index_cast %k : index to i64
      %ten = arith.constant 10 : i64
      %base = arith.muli %ii, %ten : i64
      %v = arith.addi %base, %kk : i64
      %a = arith.sitofp %v : i64 to f32
      memref.store %a, %A[%i, %k] : memref<2x4xf32>
    }
  }

  // Fill B with B[k,j] = (k*100 + j)
  affine.for %k = 0 to 4 {
    affine.for %j = 0 to 8 {
      %kk = arith.index_cast %k : index to i64
      %jj = arith.index_cast %j : index to i64
      %hund = arith.constant 100 : i64
      %base = arith.muli %kk, %hund : i64
      %v = arith.addi %base, %jj : i64
      %b = arith.sitofp %v : i64 to f32
      memref.store %b, %B[%k, %j] : memref<4x?xf32>
    }
  }

  // Zero-initialize C.
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 8 {
      %z = arith.constant 0.0 : f32
      memref.store %z, %C[%i, %j] : memref<2x?xf32>
    }
  }

  linalg.matmul ins(%A, %B : memref<2x4xf32>, memref<4x?xf32>)
      outs(%C : memref<2x?xf32>)

  %printed = memref.cast %C : memref<2x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()

  // Expected:
  // C[i,j] = sum_{k=0..3} (i*10+k) * (k*100 + j)
  // Row 0: [1400, 1406, 1412, 1418, 1424, 1430, 1436, 1442]
  // Row 1: [7400, 7446, 7492, 7538, 7584, 7630, 7676, 7722]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[2, 8\] strides = \[8, 1\] data =}}
  // CHECK{LITERAL}: [[1400, 1406, 1412, 1418, 1424, 1430, 1436, 1442],
  // CHECK{LITERAL}: [7400, 7446, 7492, 7538, 7584, 7630, 7676, 7722]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: f32 matmul accumulate into pre-initialized C.
func.func @case_linalg_matmul_to_vir_to_vector_accumulate_f32() -> i32 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 8 : index

  %A = memref.alloc() : memref<2x4xf32>
  %B = memref.alloc(%n) : memref<4x?xf32>
  %C = memref.alloc(%n) : memref<2x?xf32>

  // Fill A with A[i,k] = (i*10 + k)
  affine.for %i = 0 to 2 {
    affine.for %k = 0 to 4 {
      %ii = arith.index_cast %i : index to i64
      %kk = arith.index_cast %k : index to i64
      %ten = arith.constant 10 : i64
      %base = arith.muli %ii, %ten : i64
      %v = arith.addi %base, %kk : i64
      %a = arith.sitofp %v : i64 to f32
      memref.store %a, %A[%i, %k] : memref<2x4xf32>
    }
  }

  // Fill B with B[k,j] = (k*100 + j)
  affine.for %k = 0 to 4 {
    affine.for %j = 0 to 8 {
      %kk = arith.index_cast %k : index to i64
      %jj = arith.index_cast %j : index to i64
      %hund = arith.constant 100 : i64
      %base = arith.muli %kk, %hund : i64
      %v = arith.addi %base, %jj : i64
      %b = arith.sitofp %v : i64 to f32
      memref.store %b, %B[%k, %j] : memref<4x?xf32>
    }
  }

  // Initialize C to 1.0 (accumulate behavior).
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 8 {
      %one = arith.constant 1.0 : f32
      memref.store %one, %C[%i, %j] : memref<2x?xf32>
    }
  }

  linalg.matmul ins(%A, %B : memref<2x4xf32>, memref<4x?xf32>)
      outs(%C : memref<2x?xf32>)

  %printed = memref.cast %C : memref<2x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()

  // Same as linalg-matmul-to-vir-to-vector-f32.mlir, but +1 everywhere.
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[2, 8\] strides = \[8, 1\] data =}}
  // CHECK{LITERAL}: [[1401, 1407, 1413, 1419, 1425, 1431, 1437, 1443],
  // CHECK{LITERAL}: [7401, 7447, 7493, 7539, 7585, 7631, 7677, 7723]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: f32 matmul tail handling (non-multiple width).
func.func @case_linalg_matmul_to_vir_to_vector_tail_f32() -> i32 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 10 : index

  %A = memref.alloc() : memref<1x2xf32>
  %B = memref.alloc(%n) : memref<2x?xf32>
  %C = memref.alloc(%n) : memref<1x?xf32>

  // A = [1, 2]
  %a0 = arith.constant 1.0 : f32
  %a1 = arith.constant 2.0 : f32
  memref.store %a0, %A[%c0, %c0] : memref<1x2xf32>
  %c1 = arith.constant 1 : index
  memref.store %a1, %A[%c0, %c1] : memref<1x2xf32>

  // B[0, j] = j
  // B[1, j] = 100 + j
  affine.for %j = 0 to 10 {
    %jj = arith.index_cast %j : index to i64
    %b0 = arith.sitofp %jj : i64 to f32
    %hund = arith.constant 100 : i64
    %v1 = arith.addi %hund, %jj : i64
    %b1 = arith.sitofp %v1 : i64 to f32
    memref.store %b0, %B[%c0, %j] : memref<2x?xf32>
    memref.store %b1, %B[%c1, %j] : memref<2x?xf32>
  }

  // Zero-initialize C.
  affine.for %j = 0 to 10 {
    %z = arith.constant 0.0 : f32
    memref.store %z, %C[%c0, %j] : memref<1x?xf32>
  }

  linalg.matmul ins(%A, %B : memref<1x2xf32>, memref<2x?xf32>)
      outs(%C : memref<1x?xf32>)

  %printed = memref.cast %C : memref<1x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()

  // Expected: C[0,j] = 1*j + 2*(100+j) = 200 + 3*j
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[1, 10\] strides = \[10, 1\] data =}}
  // CHECK{LITERAL}: [[200, 203, 206, 209, 212, 215, 218, 221, 224, 227]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func @main() -> i32 {
  %res_case_linalg_matmul_to_vir_to_vector_f32 = call @case_linalg_matmul_to_vir_to_vector_f32() : () -> i32
  %res_case_linalg_matmul_to_vir_to_vector_accumulate_f32 = call @case_linalg_matmul_to_vir_to_vector_accumulate_f32() : () -> i32
  %res_case_linalg_matmul_to_vir_to_vector_tail_f32 = call @case_linalg_matmul_to_vir_to_vector_tail_f32() : () -> i32
  %ret = arith.constant 0 : i32
  return %ret : i32
}
