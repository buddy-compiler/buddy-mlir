// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Test: 1D f32 elementwise add (A + B).
func.func @case_linalg_to_vir_to_vector() -> i32 {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index

  %A = memref.alloc(%c10) : memref<?xf32>
  %B = memref.alloc(%c10) : memref<?xf32>
  %C = memref.alloc(%c10) : memref<?xf32>

  affine.for %i = 0 to 10 {
    %ii64 = arith.index_cast %i : index to i64
    %a = arith.sitofp %ii64 : i64 to f32
    %two = arith.constant 2.0 : f32
    %b = arith.mulf %a, %two : f32
    memref.store %a, %A[%i] : memref<?xf32>
    memref.store %b, %B[%i] : memref<?xf32>
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A, %B : memref<?xf32>, memref<?xf32>)
      outs(%C : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  }

  %printed = memref.cast %C : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  3,  6,  9,  12,  15,  18,  21,  24,  27]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: 1D i32 elementwise add (A + B).
func.func @case_linalg_to_vir_to_vector_1d_i32() -> i32 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 10 : index

  %A = memref.alloc(%n) : memref<?xi32>
  %B = memref.alloc(%n) : memref<?xi32>
  %C = memref.alloc(%n) : memref<?xi32>

  affine.for %i = 0 to 10 {
    %ii = arith.index_cast %i : index to i32
    %two = arith.constant 2 : i32
    %b = arith.muli %ii, %two : i32
    memref.store %ii, %A[%i] : memref<?xi32>
    memref.store %b, %B[%i] : memref<?xi32>
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A, %B : memref<?xi32>, memref<?xi32>)
      outs(%C : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
  }

  %printed = memref.cast %C : memref<?xi32> to memref<*xi32>
  call @printMemrefI32(%printed) : (memref<*xi32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: 2D f32 elementwise add.
func.func @case_linalg_to_vir_to_vector_2d_f32() -> i32 {
  %n = arith.constant 8 : index

  %A = memref.alloc(%n) : memref<4x?xf32>
  %B = memref.alloc(%n) : memref<4x?xf32>
  %C = memref.alloc(%n) : memref<4x?xf32>

  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %ii = arith.index_cast %i : index to i64
      %jj = arith.index_cast %j : index to i64
      %ten = arith.constant 10 : i64
      %base = arith.muli %ii, %ten : i64
      %v = arith.addi %base, %jj : i64
      %a = arith.sitofp %v : i64 to f32
      %two = arith.constant 2.0 : f32
      %b = arith.mulf %a, %two : f32
      memref.store %a, %A[%i, %j] : memref<4x?xf32>
      memref.store %b, %B[%i, %j] : memref<4x?xf32>
    }
  }

  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A, %B : memref<4x?xf32>, memref<4x?xf32>)
      outs(%C : memref<4x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  }

  %printed = memref.cast %C : memref<4x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[4, 8\] strides = \[8, 1\] data =}}
  // CHECK{LITERAL}: [[0, 3, 6, 9, 12, 15, 18, 21],
  // CHECK{LITERAL}: [30, 33, 36, 39, 42, 45, 48, 51],
  // CHECK{LITERAL}: [60, 63, 66, 69, 72, 75, 78, 81],
  // CHECK{LITERAL}: [90, 93, 96, 99, 102, 105, 108, 111]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: scalar broadcast add on f32 tensor.
func.func @case_linalg_to_vir_to_vector_scalar_broadcast_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %C = memref.alloc(%n) : memref<?xf32>

  affine.for %i = 0 to 10 {
    %ii64 = arith.index_cast %i : index to i64
    %a = arith.sitofp %ii64 : i64 to f32
    memref.store %a, %A[%i] : memref<?xf32>
  }

  %cst = arith.constant 3.0 : f32
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%C : memref<?xf32>) {
    ^bb0(%a: f32, %c: f32):
      %sum = arith.addf %a, %cst : f32
      linalg.yield %sum : f32
  }

  %printed = memref.cast %C : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [3,  4,  5,  6,  7,  8,  9,  10,  11,  12]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: mask generation + select on f32.
func.func @case_linalg_to_vir_to_vector_mask_select_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xi32>
  %B = memref.alloc(%n) : memref<?xf32>
  %Z = memref.alloc(%n) : memref<?xf32>
  %M = memref.alloc(%n) : memref<?xi1>
  %O = memref.alloc(%n) : memref<?xf32>

  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%Z : memref<?xf32>)

  affine.for %i = 0 to 10 {
    %ii = arith.index_cast %i : index to i32
    %bf = arith.sitofp %ii : i32 to f32
    memref.store %ii, %A[%i] : memref<?xi32>
    memref.store %bf, %B[%i] : memref<?xf32>
  }

  %five = arith.constant 5 : i32
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xi32>)
      outs(%M : memref<?xi1>) {
    ^bb0(%a: i32, %m: i1):
      %pred = arith.cmpi slt, %a, %five : i32
      linalg.yield %pred : i1
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%M, %B, %Z : memref<?xi1>, memref<?xf32>, memref<?xf32>)
      outs(%O : memref<?xf32>) {
    ^bb0(%m: i1, %b: f32, %z: f32, %o: f32):
      %v = arith.select %m, %b, %z : f32
      linalg.yield %v : f32
  }

  %printed = memref.cast %O : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  1,  2,  3,  4,  0,  0,  0,  0,  0]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func @main() -> i32 {
  %res_case_linalg_to_vir_to_vector = call @case_linalg_to_vir_to_vector() : () -> i32
  %res_case_linalg_to_vir_to_vector_1d_i32 = call @case_linalg_to_vir_to_vector_1d_i32() : () -> i32
  %res_case_linalg_to_vir_to_vector_2d_f32 = call @case_linalg_to_vir_to_vector_2d_f32() : () -> i32
  %res_case_linalg_to_vir_to_vector_scalar_broadcast_f32 = call @case_linalg_to_vir_to_vector_scalar_broadcast_f32() : () -> i32
  %res_case_linalg_to_vir_to_vector_mask_select_f32 = call @case_linalg_to_vir_to_vector_mask_select_f32() : () -> i32
  %ret = arith.constant 0 : i32
  return %ret : i32
}
