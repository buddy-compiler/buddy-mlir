// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --expand-strided-metadata --lower-affine --convert-math-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse | FileCheck %s --check-prefix=CHECK-VEC-MATH
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse | FileCheck %s --check-prefix=CHECK-VEC-CAST

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Test: f32 arithmetic chain (add/mul/sub/div/neg).
// Exercises addf/mulf/subf/divf/negf inside linalg.generic.
func.func @case_linalg_generic_to_vir_to_vector_arith_chain_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>
  %C = memref.alloc(%n) : memref<?xf32>

  // A[i] = i+1, B[i] = 2*(i+1)
  affine.for %i = 0 to 10 {
    %ii64 = arith.index_cast %i : index to i64
    %one = arith.constant 1 : i64
    %v = arith.addi %ii64, %one : i64
    %a = arith.sitofp %v : i64 to f32
    %two = arith.constant 2.0 : f32
    %b = arith.mulf %a, %two : f32
    memref.store %a, %A[%i] : memref<?xf32>
    memref.store %b, %B[%i] : memref<?xf32>
  }

  // out = -(((a+b)*2 - a) / 2) = -(0.5*a + b)
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A, %B : memref<?xf32>, memref<?xf32>)
      outs(%C : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      %two = arith.constant 2.0 : f32
      %t0 = arith.mulf %sum, %two : f32
      %t1 = arith.subf %t0, %a : f32
      %t2 = arith.divf %t1, %two : f32
      %t3 = arith.negf %t2 : f32
      linalg.yield %t3 : f32
  }

  %printed = memref.cast %C : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // Expected with A=i+1, B=2(i+1): out = -2.5*(i+1)
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [-2.5,  -5,  -7.5,  -10,  -12.5,  -15,  -17.5,  -20,  -22.5,  -25]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: i32 arithmetic chain (addi/muli/subi).
// Exercises addi/muli/subi inside linalg.generic.
func.func @case_linalg_generic_to_vir_to_vector_arith_chain_i32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xi32>
  %B = memref.alloc(%n) : memref<?xi32>
  %C = memref.alloc(%n) : memref<?xi32>

  // A[i] = i, B[i] = 2*i
  affine.for %i = 0 to 10 {
    %ii = arith.index_cast %i : index to i32
    %two = arith.constant 2 : i32
    %b = arith.muli %ii, %two : i32
    memref.store %ii, %A[%i] : memref<?xi32>
    memref.store %b, %B[%i] : memref<?xi32>
  }

  // out = ((a+b)*3) - a = 2*a + 3*b = 8*i
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A, %B : memref<?xi32>, memref<?xi32>)
      outs(%C : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %sum = arith.addi %a, %b : i32
      %three = arith.constant 3 : i32
      %t0 = arith.muli %sum, %three : i32
      %t1 = arith.subi %t0, %a : i32
      linalg.yield %t1 : i32
  }

  %printed = memref.cast %C : memref<?xi32> to memref<*xi32>
  call @printMemrefI32(%printed) : (memref<*xi32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  8,  16,  24,  32,  40,  48,  56,  64,  72]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: i32 bitwise and shift chain.
// Exercises andi/ori/xori + shifts inside linalg.generic.
func.func @case_linalg_generic_to_vir_to_vector_bitwise_shift_i32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xi32>
  %B = memref.alloc(%n) : memref<?xi32>

  // A[i] = i
  affine.for %i = 0 to 10 {
    %ii = arith.index_cast %i : index to i32
    memref.store %ii, %A[%i] : memref<?xi32>
  }

  // B[i] = ((((A[i] << 1) | 1) xor 15) & 15) >> 1
  // With i=0..9 => [7, 6, 5, 4, 3, 2, 1, 0, 7, 6]
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xi32>)
      outs(%B : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32):
      %one = arith.constant 1 : i32
      %mask = arith.constant 15 : i32
      %shl = arith.shli %a, %one : i32
      %or = arith.ori %shl, %one : i32
      %x = arith.xori %or, %mask : i32
      %and = arith.andi %x, %mask : i32
      %shr = arith.shrui %and, %one : i32
      linalg.yield %shr : i32
  }

  %printed = memref.cast %B : memref<?xi32> to memref<*xi32>
  call @printMemrefI32(%printed) : (memref<*xi32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [7,  6,  5,  4,  3,  2,  1,  0,  7,  6]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: f32 cmp + select (ReLU-like).
// Exercises cmpf/select inside linalg.generic (ReLU).
func.func @case_linalg_generic_to_vir_to_vector_cmp_select_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>

  // A = [-5, -4, ..., 4]
  affine.for %i = 0 to 10 {
    %ii32 = arith.index_cast %i : index to i32
    %five = arith.constant 5 : i32
    %x = arith.subi %ii32, %five : i32
    %xf = arith.sitofp %x : i32 to f32
    memref.store %xf, %A[%i] : memref<?xf32>
  }

  // B = relu(A)
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%B : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %zero = arith.constant 0.0 : f32
      %pred = arith.cmpf ogt, %a, %zero : f32
      %out = arith.select %pred, %a, %zero : f32
      linalg.yield %out : f32
  }

  %printed = memref.cast %B : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  0,  0,  0,  0,  0,  1,  2,  3,  4]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: f32 min/max clamp to [-1, 1].
// Exercises minimumf/maximumf inside linalg.generic (clamp to [-1, 1]).
func.func @case_linalg_generic_to_vir_to_vector_minmax_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>

  // A = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
  affine.for %i = 0 to 10 {
    %ii32 = arith.index_cast %i : index to i32
    %two = arith.constant 2 : i32
    %x = arith.subi %ii32, %two : i32
    %xf = arith.sitofp %x : i32 to f32
    %half = arith.constant 0.5 : f32
    %v = arith.mulf %xf, %half : f32
    memref.store %v, %A[%i] : memref<?xf32>
  }

  // B = clamp(A, -1, 1) = min(max(A, -1), 1)
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%B : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %lo = arith.constant -1.0 : f32
      %hi = arith.constant 1.0 : f32
      %m0 = arith.maximumf %a, %lo : f32
      %m1 = arith.minimumf %m0, %hi : f32
      linalg.yield %m1 : f32
  }

  %printed = memref.cast %B : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [-1,  -0.5,  0,  0.5,  1,  1,  1,  1,  1,  1]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// Test: f32 math.exp elementwise.
func.func @case_linalg_generic_to_vir_to_vector_math_exp_f32() -> i32 {
  %n = arith.constant 10 : index
  %zero = arith.constant 0.0 : f32
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>

  linalg.fill ins(%zero : f32) outs(%A : memref<?xf32>)
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%B : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %e = math.exp %a : f32
      linalg.yield %e : f32
  }

  %printed = memref.cast %B : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// CHECK-VEC-MATH-LABEL: func.func @case_linalg_generic_to_vir_to_vector_math_exp_f32
// CHECK-VEC-MATH: math.exp {{.*}} : vector<4xf32>
// CHECK-VEC-MATH: math.exp {{.*}} : f32

// Test: f32 math.sqrt elementwise.
func.func @case_linalg_generic_to_vir_to_vector_math_sqrt_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>

  // A = [0, 1, 4, 9, ..., 81]
  affine.for %i = 0 to 10 {
    %ii32 = arith.index_cast %i : index to i32
    %f = arith.sitofp %ii32 : i32 to f32
    %sq = arith.mulf %f, %f : f32
    memref.store %sq, %A[%i] : memref<?xf32>
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%B : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %s = math.sqrt %a : f32
      linalg.yield %s : f32
  }

  %printed = memref.cast %B : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// CHECK-VEC-MATH-LABEL: func.func @case_linalg_generic_to_vir_to_vector_math_sqrt_f32
// CHECK-VEC-MATH: math.sqrt {{.*}} : vector<4xf32>
// CHECK-VEC-MATH: math.sqrt {{.*}} : f32

// Test: f16->f32 cast vectorization.
func.func @case_linalg_generic_to_vir_to_vector_cast_extf_f16_f32() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf16>
  %B = memref.alloc(%n) : memref<?xf32>

  affine.for %i = 0 to 10 {
    %ii32 = arith.index_cast %i : index to i32
    %v = arith.sitofp %ii32 : i32 to f16
    memref.store %v, %A[%i] : memref<?xf16>
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf16>)
      outs(%B : memref<?xf32>) {
    ^bb0(%a: f16, %b: f32):
      %x = arith.extf %a : f16 to f32
      linalg.yield %x : f32
  }

  %printed = memref.cast %B : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// CHECK-VEC-CAST-LABEL: func.func @case_linalg_generic_to_vir_to_vector_cast_extf_f16_f32
// CHECK-VEC-CAST: arith.extf {{.*}} : vector<4xf16> to vector<4xf32>
// CHECK-VEC-CAST: arith.extf {{.*}} : f16 to f32

// Test: f32->f16 cast vectorization, then f16->f32 for checking.
func.func @case_linalg_generic_to_vir_to_vector_cast_truncf_f32_f16() -> i32 {
  %n = arith.constant 10 : index
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf16>
  %C = memref.alloc(%n) : memref<?xf32>

  affine.for %i = 0 to 10 {
    %ii32 = arith.index_cast %i : index to i32
    %f = arith.sitofp %ii32 : i32 to f32
    %half = arith.constant 0.5 : f32
    %v = arith.addf %f, %half : f32
    memref.store %v, %A[%i] : memref<?xf32>
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%A : memref<?xf32>)
      outs(%B : memref<?xf16>) {
    ^bb0(%a: f32, %b: f16):
      %x = arith.truncf %a : f32 to f16
      linalg.yield %x : f16
  }

  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%B : memref<?xf16>)
      outs(%C : memref<?xf32>) {
    ^bb0(%a: f16, %c: f32):
      %x = arith.extf %a : f16 to f32
      linalg.yield %x : f32
  }

  %printed = memref.cast %C : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%printed) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\] strides = \[1\] data =}}
  // CHECK{LITERAL}: [0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5]

  %ret = arith.constant 0 : i32
  return %ret : i32
}

// CHECK-VEC-CAST-LABEL: func.func @case_linalg_generic_to_vir_to_vector_cast_truncf_f32_f16
// CHECK-VEC-CAST: arith.truncf {{.*}} : vector<4xf32> to vector<4xf16>
// CHECK-VEC-CAST: arith.truncf {{.*}} : f32 to f16

func.func @main() -> i32 {
  %res_case_linalg_generic_to_vir_to_vector_arith_chain_f32 = call @case_linalg_generic_to_vir_to_vector_arith_chain_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_arith_chain_i32 = call @case_linalg_generic_to_vir_to_vector_arith_chain_i32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_bitwise_shift_i32 = call @case_linalg_generic_to_vir_to_vector_bitwise_shift_i32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_cmp_select_f32 = call @case_linalg_generic_to_vir_to_vector_cmp_select_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_minmax_f32 = call @case_linalg_generic_to_vir_to_vector_minmax_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_math_exp_f32 = call @case_linalg_generic_to_vir_to_vector_math_exp_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_math_sqrt_f32 = call @case_linalg_generic_to_vir_to_vector_math_sqrt_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_cast_extf_f16_f32 = call @case_linalg_generic_to_vir_to_vector_cast_extf_f16_f32() : () -> i32
  %res_case_linalg_generic_to_vir_to_vector_cast_truncf_f32_f16 = call @case_linalg_generic_to_vir_to_vector_cast_truncf_f32_f16() : () -> i32
  %ret = arith.constant 0 : i32
  return %ret : i32
}
