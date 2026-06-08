// RUN: buddy-opt %s -batchmatmul-optimize="vector-size=4" | FileCheck %s --check-prefix=CHECK-IR
// RUN: buddy-opt %s -batchmatmul-optimize="vector-size=4" \
// RUN:     -convert-linalg-to-loops -scf-parallel-for-to-nested-fors \
// RUN:     -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s --check-prefix=CHECK-OUT

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @batch_m_lt_8_f32(%A: memref<1x3x2xf32>,
                              %B: memref<1x2x5xf32>,
                              %C: memref<1x3x5xf32>) {
    linalg.batch_matmul
      ins(%A, %B : memref<1x3x2xf32>, memref<1x2x5xf32>)
      outs(%C : memref<1x3x5xf32>)
    return
  }

  func.func @batch_i32(%A: memref<1x9x2xi32>,
                       %B: memref<1x2x5xi32>,
                       %C: memref<1x9x5xi32>) {
    linalg.batch_matmul
      ins(%A, %B : memref<1x9x2xi32>, memref<1x2x5xi32>)
      outs(%C : memref<1x9x5xi32>)
    return
  }

  func.func @batch_transpose_b_kept(%A: memref<1x3x2xf32>,
                                    %B: memref<1x5x2xf32>,
                                    %C: memref<1x3x5xf32>) {
    linalg.batch_matmul indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
      ins(%A, %B : memref<1x3x2xf32>, memref<1x5x2xf32>)
      outs(%C : memref<1x3x5xf32>)
    return
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c9 = arith.constant 9 : index

    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f10 = arith.constant 10.0 : f32
    %A_f32 = memref.alloc() : memref<1x3x2xf32>
    %B_f32 = memref.alloc() : memref<1x2x5xf32>
    %C_f32 = memref.alloc() : memref<1x3x5xf32>
    linalg.fill ins(%f1 : f32) outs(%A_f32 : memref<1x3x2xf32>)
    linalg.fill ins(%f2 : f32) outs(%B_f32 : memref<1x2x5xf32>)
    linalg.fill ins(%f10 : f32) outs(%C_f32 : memref<1x3x5xf32>)
    call @batch_m_lt_8_f32(%A_f32, %B_f32, %C_f32)
      : (memref<1x3x2xf32>, memref<1x2x5xf32>, memref<1x3x5xf32>) -> ()

    %print_f32 = memref.cast %C_f32 : memref<1x3x5xf32> to memref<*xf32>
    call @printMemrefF32(%print_f32) : (memref<*xf32>) -> ()

    %i0 = arith.constant 0 : i32
    %i1 = arith.constant 1 : i32
    %A_i32 = memref.alloc() : memref<1x9x2xi32>
    %B_i32 = memref.alloc() : memref<1x2x5xi32>
    %C_i32 = memref.alloc() : memref<1x9x5xi32>
    linalg.fill ins(%i0 : i32) outs(%A_i32 : memref<1x9x2xi32>)
    linalg.fill ins(%i1 : i32) outs(%B_i32 : memref<1x2x5xi32>)
    linalg.fill ins(%i0 : i32) outs(%C_i32 : memref<1x9x5xi32>)

    scf.for %m = %c0 to %c9 step %c1 {
      %m_i32 = arith.index_cast %m : index to i32
      %row_val = arith.addi %m_i32, %i1 : i32
      memref.store %row_val, %A_i32[%c0, %m, %c0] : memref<1x9x2xi32>
      memref.store %row_val, %A_i32[%c0, %m, %c1] : memref<1x9x2xi32>
    }

    call @batch_i32(%A_i32, %B_i32, %C_i32)
      : (memref<1x9x2xi32>, memref<1x2x5xi32>, memref<1x9x5xi32>) -> ()

    %print_i32 = memref.cast %C_i32 : memref<1x9x5xi32> to memref<*xi32>
    call @printMemrefI32(%print_i32) : (memref<*xi32>) -> ()

    memref.dealloc %C_i32 : memref<1x9x5xi32>
    memref.dealloc %B_i32 : memref<1x2x5xi32>
    memref.dealloc %A_i32 : memref<1x9x2xi32>
    memref.dealloc %C_f32 : memref<1x3x5xf32>
    memref.dealloc %B_f32 : memref<1x2x5xf32>
    memref.dealloc %A_f32 : memref<1x3x2xf32>
    return
  }
}

// CHECK-IR-LABEL: func.func @batch_m_lt_8_f32
// CHECK-IR:       scf.parallel
// CHECK-IR:       vector.fma

// CHECK-IR-LABEL: func.func @batch_i32
// CHECK-IR:       scf.parallel
// CHECK-IR:       arith.muli {{.*}} : vector<4xi32>
// CHECK-IR:       arith.addi {{.*}} : vector<4xi32>

// CHECK-IR-LABEL: func.func @batch_transpose_b_kept
// CHECK-IR:       linalg.batch_matmul
// CHECK-IR-SAME:  indexing_maps

// CHECK-OUT: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 3, 5] strides = [15, 5, 1] data =
// CHECK-OUT: [14,   14,   14,   14,   14]
// CHECK-OUT: [14,   14,   14,   14,   14]
// CHECK-OUT: [14,   14,   14,   14,   14]
// CHECK-OUT: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 9, 5] strides = [45, 5, 1] data =
// CHECK-OUT: [2,   2,   2,   2,   2]
// CHECK-OUT: [4,   4,   4,   4,   4]
// CHECK-OUT: [6,   6,   6,   6,   6]
// CHECK-OUT: [8,   8,   8,   8,   8]
// CHECK-OUT: [10,   10,   10,   10,   10]
// CHECK-OUT: [12,   12,   12,   12,   12]
// CHECK-OUT: [14,   14,   14,   14,   14]
// CHECK-OUT: [16,   16,   16,   16,   16]
// CHECK-OUT: [18,   18,   18,   18,   18]
