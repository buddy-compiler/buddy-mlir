// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

func.func @matmul_f16_inputs_f32_output(%A: memref<2x4xf16>,
                                        %B: memref<4x?xf16>,
                                        %C: memref<2x?xf32>) {
  linalg.matmul ins(%A, %B : memref<2x4xf16>, memref<4x?xf16>)
      outs(%C : memref<2x?xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_f16_inputs_f32_output
// CHECK-NOT: linalg.matmul
// CHECK: arith.extf {{.*}} : vector<4xf16> to vector<4xf32>
// CHECK: arith.extf {{.*}} : vector<4xf16> to vector<4xf32>
// CHECK: vector.fma {{.*}} : vector<4xf32>
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: arith.mulf {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK-NOT: linalg.matmul

// -----

func.func @matmul_i8_inputs_i32_output(%A: memref<2x4xi8>,
                                       %B: memref<4x?xi8>,
                                       %C: memref<2x?xi32>) {
  linalg.matmul ins(%A, %B : memref<2x4xi8>, memref<4x?xi8>)
      outs(%C : memref<2x?xi32>)
  return
}

// CHECK-LABEL: func.func @matmul_i8_inputs_i32_output
// CHECK-NOT: linalg.matmul
// CHECK: arith.extsi {{.*}} : vector<4xi8> to vector<4xi32>
// CHECK: arith.extsi {{.*}} : vector<4xi8> to vector<4xi32>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: arith.muli {{.*}} : i32
// CHECK: arith.addi {{.*}} : i32
// CHECK-NOT: linalg.matmul

// -----

func.func @matmul_u8_inputs_i32_output(%A: memref<2x4xi8>,
                                       %B: memref<4x?xi8>,
                                       %C: memref<2x?xi32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
      ins(%A, %B : memref<2x4xi8>, memref<4x?xi8>)
      outs(%C : memref<2x?xi32>)
  return
}

// CHECK-LABEL: func.func @matmul_u8_inputs_i32_output
// CHECK-NOT: linalg.matmul
// CHECK: arith.extui {{.*}} : vector<4xi8> to vector<4xi32>
// CHECK: arith.extui {{.*}} : vector<4xi8> to vector<4xi32>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: arith.extui {{.*}} : i8 to i32
// CHECK: arith.extui {{.*}} : i8 to i32
// CHECK: arith.muli {{.*}} : i32
// CHECK: arith.addi {{.*}} : i32
// CHECK-NOT: linalg.matmul
