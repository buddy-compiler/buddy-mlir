// RUN: buddy-opt %s -matmul-vectorization="vector-size=8" | FileCheck %s

module {
  func.func @matmul_f16_f32(%a: memref<?x?xf16>, %b: memref<?x?xf16>,
                            %c: memref<?x?xf32>) {
    linalg.matmul
      ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>)
      outs(%c : memref<?x?xf32>)
    return
  }

  func.func @matmul_i8_i32(%a: memref<?x?xi8>, %b: memref<?x?xi8>,
                           %c: memref<?x?xi32>) {
    linalg.matmul
      ins(%a, %b : memref<?x?xi8>, memref<?x?xi8>)
      outs(%c : memref<?x?xi32>)
    return
  }
}

// CHECK-LABEL: func.func @matmul_f16_f32
// CHECK: vector.load {{.*}} : memref<?x?xf32>, vector<8xf32>
// CHECK: vector.load {{.*}} : memref<?x?xf16>, vector<8xf16>
// CHECK: arith.extf {{.*}} : vector<8xf16> to vector<8xf32>
// CHECK: arith.extf {{.*}} : f16 to f32
// CHECK: vector.fma {{.*}} : vector<8xf32>
// CHECK: vector.store {{.*}} : memref<?x?xf32>, vector<8xf32>

// CHECK-LABEL: func.func @matmul_i8_i32
// CHECK: vector.load {{.*}} : memref<?x?xi32>, vector<8xi32>
// CHECK: vector.load {{.*}} : memref<?x?xi8>, vector<8xi8>
// CHECK: arith.extsi {{.*}} : vector<8xi8> to vector<8xi32>
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: arith.muli {{.*}} : vector<8xi32>
// CHECK: arith.addi {{.*}} : vector<8xi32>
// CHECK: vector.store {{.*}} : memref<?x?xi32>, vector<8xi32>
