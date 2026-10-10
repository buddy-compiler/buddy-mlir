// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

func.func @matmul_f16_to_f32(%a: memref<4x8xf16>, %b: memref<8x16xf16>,
                             %c: memref<4x16xf32>) {
  linalg.matmul ins(%a, %b : memref<4x8xf16>, memref<8x16xf16>)
      outs(%c : memref<4x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_f16_to_f32
// CHECK-NOT: linalg.matmul
// CHECK: vir.load
// CHECK: vir.extf
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @matmul_bf16_to_f32(%a: memref<4x8xbf16>, %b: memref<8x16xbf16>,
                              %c: memref<4x16xf32>) {
  linalg.matmul ins(%a, %b : memref<4x8xbf16>, memref<8x16xbf16>)
      outs(%c : memref<4x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_bf16_to_f32
// CHECK-NOT: linalg.matmul
// CHECK: vir.load
// CHECK: vir.extf
// CHECK: vir.fma
// CHECK: vir.store
