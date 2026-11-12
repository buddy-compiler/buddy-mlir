// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -O2 -o - \
// RUN: | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='rvv-vlen=128' \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | buddy-translate --buddy-to-llvmir \
// RUN: | llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -O2 -filetype=obj -o /dev/null

// An integer matmul, vectorized along N: A and B of one type extended to
// the type of C (the cast of the matmul), multiplied and added (exact; LLVM
// widens i8 to i16, then multiply-adds to i32 with vwmacc).

// CHECK-LABEL: func.func @i8_i32
// CHECK: %[[A:.*]] = memref.load %{{.*}} : memref<4x8xi8>
// CHECK: %[[AW:.*]] = arith.extsi %[[A]] : i8 to i32
// CHECK: %[[AV:.*]] = vir.broadcast %[[AW]] : i32 -> !vir.vec<?xi32>
// CHECK: %[[B:.*]] = vir.load %{{.*}} : memref<8x16xi8> -> !vir.vec<?xi8>
// CHECK: %[[BW:.*]] = vir.extsi %[[B]] : !vir.vec<?xi8> -> !vir.vec<?xi32>
// CHECK: %[[P:.*]] = arith.muli %[[AV]], %[[BW]] : !vir.vec<?xi32>
// CHECK: arith.addi %{{.*}}, %[[P]] : !vir.vec<?xi32>
// CHECK-NOT: linalg.matmul
// ASM-LABEL: i8_i32:
// ASM: vsext.vf2
// ASM: vwmacc.vx
func.func @i8_i32(%a: memref<4x8xi8>, %b: memref<8x16xi8>, %c: memref<4x16xi32>) {
  linalg.matmul ins(%a, %b : memref<4x8xi8>, memref<8x16xi8>)
                outs(%c : memref<4x16xi32>)
  return
}

// CHECK-LABEL: func.func @u8_i32
// CHECK: arith.extui %{{.*}} : i8 to i32
// CHECK: vir.extui %{{.*}} : !vir.vec<?xi8> -> !vir.vec<?xi32>
// CHECK-NOT: linalg.matmul
// ASM-LABEL: u8_i32:
// ASM: vzext.vf2
// ASM: vwmaccu.vx
func.func @u8_i32(%a: memref<4x8xi8>, %b: memref<8x16xi8>, %c: memref<4x16xi32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
                ins(%a, %b : memref<4x8xi8>, memref<8x16xi8>)
                outs(%c : memref<4x16xi32>)
  return
}

// CHECK-LABEL: func.func @i32_i32
// CHECK-NOT: ext
// CHECK: arith.muli %{{.*}}, %{{.*}} : !vir.vec<?xi32>
// CHECK-NOT: linalg.matmul
func.func @i32_i32(%a: memref<?x?xi32>, %b: memref<?x?xi32>, %c: memref<?x?xi32>) {
  linalg.matmul ins(%a, %b : memref<?x?xi32>, memref<?x?xi32>)
                outs(%c : memref<?x?xi32>)
  return
}

// Floats of different types are left to the other passes.
// CHECK-LABEL: func.func @f16_f32
// CHECK: linalg.matmul
func.func @f16_f32(%a: memref<4x8xf16>, %b: memref<8x16xf16>, %c: memref<4x16xf32>) {
  linalg.matmul ins(%a, %b : memref<4x8xf16>, memref<8x16xf16>)
                outs(%c : memref<4x16xf32>)
  return
}
