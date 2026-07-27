// RUN: buddy-opt %s -lower-linalg-to-boscame | FileCheck %s
// RUN: buddy-opt %s \
// RUN:   -lower-linalg-to-boscame \
// RUN:   -lower-bosc-ame \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -expand-strided-metadata \
// RUN:   -lower-affine \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64 \
// RUN:   -mattr=+xboscame -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s \
// RUN:   -lower-linalg-to-boscame \
// RUN:   -expand-strided-metadata \
// RUN:   -canonicalize | FileCheck %s --check-prefix=STRIDE
//
// This file tests the strict Qwen3 i8 x i8 -> f32 direct-C lowering. Only a
// fresh zero-filled temporary with one matmul and one copy is eligible; a
// temporary with multiple copies must remain for the later VIR pipeline.
//
module {

  func.func @matmul_unique_copy(%A: memref<4x32xi8>,
                                %B: memref<32x4xi8>,
                                %out: memref<*xf32>,
                                %rowOffset: index,
                                %columnOffset: index) {
    %C = memref.alloc() : memref<4x4xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A, %B : memref<4x32xi8>, memref<32x4xi8>)
        outs(%C : memref<4x4xf32>)
    %targetOffset = arith.addi %rowOffset, %columnOffset : index
    %target = memref.reinterpret_cast %out to
        offset: [%targetOffset], sizes: [4, 4], strides: [16, 1]
        : memref<*xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
    memref.copy %C, %target
        : memref<4x4xf32>
          to memref<4x4xf32, strided<[16, 1], offset: ?>>
    memref.dealloc %C : memref<4x4xf32>
    return
  }

  func.func @matmul_multiple_copies(%A: memref<4x32xi8>,
                                    %B: memref<32x4xi8>,
                                    %out0: memref<4x4xf32>,
                                    %out1: memref<4x4xf32>) {
    %C = memref.alloc() : memref<4x4xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A, %B : memref<4x32xi8>, memref<32x4xi8>)
        outs(%C : memref<4x4xf32>)
    memref.copy %C, %out0 : memref<4x4xf32> to memref<4x4xf32>
    memref.copy %C, %out1 : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %C : memref<4x4xf32>
    return
  }

  func.func @matmul_nonzero_fill(%A: memref<4x32xi8>,
                                 %B: memref<32x4xi8>,
                                 %out: memref<4x4xf32>) {
    %C = memref.alloc() : memref<4x4xf32>
    %one = arith.constant 1.0 : f32
    linalg.fill ins(%one : f32) outs(%C : memref<4x4xf32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>}
        ins(%A, %B : memref<4x32xi8>, memref<32x4xi8>)
        outs(%C : memref<4x4xf32>)
    memref.copy %C, %out : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %C : memref<4x4xf32>
    return
  }
}

// CHECK-LABEL: func.func @matmul_unique_copy
// CHECK-SAME: %[[A:.*]]: memref<4x32xi8>
// CHECK-SAME: %[[B:.*]]: memref<32x4xi8>
// CHECK-SAME: %[[OUT:.*]]: memref<*xf32>
// CHECK-NOT: memref.alloc
// CHECK: %[[TARGET_OFFSET:.*]] = arith.addi
// CHECK: %[[TARGET:.*]] = memref.reinterpret_cast %[[OUT]] to offset: [%[[TARGET_OFFSET]]], sizes: [4, 4], strides: [16, 1]
// CHECK: linalg.fill {{.*}} outs(%[[TARGET]] : memref<4x4xf32, strided<[16, 1], offset: ?>>)
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     %[[SUBVIEW_C:.*]] = memref.subview %[[TARGET]]
// CHECK:     bosc_ame.mlce32.m 0, %[[SUBVIEW_C]]
// CHECK:     scf.for
// CHECK:       %[[SUBVIEW_A:.*]] = memref.subview %[[A]]
// CHECK:       %[[SUBVIEW_B:.*]] = memref.subview %[[B]]
// CHECK:       bosc_ame.mlae8.m 0, %[[SUBVIEW_A]]
// CHECK:       bosc_ame.mlbte8.m 1, %[[SUBVIEW_B]]
// CHECK:       bosc_ame.mqma.b.mm 0, 0, 1
// CHECK:     bosc_ame.msce32.m 0, %[[SUBVIEW_C]]
// CHECK-NOT: memref.copy
// CHECK-NOT: memref.dealloc
// CHECK: return

// CHECK-LABEL: func.func @matmul_multiple_copies
// CHECK-NOT: bosc_ame
// CHECK: %[[C:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: linalg.fill {{.*}} outs(%[[C]] : memref<4x4xf32>)
// CHECK: linalg.matmul
// CHECK-SAME: outs(%[[C]] : memref<4x4xf32>)
// CHECK: memref.copy %[[C]], {{.*}}
// CHECK: memref.copy %[[C]], {{.*}}
// CHECK: memref.dealloc %[[C]]

// CHECK-LABEL: func.func @matmul_nonzero_fill
// CHECK-NOT: bosc_ame
// CHECK: linalg.fill
// CHECK-NOT: bosc_ame
// CHECK: linalg.matmul
// CHECK-NOT: bosc_ame
// CHECK: memref.copy
// CHECK-NOT: bosc_ame
// CHECK: return

// STRIDE-LABEL: func.func @matmul_unique_copy
// STRIDE: %[[C_STRIDE:.*]] = arith.constant 64 : i64
// STRIDE: %[[K_BOUND:.*]] = arith.constant 32 : index
// STRIDE: %[[K_STEP:.*]] = arith.constant 16 : index
// STRIDE: bosc_ame.mlce32.m 0, {{.*}}, %[[C_STRIDE]]
// STRIDE: scf.for {{.*}} = {{.*}} to %[[K_BOUND]] step %[[K_STEP]]
// STRIDE: bosc_ame.mqma.b.mm 0, 0, 1
// STRIDE: bosc_ame.msce32.m 0, {{.*}}, %[[C_STRIDE]]

// ASM-DAG: mlce32.m
// ASM-DAG: mlae8.m
// ASM-DAG: mlbte8.m
// ASM-DAG: mqma.b.mm
// ASM-DAG: msce32.m
