// RUN: buddy-opt %s --lower-linalg-to-boscame='target=qwen3-fpga' | FileCheck %s
// RUN: buddy-opt %s --lower-linalg-to-boscame='target=qwen3-fpga' \
// RUN:   -expand-strided-metadata -canonicalize | FileCheck %s --check-prefix=STRIDE
// RUN: not buddy-opt %s --lower-linalg-to-boscame > %t.default 2>&1
// RUN: FileCheck %s --check-prefix=DEFAULT < %t.default
//
// This file tests the strict Qwen3 i8 x i8 -> f32 direct-C lowering on the
// FPGA AME target (bosc_ame.target = "qwen3-fpga").  Only a fresh zero-filled
// temporary with one matmul and one copy is eligible; a temporary with multiple
// copies, or a non-zero initial value, must stay for the later VIR pipeline.
//
// The same input is run without the FPGA target to prove the default
// (upstream/GEM5) pathway neither claims these operations nor emits the
// bit-field mtype encoding.
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
// The temporary buffer, its zero fill and the copy are all elided: the AME
// writes straight into the final destination through the offset-aware pointer.
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
//
// The bit-field mtype value must be bound to the phase that consumes it:
// 65602 (0x10042, i32 accumulator) seeds mlce32.m and closes with msce32.m,
// 65552 (0x10010, i8 MMA) covers the K reduction.
// CHECK: %[[ACC_SEED_TYPE:.*]] = arith.constant 65602 : i64
// CHECK: bosc_ame.msettype %[[ACC_SEED_TYPE]]
// CHECK: %[[ACC_SEED:.*]] = bosc_ame.mlce32.m {{.*}} -> vector<4x4xi32>
// CHECK: %[[MMA_TYPE:.*]] = arith.constant 65552 : i64
// CHECK: bosc_ame.msettype %[[MMA_TYPE]]
// CHECK: %[[ACC_FINAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ACC_SEED]]) -> (vector<4x4xi32>) {
// CHECK:   bosc_ame.msettilek
// CHECK:   bosc_ame.mlae8.m {{.*}} -> vector<4x4xi8>
// CHECK:   bosc_ame.mlbte8.m {{.*}} -> vector<4x4xi8>
// CHECK:   %[[NEXT:.*]] = bosc_ame.mqma.b.mm %[[ACC]], {{.*}} : vector<4x4xi32>, vector<4x4xi8>, vector<4x4xi8> -> vector<4x4xi32>
// CHECK:   scf.yield %[[NEXT]] : vector<4x4xi32>
// CHECK: }
// CHECK: bosc_ame.msettype %{{.*}} : i64
// CHECK: bosc_ame.msce32.m %[[ACC_FINAL]], {{.*}} : vector<4x4xi32>, memref<{{.*}}xf32

// The two rejected shapes keep their temporary and are left for the CPU/VIR
// pipeline instead of being lowered with a different hardware contract.
// CHECK-LABEL: func.func @matmul_multiple_copies
// CHECK: memref.alloc
// CHECK: linalg.matmul
// CHECK: memref.copy
// CHECK: memref.copy
// CHECK-NOT: bosc_ame

// CHECK-LABEL: func.func @matmul_nonzero_fill
// CHECK: linalg.matmul
// CHECK-NOT: bosc_ame

// The default pathway must reject these shapes instead of silently emitting a
// different hardware contract: upstream has no i8 x i8 -> f32 AME matmul.
// DEFAULT: error: failed to legalize operation 'linalg.matmul'
// DEFAULT-NOT: bosc_ame

// After -expand-strided-metadata -canonicalize the tile extents and byte
// strides must become constants that match the real problem: the 4x32 tile is
// smaller than the 16x64 hardware tile, so mtilem/mtilek collapse to 4/32, the
// f32 destination keeps its 16-element (64 B) leading stride, A has a 32 B row
// stride and the transposed B view a 4 B contiguity.
// STRIDE-LABEL: func.func @matmul_unique_copy
// STRIDE-DAG: %[[FOUR:.*]] = arith.constant 4 : i64
// (K = 32 and A's 32 B row stride happen to be the same value here, so a single
// constant feeds both msettilek and mlae8.m.)
// STRIDE-DAG: %[[THIRTY_TWO:.*]] = arith.constant 32 : i64
// STRIDE-DAG: %[[C_STRIDE:.*]] = arith.constant 64 : i64
// STRIDE: bosc_ame.msettilem %[[FOUR]] : i64
// STRIDE: bosc_ame.msettilen %[[FOUR]] : i64
// STRIDE: bosc_ame.mlce32.m {{.*}}, %[[C_STRIDE]] : memref<?x?xf32, strided<[16, 1], offset: ?>>
// STRIDE: bosc_ame.msettilek %[[THIRTY_TWO]] : i64
// STRIDE: bosc_ame.mlae8.m {{.*}}, %[[THIRTY_TWO]] : memref<?x?xi8, strided<[32, 1], offset: ?>>
// STRIDE: bosc_ame.mlbte8.m {{.*}}, %[[FOUR]] : memref<?x?xi8, strided<[4, 1], offset: ?>>
// STRIDE: bosc_ame.msce32.m {{.*}}, {{.*}}, %[[C_STRIDE]] :
