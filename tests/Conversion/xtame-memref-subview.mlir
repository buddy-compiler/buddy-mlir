// RUN: buddy-opt %s -lower-xt-ame | FileCheck %s

func.func @load_subview(%source: memref<4x8xi8>, %k: index) {
  %tile = memref.subview %source[0, %k] [4, 4] [1, 1]
      : memref<4x8xi8> to memref<4x4xi8, strided<[8, 1], offset: ?>>
  %stride = arith.constant 8 : i64
  xt_ame.th.mlde8 1, %stride, %tile
      : memref<4x4xi8, strided<[8, 1], offset: ?>>
  return
}

// CHECK-LABEL: func.func @load_subview
// A subview load must address the tile through the descriptor offset, not only
// through the aligned base pointer.
// CHECK-DAG: %{{.*}}, %[[OFFSET:.*]], %{{.*}}:2, %{{.*}}:2 = memref.extract_strided_metadata
// CHECK-DAG: %[[PTRINT:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[I64PTR:.*]] = arith.index_cast %[[PTRINT]] : index to i64
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[I64PTR]]
// CHECK: %[[I64OFFSET:.*]] = arith.index_cast %[[OFFSET]] : index to i64
// CHECK: llvm.getelementptr %[[PTR]][%[[I64OFFSET]]]
// CHECK: llvm.inline_asm has_side_effects "th.mlde8 m1, $0, $1"
