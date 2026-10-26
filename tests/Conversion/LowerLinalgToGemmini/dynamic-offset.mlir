// RUN: buddy-opt %s -lower-gemmini | FileCheck %s

func.func @packed_parameters(
    %a: memref<2x3xi8, strided<[3, 1], offset: ?>>,
    %b: memref<3x4xi8, strided<[4, 1], offset: ?>>,
    %c: memref<2x4xi32, strided<[4, 1], offset: ?>>,
    %d: memref<2x4xi32, strided<[4, 1], offset: ?>>) {
  gemmini.tile_matmul %a %b %c %d <{fullC = true}> :
      memref<2x3xi8, strided<[3, 1], offset: ?>>
      memref<3x4xi8, strided<[4, 1], offset: ?>>
      memref<2x4xi32, strided<[4, 1], offset: ?>>
      memref<2x4xi32, strided<[4, 1], offset: ?>>
  return
}

// CHECK-LABEL: llvm.func @packed_parameters
// CHECK-NOT: -9223372036854775808
// CHECK: %[[AOFFSET:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.add {{.*}}, %[[AOFFSET]]
// CHECK: %[[BOFFSET:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.add {{.*}}, %[[BOFFSET]]
// CHECK: %[[COFFSET:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CSIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK: llvm.mul %[[COFFSET]], %[[CSIZE]]
// CHECK: %[[DOFFSET:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DSIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK: llvm.mul %[[DOFFSET]], %[[DSIZE]]
// CHECK: gemmini.intr.loop_ws
