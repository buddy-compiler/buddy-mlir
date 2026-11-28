// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s --implicit-check-not='!xsai_ame.'

// -linalg-to-xsai-ame keeps the C accumulator of a K loop in a tile register:
// the loop carries it in iter_args instead of storing and reloading it.
// CHECK-LABEL: func.func @loop_accumulator
// CHECK: scf.if
// CHECK: %[[INIT:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (!llvm.target<"riscv.xsai.zames.tile.int32">)
// CHECK: %[[A:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: %[[B:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zames.mlbt"
// CHECK: %[[M:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"(%[[A]], %[[B]], %[[ACC]])
// CHECK: scf.yield %[[M]] : !llvm.target<"riscv.xsai.zames.tile.int32">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"(%[[R]],
func.func @loop_accumulator(%A: memref<128x512xi8>, %B: memref<512x128xi8>, %C: memref<128x128xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %nonempty = arith.cmpi slt, %c0, %c512 : index
  scf.if %nonempty {
    %init = xsai_ame.load %C : memref<128x128xi32> -> !xsai_ame.tile<c, 128x128, i32>
    %r = scf.for %k = %c0 to %c512 step %c64 iter_args(%acc = %init) -> (!xsai_ame.tile<c, 128x128, i32>) {
      %av = memref.subview %A[0, %k] [128, 64] [1, 1] : memref<128x512xi8> to memref<128x64xi8, strided<[512, 1], offset: ?>>
      %bv = memref.subview %B[%k, 0] [64, 128] [1, 1] : memref<512x128xi8> to memref<64x128xi8, strided<[128, 1], offset: ?>>
      %a = xsai_ame.load %av : memref<128x64xi8, strided<[512, 1], offset: ?>> -> !xsai_ame.tile<a, 128x64, i8>
      %b = xsai_ame.load %bv <{transpose = true}> : memref<64x128xi8, strided<[128, 1], offset: ?>> -> !xsai_ame.tile<b, 128x64, i8>
      %m = xsai_ame.mma %a, %b, %acc : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, i32>
      scf.yield %m : !xsai_ame.tile<c, 128x128, i32>
    }
    xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  }
  return
}
