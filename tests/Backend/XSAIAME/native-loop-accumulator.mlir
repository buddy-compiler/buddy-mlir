// RUN: buddy-opt %s -lower-xsai-ame -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: FileCheck %s --check-prefix=IR < %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llc -mtriple=riscv64 -mattr=+zames -verify-machineinstrs -O2 -filetype=obj %t.bc -o %t.o
// RUN: llvm-objdump --mattr=+zames -d %t.o | FileCheck %s --check-prefix=OBJ
//
// The C accumulator that -linalg-to-xsai-ame carries in scf.for iter_args
// becomes a tile phi and stays in one accumulator register across the loop.
// The fast register allocator (-O0) cannot spill tile registers across blocks;
// the model pipeline runs llc at -O3.
// IR-LABEL: define void @loop_accumulator
// IR: call target("riscv.xsai.zames.tile.int32") @llvm.riscv.xsai.zames.mlc.
// IR: phi target("riscv.xsai.zames.tile.int32")
// IR: call target("riscv.xsai.zames.tile.int32") @llvm.riscv.xsai.zames.mmacc.
// IR: call void @llvm.riscv.xsai.zames.msc.
// OBJ-LABEL: <loop_accumulator>:
// OBJ: mlc acc[[ACC:[0-9]+]],
// OBJ: mla
// OBJ: mlbt
// OBJ: mmacc acc[[ACC]],
// OBJ: msc acc[[ACC]],
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
