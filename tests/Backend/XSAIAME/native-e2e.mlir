// RUN: buddy-opt %s -lower-xsai-ame -expand-strided-metadata -lower-affine -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=llvm.riscv.buddy --implicit-check-not=@llvm.riscv.xsai.zames.msettilem( --implicit-check-not=@llvm.riscv.xsai.zames.msettilen( --implicit-check-not=@llvm.riscv.xsai.zames.msettilek( < %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llc -mtriple=riscv64 -mattr=+zmasync -verify-machineinstrs -O0 %t.bc -o %t.s
// RUN: FileCheck %s --check-prefix=ASM < %t.s
// RUN: llc -mtriple=riscv64 -mattr=+zmasync -verify-machineinstrs -O0 -filetype=obj %t.bc -o %t.o
// RUN: llvm-objdump --mattr=+zmasync -d %t.o | FileCheck %s --check-prefix=OBJ
// RUN: llvm-readelf -A %t.o | FileCheck %s --check-prefix=ATTR

// The Buddy boundary starts with MLIR; descriptors, byte strides and
// target-extension overloads must export and pass the native LLVM verifier.
// IR-LABEL: define void @native_i8
// IR: call void @llvm.riscv.xsai.zames.msettilemi(i64 128)
// IR: call void @llvm.riscv.xsai.zames.msettileki(i64 64)
// IR: call target("riscv.xsai.zames.tile.int8") @llvm.riscv.xsai.zames.mla.
// IR: call void @llvm.riscv.xsai.zames.msettileni(i64 128)
// IR: call target("riscv.xsai.zames.tile.int8") @llvm.riscv.xsai.zames.mlb.
// IR: call target("riscv.xsai.zames.tile.int32") @llvm.riscv.xsai.zames.mzero.
// IR: call target("riscv.xsai.zames.tile.int32") @llvm.riscv.xsai.zames.mmacc.
// IR: call void @llvm.riscv.xsai.zames.msc.
// ASM-LABEL: native_i8:
// ASM: msettilemi 128
// ASM: msettileki 64
// ASM: mla
// ASM: msettileni 128
// ASM: mlb
// ASM: mzero
// ASM: mmacc
// ASM: msc
// OBJ-LABEL: <native_i8>:
// OBJ: msettilemi
// OBJ: msettileki
// OBJ: mla
// OBJ: msettileni
// OBJ: mlb
// OBJ: mzero
// OBJ: mmacc
// OBJ: msc
func.func @native_i8(%A: memref<128x64xi8>, %B: memref<128x64xi8>, %C: memref<128x128xi32>) {
  %a = xsai_ame.load %A : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.load %B : memref<128x64xi8> -> !xsai_ame.tile<b, 128x64, i8>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, i32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  return
}

// A complete tile over a fully dynamic descriptor reads its byte address and
// row stride from the runtime metadata; the tile shape stays immediate.
// IR-LABEL: define void @native_dynamic_view
// IR: call void @llvm.riscv.xsai.zames.msettilemi(i64 128)
// IR: call void @llvm.riscv.xsai.zames.msettileki(i64 32)
// IR: call target("riscv.xsai.zames.tile.fp16") @llvm.riscv.xsai.zames.mla.
// IR: call void @llvm.riscv.xsai.zames.msa.
// ASM-LABEL: native_dynamic_view:
// ASM: msettilemi 128
// ASM: msettileki 32
// ASM: mla
// ASM: msa
// OBJ-LABEL: <native_dynamic_view>:
// OBJ: msettilemi
// OBJ: msettileki
// OBJ: mla
// OBJ: msa
func.func @native_dynamic_view(%view: memref<?x?xf16, strided<[?, 1], offset: ?>>) {
  %a = xsai_ame.load %view : memref<?x?xf16, strided<[?, 1], offset: ?>> -> !xsai_ame.tile<a, 128x32, fp16>
  xsai_ame.store %a, %view : !xsai_ame.tile<a, 128x32, fp16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
  return
}

// Register selection is checked without pinning incidental physical numbering.
// IR-LABEL: define void @native_sync
// IR: call target("riscv.xsai.zmasync.sync") @llvm.riscv.xsai.zmasync.msyncregreset.
// IR: call void @llvm.riscv.xsai.zmasync.mfence()
// IR: call target("riscv.xsai.zmasync.sync") @llvm.riscv.xsai.zmasync.mrelease.
// IR: call void @llvm.riscv.xsai.zmasync.macquire.
// ASM-LABEL: native_sync:
// ASM: msyncregreset sync{{[0-9]+}}
// ASM: mfence
// ASM: mrelease sync{{[0-9]+}}
// ASM: macquire sync{{[0-9]+}},
// OBJ-LABEL: <native_sync>:
// OBJ: msyncregreset sync{{[0-9]+}}
// OBJ: mfence
// OBJ: mrelease sync{{[0-9]+}}
// OBJ: macquire sync{{[0-9]+}},
func.func @native_sync(%target: index) {
  %s0 = xsai_ame.msyncregreset : !xsai_ame.sync
  xsai_ame.mfence
  %s1 = xsai_ame.mrelease %s0 : !xsai_ame.sync
  xsai_ame.macquire %s1, %target : !xsai_ame.sync, index
  return
}

// ATTR: TagName: arch
// ATTR: Value: rv64{{.*}}zmasync1p0{{.*}}zames1p0
