// RUN: buddy-opt %s -lower-xsai-ame -expand-strided-metadata -lower-affine -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=@llvm.riscv.xsai.zames.msettilem( --implicit-check-not=@llvm.riscv.xsai.zames.msettilen( --implicit-check-not=@llvm.riscv.xsai.zames.msettilek( < %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llc -mtriple=riscv64 -mattr=+zames -verify-machineinstrs -O0 -filetype=obj %t.bc -o %t.o
// RUN: llvm-objdump --mattr=+zames -d %t.o | FileCheck %s --check-prefix=OBJ
// RUN: llvm-readelf -A %t.o | FileCheck %s --check-prefix=ATTR

// A/B/C normal and transposed memory forms all survive native object emission.
// IR-LABEL: define void @all_memory_roles
// IR: @llvm.riscv.xsai.zames.mla.
// IR: @llvm.riscv.xsai.zames.msa.
// IR: @llvm.riscv.xsai.zames.mlat.
// IR: @llvm.riscv.xsai.zames.msat.
// IR: @llvm.riscv.xsai.zames.mlb.
// IR: @llvm.riscv.xsai.zames.msb.
// IR: @llvm.riscv.xsai.zames.mlbt.
// IR: @llvm.riscv.xsai.zames.msbt.
// IR: @llvm.riscv.xsai.zames.mlc.
// IR: @llvm.riscv.xsai.zames.msc.
// IR: @llvm.riscv.xsai.zames.mlct.
// IR: @llvm.riscv.xsai.zames.msct.
// OBJ-LABEL: <all_memory_roles>:
// OBJ: mla
// OBJ: msa
// OBJ: mlat
// OBJ: msat
// OBJ: mlb
// OBJ: msb
// OBJ: mlbt
// OBJ: msbt
// OBJ: mlc
// OBJ: msc
// OBJ: mlct
// OBJ: msct
func.func @all_memory_roles(%A: memref<128x64xi8>, %AT: memref<64x128xi8>, %B: memref<128x64xi8>, %BT: memref<64x128xi8>, %C: memref<128x128xf32>, %CT: memref<128x128xf32>) {
  %a = xsai_ame.load %A : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a, %A : !xsai_ame.tile<a, 128x64, i8>, memref<128x64xi8>
  %at = xsai_ame.load %AT <{transpose = true}> : memref<64x128xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %at, %AT <{transpose = true}> : !xsai_ame.tile<a, 128x64, i8>, memref<64x128xi8>
  %b = xsai_ame.load %B : memref<128x64xi8> -> !xsai_ame.tile<b, 128x64, i8>
  xsai_ame.store %b, %B : !xsai_ame.tile<b, 128x64, i8>, memref<128x64xi8>
  %bt = xsai_ame.load %BT <{transpose = true}> : memref<64x128xi8> -> !xsai_ame.tile<b, 128x64, i8>
  xsai_ame.store %bt, %BT <{transpose = true}> : !xsai_ame.tile<b, 128x64, i8>, memref<64x128xi8>
  %c = xsai_ame.load %C : memref<128x128xf32> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %c, %C : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  %ct = xsai_ame.load %CT <{transpose = true}> : memref<128x128xf32> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %ct, %CT <{transpose = true}> : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}

// Complete tiles configure every axis with immediate setters; K follows the
// element width (64 for 8-bit, 32 for 16-bit), never a register setter.
// IR-LABEL: define void @immediate_setters
// IR: call void @llvm.riscv.xsai.zames.msettilemi(i64 128)
// IR: call void @llvm.riscv.xsai.zames.msettileki(i64 64)
// IR: call void @llvm.riscv.xsai.zames.msa.
// IR: call void @llvm.riscv.xsai.zames.msettileni(i64 128)
// IR: call void @llvm.riscv.xsai.zames.msettileki(i64 32)
// IR: call void @llvm.riscv.xsai.zames.msbt.
// IR-LABEL: define void @subview_bytes
// OBJ-LABEL: <immediate_setters>:
// OBJ: msettilemi
// OBJ: msettileki
// OBJ: msa
// OBJ: msettileni
// OBJ: msettileki
// OBJ: msbt
func.func @immediate_setters(%a_out: memref<128x64xi8>, %b_out: memref<32x128xf16>) {
  %a = xsai_ame.mzero : !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a, %a_out : !xsai_ame.tile<a, 128x64, i8>, memref<128x64xi8>
  %b = xsai_ame.mzero : !xsai_ame.tile<b, 128x32, fp16>
  xsai_ame.store %b, %b_out <{transpose = true}> : !xsai_ame.tile<b, 128x32, fp16>, memref<32x128xf16>
  return
}

// Both offsets and row strides are measured in elements in memref metadata,
// but native tile memory operations require byte addressing.
// IR: %[[LOAD_ADDRESS:.*]] = add i64 {{.*}}, 1700
// IR: %[[LOAD_PTR:.*]] = inttoptr i64 %[[LOAD_ADDRESS]] to ptr
// IR: %[[C_TILE:.*]] = call target("riscv.xsai.zames.tile.fp32") @llvm.riscv.xsai.zames.mlc.triscv.xsai.zames.tile.fp32t(ptr %[[LOAD_PTR]], i64 560)
// IR: %[[STORE_ADDRESS:.*]] = add i64 {{.*}}, 1700
// IR: %[[STORE_PTR:.*]] = inttoptr i64 %[[STORE_ADDRESS]] to ptr
// IR: call void @llvm.riscv.xsai.zames.msc.triscv.xsai.zames.tile.fp32t(target("riscv.xsai.zames.tile.fp32") %[[C_TILE]], ptr %[[STORE_PTR]], i64 560)
// OBJ-LABEL: <subview_bytes>:
// OBJ: mlc
// OBJ: msc
func.func @subview_bytes(%base: memref<140x140xf32>) {
  %view = memref.subview %base[3, 5] [128, 128] [1, 1] : memref<140x140xf32> to memref<128x128xf32, strided<[140, 1], offset: 425>>
  %c = xsai_ame.load %view : memref<128x128xf32, strided<[140, 1], offset: 425>> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %c, %view : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32, strided<[140, 1], offset: 425>>
  return
}

// ATTR: TagName: arch
// ATTR: Value: rv64{{.*}}zames1p0
