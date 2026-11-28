// RUN: buddy-opt %s -lower-xsai-ame -expand-strided-metadata -lower-affine -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llc -mtriple=riscv64 -mattr=+zames,+f -verify-machineinstrs -O0 -filetype=obj %t.bc -o %t.o
// RUN: llvm-objdump --mattr=+zames,+f -d %t.o | FileCheck %s
// RUN: llvm-readelf -A %t.o | FileCheck %s --check-prefix=ATTR

// Each function starts at the Buddy typed-memory boundary and consumes its
// MMA result in a store. No tile ABI, spill, synthetic intrinsic or overlay.
// CHECK-LABEL: <i8_i32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
func.func @i8_i32(%A: memref<128x64xi8>, %B: memref<128x64xi8>, %C: memref<128x128xi32>) {
  %a = xsai_ame.load %A : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.load %B : memref<128x64xi8> -> !xsai_ame.tile<b, 128x64, i8>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, i32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  return
}

// Integer dots are stored as i32 before scalar signed conversion to f32.
// CHECK-LABEL: <i8_i32_to_fp32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
// CHECK: fcvt.s.w
// CHECK: fsw
func.func @i8_i32_to_fp32(%A: memref<128x64xi8>, %B: memref<128x64xi8>, %I: memref<128x128xi32>, %C: memref<128x128xf32>) {
  %zero = arith.constant 0 : index
  %a = xsai_ame.load %A : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.load %B : memref<128x64xi8> -> !xsai_ame.tile<b, 128x64, i8>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, i32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %r, %I : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  %integer = memref.load %I[%zero, %zero] : memref<128x128xi32>
  %converted = arith.sitofp %integer : i32 to f32
  memref.store %converted, %C[%zero, %zero] : memref<128x128xf32>
  return
}

// CHECK-LABEL: <e4_fp32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
func.func @e4_fp32(%A: memref<128x64xf8E4M3FN>, %B: memref<128x64xf8E4M3FN>, %C: memref<128x128xf32>) {
  %a = xsai_ame.load %A : memref<128x64xf8E4M3FN> -> !xsai_ame.tile<a, 128x64, fp8e4m3>
  %b = xsai_ame.load %B : memref<128x64xf8E4M3FN> -> !xsai_ame.tile<b, 128x64, fp8e4m3>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, fp8e4m3>, !xsai_ame.tile<b, 128x64, fp8e4m3>, !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}

// CHECK-LABEL: <e5_fp32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
func.func @e5_fp32(%A: memref<128x64xf8E5M2>, %B: memref<128x64xf8E5M2>, %C: memref<128x128xf32>) {
  %a = xsai_ame.load %A : memref<128x64xf8E5M2> -> !xsai_ame.tile<a, 128x64, fp8e5m2>
  %b = xsai_ame.load %B : memref<128x64xf8E5M2> -> !xsai_ame.tile<b, 128x64, fp8e5m2>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, fp8e5m2>, !xsai_ame.tile<b, 128x64, fp8e5m2>, !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}

// CHECK-LABEL: <fp16_fp32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
func.func @fp16_fp32(%A: memref<128x32xf16>, %B: memref<128x32xf16>, %C: memref<128x128xf32>) {
  %a = xsai_ame.load %A : memref<128x32xf16> -> !xsai_ame.tile<a, 128x32, fp16>
  %b = xsai_ame.load %B : memref<128x32xf16> -> !xsai_ame.tile<b, 128x32, fp16>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x32, fp16>, !xsai_ame.tile<b, 128x32, fp16>, !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}

// CHECK-LABEL: <bf16_fp32>:
// CHECK: mla
// CHECK: mlb
// CHECK: mzero
// CHECK: mmacc
// CHECK: msc
func.func @bf16_fp32(%A: memref<128x32xbf16>, %B: memref<128x32xbf16>, %C: memref<128x128xf32>) {
  %a = xsai_ame.load %A : memref<128x32xbf16> -> !xsai_ame.tile<a, 128x32, bf16>
  %b = xsai_ame.load %B : memref<128x32xbf16> -> !xsai_ame.tile<b, 128x32, bf16>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x32, bf16>, !xsai_ame.tile<b, 128x32, bf16>, !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %r, %C : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}

// ATTR: TagName: arch
// ATTR: Value: rv64{{.*}}zames1p0
