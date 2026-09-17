; RUN: buddy-llc %s -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -verify-machineinstrs -o - | \
; RUN:   FileCheck %s
;
; Eight accumulator chains are exactly the capacity of the FPGA register file,
; so the same kernel with eight accumulators must compile.  Together with
; qwen-w8a8-matrix-spill.ll this pins the capacity boundary.
;
; RUN: buddy-llc %s -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -O0 \
; RUN:   -verify-machineinstrs -o - | FileCheck %s --check-prefix=O0
; RUN: buddy-llc %s -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -O3 \
; RUN:   -verify-machineinstrs -o - | FileCheck %s --check-prefix=O3
;
; The register counts below are an -O2 property; the datapath itself must be
; present and verifiable at every optimisation level.
; O0: mlce32.m
; O0: mqma.b.mm
; O0: msce32.m
; O0-NOT: Can't store this register to stack slot
; O3: mlce32.m
; O3: mqma.b.mm
; O3: msce32.m
; O3-NOT: Can't store this register to stack slot
;
; CHECK: mlce32.m
; CHECK-COUNT-8: mqma.b.mm
; CHECK: msce32.m
; Explicit slots keep these eight chains in acc0..acc7 at every level.
target triple = "riscv64"
declare <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr, i64, i32 immarg)
declare <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32>, <vscale x 128 x i8>, <vscale x 128 x i8>)
declare void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32>, ptr, i64)
declare <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlae8.m.nxv128i8.i64(ptr, i64, i32 immarg)
declare <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlbe8.m.nxv128i8.i64(ptr, i64, i32 immarg)

define void @eight_accumulators(ptr %p) #0 {
entry:
  %a = call <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlae8.m.nxv128i8.i64(ptr %p, i64 8, i32 0)
  %b = call <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlbe8.m.nxv128i8.i64(ptr %p, i64 8, i32 4)
  %s0 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 0)
  %s1 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 1)
  %s2 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 2)
  %s3 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 3)
  %s4 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 4)
  %s5 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 5)
  %s6 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 6)
  %s7 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 7)
  %r0 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s0, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r1 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s1, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r2 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s2, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r3 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s3, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r4 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s4, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r5 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s5, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r6 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s6, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r7 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s7, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r0, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r1, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r2, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r3, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r4, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r5, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r6, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r7, ptr %p, i64 4)
  ret void
}
attributes #0 = { "target-features"="+xboscame-fpga" }
