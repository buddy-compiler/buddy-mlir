; The diagnostic terminates the process, so the check runs a real shell script
; instead of a lit pipeline.
; RUN: bash %S/check-matrix-spill.sh buddy-llc %s
;
; The FPGA convention has exactly eight accumulator registers and no spill
; implementation: any schedule that needs a ninth resident accumulator must be
; rejected by the compiler instead of silently spilling (an `msce32.m` write
; back to memory is a numeric i32 -> f32 conversion and cannot be used as a
; lossless save).  Register pressure is therefore a compile-time error.
;
; CHECK: LLVM ERROR: BOSC AME matrix registers cannot be spilled
; 9 live accumulator chains but only acc0..acc7 exist: the allocator must spill
; one matrix register, which the FPGA convention rejects instead of silently
; corrupting the accumulator.
target triple = "riscv64"
declare <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr, i64, i32 immarg)
declare <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32>, <vscale x 128 x i8>, <vscale x 128 x i8>)
declare void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32>, ptr, i64)
declare <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlae8.m.nxv128i8.i64(ptr, i64, i32 immarg)
declare <vscale x 128 x i8> @llvm.riscv.bosc.fpga.mlbe8.m.nxv128i8.i64(ptr, i64, i32 immarg)

define void @nine_accumulators(ptr %p) #0 {
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
  %s8 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64(ptr %p, i64 4, i32 0)
  %r0 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s0, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r1 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s1, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r2 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s2, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r3 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s3, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r4 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s4, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r5 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s5, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r6 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s6, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r7 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s7, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  %r8 = call <vscale x 32 x i32> @llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8(<vscale x 32 x i32> %s8, <vscale x 128 x i8> %a, <vscale x 128 x i8> %b)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r0, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r1, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r2, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r3, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r4, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r5, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r6, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r7, ptr %p, i64 4)
  call void @llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64(<vscale x 32 x i32> %r8, ptr %p, i64 4)
  ret void
}
attributes #0 = { "target-features"="+xboscame-fpga" }
