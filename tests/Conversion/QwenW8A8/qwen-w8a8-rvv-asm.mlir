// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame='target=qwen3-fpga' --lower-bosc-ame --cse \
// RUN:   --expand-strided-metadata --lower-affine \
// RUN:   --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm="vector-transpose-lowering=eltwise" \
// RUN:   --convert-vector-to-scf \
// RUN:   --convert-vector-to-llvm="vector-transpose-lowering=eltwise" \
// RUN:   --convert-ub-to-llvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-complex-to-llvm \
// RUN:   --convert-index-to-llvm --memref-expand --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm --convert-arith-to-llvm \
// RUN:   --reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: llc %t.ll -O0 -verify-machineinstrs -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -o - | FileCheck %s --check-prefix=ASM
// RUN: llc %t.ll -O2 -verify-machineinstrs -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -o - | FileCheck %s --check-prefix=ASM
// RUN: llc %t.ll -O3 -verify-machineinstrs -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame -o - | FileCheck %s --check-prefix=ASM

module {
  func.func @decode_pair_rvv(
      %xq: memref<1x512xi8>, %xs: memref<1x1xf32>,
      %wq: memref<2x1x64x512xi8>, %ws: memref<1x128xf32>,
      %output: memref<1x128xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x512xi8>, memref<1x1xf32>, memref<2x1x64x512xi8>,
         memref<1x128xf32>, memref<1x128xf32>) -> ()
    return
  }

  func.func @prefill_pair_rvv(
      %xq: memref<32x512xi8>, %xs: memref<32x1xf32>,
      %wq: memref<1x1x64x512xi8>, %ws: memref<1x64xf32>,
      %output: memref<32x64xf32>) {
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 512 : i64, weight_layout = "ame_outblk64"} :
        (memref<32x512xi8>, memref<32x1xf32>, memref<1x1x64x512xi8>,
         memref<1x64xf32>, memref<32x64xf32>) -> ()
    return
  }
}

// The FPGA register-file convention is a prototype convention: the MLIR target
// (bosc_ame.target = "qwen3-fpga") carries it into the LLVM IR as the
// `+xboscame-fpga` target feature, so llc needs no extra -mattr flag here.  The
// default +xboscame target keeps upstream's mapping.
// ASM-LABEL: decode_pair_rvv:
// The destination zeroing and the scale accumulation stay on RVV.
// ASM-DAG: vsetivli
// ASM-DAG: vse32.v
// ASM: msettilen
// ASM: msettilek
// The 1x1 resynchronisation round trip.  The MMA result is deliberately
// overwritten by the following accumulator load, so it must survive codegen on
// the strength of writing accelerator state alone: MMA intrinsics are modelled
// as writing state for exactly this reason.
// ASM: msettype
// ASM: mlce32.m{{[ \t]+}}acc{{[0-9]+}}
// ASM: mlae8.m{{[ \t]+}}tr0
// ASM: mlbte8.m{{[ \t]+}}tr4
// ASM: mqma.b.mm{{[ \t]+}}acc{{[0-9]+}}, tr{{[0-9]+}}, tr{{[0-9]+}}
// ASM: msettype
// ASM: mlce32.m{{[ \t]+}}acc{{[0-9]+}}
// ASM: msce32.m{{[ \t]+}}acc{{[0-9]+}}
// ASM-NEXT: fence{{[ \t]+}}rw, rw
// The wide schedule keeps eight accumulator chains resident and writes them out
// once, after the whole K reduction.
// The wide 2A4B kernel issues one MMA per accumulator chain, so exactly eight
// accumulator-destination MMAs follow the resynchronisation round trip, and the
// eight chains cover the whole acc0..acc7 register file.
// ASM-COUNT-8: mqma.b.mm{{[ \t]+}}acc{{[0-9]+}},
// ASM: msce32.m{{[ \t]+}}acc{{[0-9]+}}
// ASM: fence{{[ \t]+}}rw, rw
// ASM: call{{[ \t]+}}{{.*}}buddy_w8a8_rvv_accumulate_n64
// ASM: call{{[ \t]+}}{{.*}}buddy_w8a8_rvv_accumulate_n64

// ASM-LABEL: prefill_pair_rvv:
// The resync precedes the 2A4B kernel and uses the same fixed A0/B4 slots.
// ASM: mqma.b.mm{{[ \t]+}}acc0, tr0, tr4
// ASM: fence{{[ \t]+}}rw, rw
// ASM: mqma.b.mm{{[ \t]+}}acc0, tr0, tr4
// ASM: mqma.b.mm{{[ \t]+}}acc4, tr2, tr4
// ASM: mqma.b.mm{{[ \t]+}}acc1, tr0, tr5
// ASM: mqma.b.mm{{[ \t]+}}acc5, tr2, tr5
// ASM: mqma.b.mm{{[ \t]+}}acc2, tr0, tr6
// ASM: mqma.b.mm{{[ \t]+}}acc6, tr2, tr6
// ASM: mqma.b.mm{{[ \t]+}}acc3, tr0, tr7
// ASM: mqma.b.mm{{[ \t]+}}acc7, tr2, tr7
// ASM: msce32.m{{[ \t]+}}acc7
// ASM-NEXT: fence{{[ \t]+}}rw, rw
