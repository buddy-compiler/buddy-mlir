// RUN: buddy-opt %s --lower-qwen-w8a8-to-boscame --lower-bosc-ame --cse \
// RUN:   --expand-strided-metadata --lower-affine \
// RUN:   --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm="vector-transpose-lowering=eltwise" \
// RUN:   --convert-vector-to-scf \
// RUN:   --convert-vector-to-llvm="vector-transpose-lowering=eltwise" \
// RUN:   --convert-ub-to-llvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-complex-to-llvm \
// RUN:   --convert-index-to-llvm --memref-expand --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm --convert-arith-to-llvm \
// RUN:   --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | \
// RUN:   llc -mtriple=riscv64 -mattr=+m,+f,+d,+v,+xboscame | \
// RUN:   FileCheck %s --check-prefix=ASM

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
}

// ASM-LABEL: decode_pair_rvv:
// ASM: msettilen
// ASM: msettilek
// ASM: mlae8.m tr0
// ASM: mqma.b.mm acc7, tr0, tr7
// ASM: msce32.m acc7
// ASM-NEXT: fence rw, rw
// ASM: call {{.*}}buddy_w8a8_rvv_accumulate_n64
// ASM: call {{.*}}buddy_w8a8_rvv_accumulate_n64
