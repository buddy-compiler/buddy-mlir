// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='"ame.'
// RUN: buddy-translate -buddy-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xame -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM

// Every XAME intrinsic has an MLIR bridge with its LLVM overload signature.
// Each function checks translation and backend selection independently.
!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!a = !llvm.target<"riscv.ztt.acc", i32, 8, 8>
!pair = !llvm.struct<(!m, !m)>

llvm.func @ame_acquire(%arg0: i64) -> i64 {
  %result = "ame.intr.ame.acquire"(%arg0) :
      (i64) -> i64
  llvm.return %result : i64
}

llvm.func @ame_release() {
  "ame.intr.ame.release"() :
      () -> ()
  llvm.return
}

llvm.func @agettyp(%arg0: !a) -> i64 {
  %result = "ame.intr.agettyp"(%arg0) :
      (!a) -> i64
  llvm.return %result : i64
}

llvm.func @asettyp(%arg0: !a, %arg1: i64) -> !a {
  %result = "ame.intr.asettyp"(%arg0, %arg1) :
      (!a, i64) -> !a
  llvm.return %result : !a
}

llvm.func @mabs_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mabs.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mabsdiff_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mabsdiff.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mabsdiff_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mabsdiff.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @madd_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.madd.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @madd_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.madd.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mand_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mand.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mand_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mand.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mandnot_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mandnot.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mandnot_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mandnot.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmovge_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcmovge.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmovlt_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcmovlt.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmpge_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcmpge.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmpge_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mcmpge.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmplt_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcmplt.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcmplt_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mcmplt.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolbcast_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mcolbcast.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolgather_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcolgather.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolid_ew(%arg0: !m) -> !m {
  %result = "ame.intr.mcolid.ew"(%arg0) :
      (!m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolshift_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mcolshift.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolunzip_ew(%arg0: !m, %arg1: !m) -> !pair {
  %result = "ame.intr.mcolunzip.ew"(%arg0, %arg1) :
      (!m, !m) -> !pair
  llvm.return %result : !pair
}

llvm.func @mcolzip_ew(%arg0: !m, %arg1: !m) -> !pair {
  %result = "ame.intr.mcolzip.ew"(%arg0, %arg1) :
      (!m, !m) -> !pair
  llvm.return %result : !pair
}

llvm.func @mconv_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mconv.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mbcast_m_x(%arg0: !m, %arg1: i64, %arg2: i64) -> !m {
  %result = "ame.intr.mbcast.m.x"(%arg0, %arg1, %arg2) :
      (!m, i64, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mcos_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mcos.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mexp2_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mexp2.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mfrintm_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mfrintm.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mfrintn_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mfrintn.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mfrintp_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mfrintp.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mfrintz_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mfrintz.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mgettyp(%arg0: !m) -> i64 {
  %result = "ame.intr.mgettyp"(%arg0) :
      (!m) -> i64
  llvm.return %result : i64
}

llvm.func @mhdiff_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mhdiff.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mhdiff_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mhdiff.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mldexp_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mldexp.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mldexp_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mldexp.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mldexpacc_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mldexpacc.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mldexpacc_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mldexpacc.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mlog2_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mlog2.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mlog2sub_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mlog2sub.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mlog2sub_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mlog2sub.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mls_1r(%arg0: !m, %arg1: !llvm.ptr) -> !m {
  %result = "ame.intr.mls.1r"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> !m
  llvm.return %result : !m
}

llvm.func @mls_cm(%arg0: !m, %arg1: !llvm.ptr) -> !m {
  %result = "ame.intr.mls.cm"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> !m
  llvm.return %result : !m
}

llvm.func @mls_rm(%arg0: !m, %arg1: !llvm.ptr) -> !m {
  %result = "ame.intr.mls.rm"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> !m
  llvm.return %result : !m
}

llvm.func @mls_st(%arg0: !m, %arg1: !llvm.ptr, %arg2: i64) -> !m {
  %result = "ame.intr.mls.st"(%arg0, %arg1, %arg2) :
      (!m, !llvm.ptr, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mls_tst(%arg0: !m, %arg1: !llvm.ptr, %arg2: i64) -> !m {
  %result = "ame.intr.mls.tst"(%arg0, %arg1, %arg2) :
      (!m, !llvm.ptr, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mmax_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmax.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmax_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmax.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmean_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmean.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmean_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmean.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmin_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmin.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmin_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmin.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmov_m_a(%arg0: !m, %arg1: !a) -> !m {
  %result = "ame.intr.mmov.m.a"(%arg0, %arg1) :
      (!m, !a) -> !m
  llvm.return %result : !m
}

llvm.func @mmov_a_m(%arg0: !a, %arg1: !m) -> !a {
  %result = "ame.intr.mmov.a.m"(%arg0, %arg1) :
      (!a, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmov_m_m(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mmov.m.m"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmove8_m_x(%arg0: !m, %arg1: i64, %arg2: i64) -> !m {
  %result = "ame.intr.mmove8.m.x"(%arg0, %arg1, %arg2) :
      (!m, i64, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mmove16_m_x(%arg0: !m, %arg1: i64, %arg2: i64) -> !m {
  %result = "ame.intr.mmove16.m.x"(%arg0, %arg1, %arg2) :
      (!m, i64, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mmove32_m_x(%arg0: !m, %arg1: i64, %arg2: i64) -> !m {
  %result = "ame.intr.mmove32.m.x"(%arg0, %arg1, %arg2) :
      (!m, i64, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mmove64_m_x(%arg0: !m, %arg1: i64, %arg2: i64) -> !m {
  %result = "ame.intr.mmove64.m.x"(%arg0, %arg1, %arg2) :
      (!m, i64, i64) -> !m
  llvm.return %result : !m
}

llvm.func @mmove8_x_m(%arg0: !m, %arg1: i64) -> i64 {
  %result = "ame.intr.mmove8.x.m"(%arg0, %arg1) :
      (!m, i64) -> i64
  llvm.return %result : i64
}

llvm.func @mmove16_x_m(%arg0: !m, %arg1: i64) -> i64 {
  %result = "ame.intr.mmove16.x.m"(%arg0, %arg1) :
      (!m, i64) -> i64
  llvm.return %result : i64
}

llvm.func @mmove32_x_m(%arg0: !m, %arg1: i64) -> i64 {
  %result = "ame.intr.mmove32.x.m"(%arg0, %arg1) :
      (!m, i64) -> i64
  llvm.return %result : i64
}

llvm.func @mmove64_x_m(%arg0: !m, %arg1: i64) -> i64 {
  %result = "ame.intr.mmove64.x.m"(%arg0, %arg1) :
      (!m, i64) -> i64
  llvm.return %result : i64
}

llvm.func @mmul_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmul.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmul_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmul.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulacc_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulacc.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulacc_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmulacc.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulacc_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmulacc.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulaccneg_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulaccneg.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulaccneg_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmulaccneg.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulaccneg_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmulaccneg.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmuladd_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmuladd.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmuladd_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmuladd.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulatacc_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulatacc.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulataccneg_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulataccneg.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulbtacc_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulbtacc.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulbtaccneg_2d(%arg0: !a, %arg1: !m, %arg2: !m) -> !a {
  %result = "ame.intr.mmulbtaccneg.2d"(%arg0, %arg1, %arg2) :
      (!a, !m, !m) -> !a
  llvm.return %result : !a
}

llvm.func @mmulneg_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmulneg.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulneg_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmulneg.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulsub_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mmulsub.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mmulsub_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mmulsub.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mor_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mor.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mor_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mor.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mornot_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mornot.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mornot_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mornot.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mpack_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mpack.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mprefixadd_col(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mprefixadd.col"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mprefixadd_row(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mprefixadd.row"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mprefixmax_col(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mprefixmax.col"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mprefixmax_row(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mprefixmax.row"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrdexp_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mrdexp.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrdexpacc_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mrdexpacc.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrec_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mrec.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreduceadd_col(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreduceadd.col"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreduceadd_row(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreduceadd.row"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreducemax_col(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreducemax.col"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreducemax_row(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreducemax.row"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreducemin_col(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreducemin.col"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mreducemin_row(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mreducemin.row"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowbcast_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mrowbcast.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowgather_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mrowgather.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowid_ew(%arg0: !m) -> !m {
  %result = "ame.intr.mrowid.ew"(%arg0) :
      (!m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowshift_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mrowshift.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowunzip_ew(%arg0: !m, %arg1: !m) -> !pair {
  %result = "ame.intr.mrowunzip.ew"(%arg0, %arg1) :
      (!m, !m) -> !pair
  llvm.return %result : !pair
}

llvm.func @mrowzip_ew(%arg0: !m, %arg1: !m) -> !pair {
  %result = "ame.intr.mrowzip.ew"(%arg0, %arg1) :
      (!m, !m) -> !pair
  llvm.return %result : !pair
}

llvm.func @mrsqrt_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mrsqrt.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowscatadd_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mrowscatadd.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolscatadd_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcolscatadd.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mrowscatmax_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mrowscatmax.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mcolscatmax_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mcolscatmax.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mselge_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mselge.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msellt_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msellt.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msettyp(%arg0: !m, %arg1: i64) -> !m {
  %result = "ame.intr.msettyp"(%arg0, %arg1) :
      (!m, i64) -> !m
  llvm.return %result : !m
}

llvm.func @msin_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.msin.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msll_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msll.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msll_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.msll.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msqrt_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.msqrt.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msra_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msra.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msra_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.msra.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msrl_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msrl.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msrl_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.msrl.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mss_1r(%arg0: !m, %arg1: !llvm.ptr) {
  "ame.intr.mss.1r"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @mss_cm(%arg0: !m, %arg1: !llvm.ptr) {
  "ame.intr.mss.cm"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @mss_rm(%arg0: !m, %arg1: !llvm.ptr) {
  "ame.intr.mss.rm"(%arg0, %arg1) :
      (!m, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @mss_st(%arg0: !m, %arg1: !llvm.ptr, %arg2: i64) {
  "ame.intr.mss.st"(%arg0, %arg1, %arg2) :
      (!m, !llvm.ptr, i64) -> ()
  llvm.return
}

llvm.func @mss_tst(%arg0: !m, %arg1: !llvm.ptr, %arg2: i64) {
  "ame.intr.mss.tst"(%arg0, %arg1, %arg2) :
      (!m, !llvm.ptr, i64) -> ()
  llvm.return
}

llvm.func @msub_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msub.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msub_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.msub.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msublog2_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.msublog2.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @msublog2_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.msublog2.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mtanh_ew(%arg0: !m, %arg1: !m) -> !m {
  %result = "ame.intr.mtanh.ew"(%arg0, %arg1) :
      (!m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @munpack_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.munpack.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mxor_ew(%arg0: !m, %arg1: !m, %arg2: !m) -> !m {
  %result = "ame.intr.mxor.ew"(%arg0, %arg1, %arg2) :
      (!m, !m, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mxor_ew_x(%arg0: !m, %arg1: i64, %arg2: !m) -> !m {
  %result = "ame.intr.mxor.ew.x"(%arg0, %arg1, %arg2) :
      (!m, i64, !m) -> !m
  llvm.return %result : !m
}

llvm.func @mzero_2d_acc(%arg0: !a) -> !a {
  %result = "ame.intr.mzero.2d.acc"(%arg0) :
      (!a) -> !a
  llvm.return %result : !a
}

llvm.func @mzero_2d_m(%arg0: !m) -> !m {
  %result = "ame.intr.mzero.2d.m"(%arg0) :
      (!m) -> !m
  llvm.return %result : !m
}

// LLVM-LABEL: @ame_acquire(
// LLVM: call {{.*}}@llvm.riscv.ztt.ame.acquire
// LLVM-LABEL: @ame_release(
// LLVM: call {{.*}}@llvm.riscv.ztt.ame.release
// LLVM-LABEL: @agettyp(
// LLVM: call {{.*}}@llvm.riscv.ztt.agettyp
// LLVM-LABEL: @asettyp(
// LLVM: call {{.*}}@llvm.riscv.ztt.asettyp
// LLVM-LABEL: @mabs_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mabs.ew
// LLVM-LABEL: @mabsdiff_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mabsdiff.ew
// LLVM-LABEL: @mabsdiff_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mabsdiff.ew.x
// LLVM-LABEL: @madd_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.madd.ew
// LLVM-LABEL: @madd_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.madd.ew.x
// LLVM-LABEL: @mand_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mand.ew
// LLVM-LABEL: @mand_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mand.ew.x
// LLVM-LABEL: @mandnot_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mandnot.ew
// LLVM-LABEL: @mandnot_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mandnot.ew.x
// LLVM-LABEL: @mcmovge_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmovge.ew
// LLVM-LABEL: @mcmovlt_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmovlt.ew
// LLVM-LABEL: @mcmpge_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmpge.ew
// LLVM-LABEL: @mcmpge_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmpge.ew.x
// LLVM-LABEL: @mcmplt_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmplt.ew
// LLVM-LABEL: @mcmplt_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcmplt.ew.x
// LLVM-LABEL: @mcolbcast_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolbcast.ew.x
// LLVM-LABEL: @mcolgather_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolgather.ew
// LLVM-LABEL: @mcolid_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolid.ew
// LLVM-LABEL: @mcolshift_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolshift.ew.x
// LLVM-LABEL: @mcolunzip_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolunzip.ew
// LLVM-LABEL: @mcolzip_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolzip.ew
// LLVM-LABEL: @mconv_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mconv.ew
// LLVM-LABEL: @mbcast_m_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mbcast.m.x
// LLVM-LABEL: @mcos_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcos.ew
// LLVM-LABEL: @mexp2_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mexp2.ew
// LLVM-LABEL: @mfrintm_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mfrintm.ew
// LLVM-LABEL: @mfrintn_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mfrintn.ew
// LLVM-LABEL: @mfrintp_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mfrintp.ew
// LLVM-LABEL: @mfrintz_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mfrintz.ew
// LLVM-LABEL: @mgettyp(
// LLVM: call {{.*}}@llvm.riscv.ztt.mgettyp
// LLVM-LABEL: @mhdiff_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mhdiff.ew
// LLVM-LABEL: @mhdiff_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mhdiff.ew.x
// LLVM-LABEL: @mldexp_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mldexp.ew
// LLVM-LABEL: @mldexp_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mldexp.ew.x
// LLVM-LABEL: @mldexpacc_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mldexpacc.ew
// LLVM-LABEL: @mldexpacc_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mldexpacc.ew.x
// LLVM-LABEL: @mlog2_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mlog2.ew
// LLVM-LABEL: @mlog2sub_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mlog2sub.ew
// LLVM-LABEL: @mlog2sub_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mlog2sub.ew.x
// LLVM-LABEL: @mls_1r(
// LLVM: call {{.*}}@llvm.riscv.ztt.mls.1r
// LLVM-LABEL: @mls_cm(
// LLVM: call {{.*}}@llvm.riscv.ztt.mls.cm
// LLVM-LABEL: @mls_rm(
// LLVM: call {{.*}}@llvm.riscv.ztt.mls.rm
// LLVM-LABEL: @mls_st(
// LLVM: call {{.*}}@llvm.riscv.ztt.mls.st
// LLVM-LABEL: @mls_tst(
// LLVM: call {{.*}}@llvm.riscv.ztt.mls.tst
// LLVM-LABEL: @mmax_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmax.ew
// LLVM-LABEL: @mmax_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmax.ew.x
// LLVM-LABEL: @mmean_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmean.ew
// LLVM-LABEL: @mmean_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmean.ew.x
// LLVM-LABEL: @mmin_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmin.ew
// LLVM-LABEL: @mmin_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmin.ew.x
// LLVM-LABEL: @mmov_m_a(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmov.m.a
// LLVM-LABEL: @mmov_a_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmov.a.m
// LLVM-LABEL: @mmov_m_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmov.m.m
// LLVM-LABEL: @mmove8_m_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove8.m.x
// LLVM-LABEL: @mmove16_m_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove16.m.x
// LLVM-LABEL: @mmove32_m_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove32.m.x
// LLVM-LABEL: @mmove64_m_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove64.m.x
// LLVM-LABEL: @mmove8_x_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove8.x.m
// LLVM-LABEL: @mmove16_x_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove16.x.m
// LLVM-LABEL: @mmove32_x_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove32.x.m
// LLVM-LABEL: @mmove64_x_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmove64.x.m
// LLVM-LABEL: @mmul_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmul.ew
// LLVM-LABEL: @mmul_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmul.ew.x
// LLVM-LABEL: @mmulacc_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulacc.2d
// LLVM-LABEL: @mmulacc_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulacc.ew
// LLVM-LABEL: @mmulacc_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulacc.ew.x
// LLVM-LABEL: @mmulaccneg_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulaccneg.2d
// LLVM-LABEL: @mmulaccneg_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulaccneg.ew
// LLVM-LABEL: @mmulaccneg_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulaccneg.ew.x
// LLVM-LABEL: @mmuladd_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmuladd.ew
// LLVM-LABEL: @mmuladd_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmuladd.ew.x
// LLVM-LABEL: @mmulatacc_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulatacc.2d
// LLVM-LABEL: @mmulataccneg_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulataccneg.2d
// LLVM-LABEL: @mmulbtacc_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulbtacc.2d
// LLVM-LABEL: @mmulbtaccneg_2d(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulbtaccneg.2d
// LLVM-LABEL: @mmulneg_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulneg.ew
// LLVM-LABEL: @mmulneg_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulneg.ew.x
// LLVM-LABEL: @mmulsub_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulsub.ew
// LLVM-LABEL: @mmulsub_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mmulsub.ew.x
// LLVM-LABEL: @mor_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mor.ew
// LLVM-LABEL: @mor_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mor.ew.x
// LLVM-LABEL: @mornot_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mornot.ew
// LLVM-LABEL: @mornot_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mornot.ew.x
// LLVM-LABEL: @mpack_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mpack.ew.x
// LLVM-LABEL: @mprefixadd_col(
// LLVM: call {{.*}}@llvm.riscv.ztt.mprefixadd.col
// LLVM-LABEL: @mprefixadd_row(
// LLVM: call {{.*}}@llvm.riscv.ztt.mprefixadd.row
// LLVM-LABEL: @mprefixmax_col(
// LLVM: call {{.*}}@llvm.riscv.ztt.mprefixmax.col
// LLVM-LABEL: @mprefixmax_row(
// LLVM: call {{.*}}@llvm.riscv.ztt.mprefixmax.row
// LLVM-LABEL: @mrdexp_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrdexp.ew
// LLVM-LABEL: @mrdexpacc_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrdexpacc.ew
// LLVM-LABEL: @mrec_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrec.ew
// LLVM-LABEL: @mreduceadd_col(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreduceadd.col
// LLVM-LABEL: @mreduceadd_row(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreduceadd.row
// LLVM-LABEL: @mreducemax_col(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreducemax.col
// LLVM-LABEL: @mreducemax_row(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreducemax.row
// LLVM-LABEL: @mreducemin_col(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreducemin.col
// LLVM-LABEL: @mreducemin_row(
// LLVM: call {{.*}}@llvm.riscv.ztt.mreducemin.row
// LLVM-LABEL: @mrowbcast_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowbcast.ew.x
// LLVM-LABEL: @mrowgather_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowgather.ew
// LLVM-LABEL: @mrowid_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowid.ew
// LLVM-LABEL: @mrowshift_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowshift.ew.x
// LLVM-LABEL: @mrowunzip_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowunzip.ew
// LLVM-LABEL: @mrowzip_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowzip.ew
// LLVM-LABEL: @mrsqrt_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrsqrt.ew
// LLVM-LABEL: @mrowscatadd_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowscatadd.ew
// LLVM-LABEL: @mcolscatadd_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolscatadd.ew
// LLVM-LABEL: @mrowscatmax_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mrowscatmax.ew
// LLVM-LABEL: @mcolscatmax_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mcolscatmax.ew
// LLVM-LABEL: @mselge_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mselge.ew
// LLVM-LABEL: @msellt_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msellt.ew
// LLVM-LABEL: @msettyp(
// LLVM: call {{.*}}@llvm.riscv.ztt.msettyp
// LLVM-LABEL: @msin_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msin.ew
// LLVM-LABEL: @msll_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msll.ew
// LLVM-LABEL: @msll_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.msll.ew.x
// LLVM-LABEL: @msqrt_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msqrt.ew
// LLVM-LABEL: @msra_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msra.ew
// LLVM-LABEL: @msra_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.msra.ew.x
// LLVM-LABEL: @msrl_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msrl.ew
// LLVM-LABEL: @msrl_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.msrl.ew.x
// LLVM-LABEL: @mss_1r(
// LLVM: call {{.*}}@llvm.riscv.ztt.mss.1r
// LLVM-LABEL: @mss_cm(
// LLVM: call {{.*}}@llvm.riscv.ztt.mss.cm
// LLVM-LABEL: @mss_rm(
// LLVM: call {{.*}}@llvm.riscv.ztt.mss.rm
// LLVM-LABEL: @mss_st(
// LLVM: call {{.*}}@llvm.riscv.ztt.mss.st
// LLVM-LABEL: @mss_tst(
// LLVM: call {{.*}}@llvm.riscv.ztt.mss.tst
// LLVM-LABEL: @msub_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msub.ew
// LLVM-LABEL: @msub_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.msub.ew.x
// LLVM-LABEL: @msublog2_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.msublog2.ew
// LLVM-LABEL: @msublog2_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.msublog2.ew.x
// LLVM-LABEL: @mtanh_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mtanh.ew
// LLVM-LABEL: @munpack_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.munpack.ew.x
// LLVM-LABEL: @mxor_ew(
// LLVM: call {{.*}}@llvm.riscv.ztt.mxor.ew
// LLVM-LABEL: @mxor_ew_x(
// LLVM: call {{.*}}@llvm.riscv.ztt.mxor.ew.x
// LLVM-LABEL: @mzero_2d_acc(
// LLVM: call {{.*}}@llvm.riscv.ztt.mzero.2d.acc
// LLVM-LABEL: @mzero_2d_m(
// LLVM: call {{.*}}@llvm.riscv.ztt.mzero.2d.m

// ASM-LABEL: ame_acquire:
// ASM: ame.acquire
// ASM-LABEL: ame_release:
// ASM: ame.release
// ASM-LABEL: agettyp:
// ASM: agettyp
// ASM-LABEL: asettyp:
// ASM: asettyp
// ASM-LABEL: mabs_ew:
// ASM: mabs.ew
// ASM-LABEL: mabsdiff_ew:
// ASM: mabsdiff.ew
// ASM-LABEL: mabsdiff_ew_x:
// ASM: mabsdiff.ew.x
// ASM-LABEL: madd_ew:
// ASM: madd.ew
// ASM-LABEL: madd_ew_x:
// ASM: madd.ew.x
// ASM-LABEL: mand_ew:
// ASM: mand.ew
// ASM-LABEL: mand_ew_x:
// ASM: mand.ew.x
// ASM-LABEL: mandnot_ew:
// ASM: mandnot.ew
// ASM-LABEL: mandnot_ew_x:
// ASM: mandnot.ew.x
// ASM-LABEL: mcmovge_ew:
// ASM: mcmovge.ew
// ASM-LABEL: mcmovlt_ew:
// ASM: mcmovlt.ew
// ASM-LABEL: mcmpge_ew:
// ASM: mcmpge.ew
// ASM-LABEL: mcmpge_ew_x:
// ASM: mcmpge.ew.x
// ASM-LABEL: mcmplt_ew:
// ASM: mcmplt.ew
// ASM-LABEL: mcmplt_ew_x:
// ASM: mcmplt.ew.x
// ASM-LABEL: mcolbcast_ew_x:
// ASM: mcolbcast.ew.x
// ASM-LABEL: mcolgather_ew:
// ASM: mcolgather.ew
// ASM-LABEL: mcolid_ew:
// ASM: mcolid.ew
// ASM-LABEL: mcolshift_ew_x:
// ASM: mcolshift.ew.x
// ASM-LABEL: mcolunzip_ew:
// ASM: mcolunzip.ew
// ASM-LABEL: mcolzip_ew:
// ASM: mcolzip.ew
// ASM-LABEL: mconv_ew:
// ASM: mconv.ew
// ASM-LABEL: mbcast_m_x:
// ASM: mbcast.m.x
// ASM-LABEL: mcos_ew:
// ASM: mcos.ew
// ASM-LABEL: mexp2_ew:
// ASM: mexp2.ew
// ASM-LABEL: mfrintm_ew:
// ASM: mfrintm.ew
// ASM-LABEL: mfrintn_ew:
// ASM: mfrintn.ew
// ASM-LABEL: mfrintp_ew:
// ASM: mfrintp.ew
// ASM-LABEL: mfrintz_ew:
// ASM: mfrintz.ew
// ASM-LABEL: mgettyp:
// ASM: mgettyp
// ASM-LABEL: mhdiff_ew:
// ASM: mhdiff.ew
// ASM-LABEL: mhdiff_ew_x:
// ASM: mhdiff.ew.x
// ASM-LABEL: mldexp_ew:
// ASM: mldexp.ew
// ASM-LABEL: mldexp_ew_x:
// ASM: mldexp.ew.x
// ASM-LABEL: mldexpacc_ew:
// ASM: mldexpacc.ew
// ASM-LABEL: mldexpacc_ew_x:
// ASM: mldexpacc.ew.x
// ASM-LABEL: mlog2_ew:
// ASM: mlog2.ew
// ASM-LABEL: mlog2sub_ew:
// ASM: mlog2sub.ew
// ASM-LABEL: mlog2sub_ew_x:
// ASM: mlog2sub.ew.x
// ASM-LABEL: mls_1r:
// ASM: mls.1r
// ASM-LABEL: mls_cm:
// ASM: mls.cm
// ASM-LABEL: mls_rm:
// ASM: mls.rm
// ASM-LABEL: mls_st:
// ASM: mls.st
// ASM-LABEL: mls_tst:
// ASM: mls.tst
// ASM-LABEL: mmax_ew:
// ASM: mmax.ew
// ASM-LABEL: mmax_ew_x:
// ASM: mmax.ew.x
// ASM-LABEL: mmean_ew:
// ASM: mmean.ew
// ASM-LABEL: mmean_ew_x:
// ASM: mmean.ew.x
// ASM-LABEL: mmin_ew:
// ASM: mmin.ew
// ASM-LABEL: mmin_ew_x:
// ASM: mmin.ew.x
// ASM-LABEL: mmov_m_a:
// ASM: mmov.m.a
// ASM-LABEL: mmov_a_m:
// ASM: mmov.a.m
// ASM-LABEL: mmov_m_m:
// ASM: mmov.m.m
// ASM-LABEL: mmove8_m_x:
// ASM: mmove8.m.x
// ASM-LABEL: mmove16_m_x:
// ASM: mmove16.m.x
// ASM-LABEL: mmove32_m_x:
// ASM: mmove32.m.x
// ASM-LABEL: mmove64_m_x:
// ASM: mmove64.m.x
// ASM-LABEL: mmove8_x_m:
// ASM: mmove8.x.m
// ASM-LABEL: mmove16_x_m:
// ASM: mmove16.x.m
// ASM-LABEL: mmove32_x_m:
// ASM: mmove32.x.m
// ASM-LABEL: mmove64_x_m:
// ASM: mmove64.x.m
// ASM-LABEL: mmul_ew:
// ASM: mmul.ew
// ASM-LABEL: mmul_ew_x:
// ASM: mmul.ew.x
// ASM-LABEL: mmulacc_2d:
// ASM: mmulacc.2d
// ASM-LABEL: mmulacc_ew:
// ASM: mmulacc.ew
// ASM-LABEL: mmulacc_ew_x:
// ASM: mmulacc.ew.x
// ASM-LABEL: mmulaccneg_2d:
// ASM: mmulaccneg.2d
// ASM-LABEL: mmulaccneg_ew:
// ASM: mmulaccneg.ew
// ASM-LABEL: mmulaccneg_ew_x:
// ASM: mmulaccneg.ew.x
// ASM-LABEL: mmuladd_ew:
// ASM: mmuladd.ew
// ASM-LABEL: mmuladd_ew_x:
// ASM: mmuladd.ew.x
// ASM-LABEL: mmulatacc_2d:
// ASM: mmulatacc.2d
// ASM-LABEL: mmulataccneg_2d:
// ASM: mmulataccneg.2d
// ASM-LABEL: mmulbtacc_2d:
// ASM: mmulbtacc.2d
// ASM-LABEL: mmulbtaccneg_2d:
// ASM: mmulbtaccneg.2d
// ASM-LABEL: mmulneg_ew:
// ASM: mmulneg.ew
// ASM-LABEL: mmulneg_ew_x:
// ASM: mmulneg.ew.x
// ASM-LABEL: mmulsub_ew:
// ASM: mmulsub.ew
// ASM-LABEL: mmulsub_ew_x:
// ASM: mmulsub.ew.x
// ASM-LABEL: mor_ew:
// ASM: mor.ew
// ASM-LABEL: mor_ew_x:
// ASM: mor.ew.x
// ASM-LABEL: mornot_ew:
// ASM: mornot.ew
// ASM-LABEL: mornot_ew_x:
// ASM: mornot.ew.x
// ASM-LABEL: mpack_ew_x:
// ASM: mpack.ew.x
// ASM-LABEL: mprefixadd_col:
// ASM: mprefixadd.col
// ASM-LABEL: mprefixadd_row:
// ASM: mprefixadd.row
// ASM-LABEL: mprefixmax_col:
// ASM: mprefixmax.col
// ASM-LABEL: mprefixmax_row:
// ASM: mprefixmax.row
// ASM-LABEL: mrdexp_ew:
// ASM: mrdexp.ew
// ASM-LABEL: mrdexpacc_ew:
// ASM: mrdexpacc.ew
// ASM-LABEL: mrec_ew:
// ASM: mrec.ew
// ASM-LABEL: mreduceadd_col:
// ASM: mreduceadd.col
// ASM-LABEL: mreduceadd_row:
// ASM: mreduceadd.row
// ASM-LABEL: mreducemax_col:
// ASM: mreducemax.col
// ASM-LABEL: mreducemax_row:
// ASM: mreducemax.row
// ASM-LABEL: mreducemin_col:
// ASM: mreducemin.col
// ASM-LABEL: mreducemin_row:
// ASM: mreducemin.row
// ASM-LABEL: mrowbcast_ew_x:
// ASM: mrowbcast.ew.x
// ASM-LABEL: mrowgather_ew:
// ASM: mrowgather.ew
// ASM-LABEL: mrowid_ew:
// ASM: mrowid.ew
// ASM-LABEL: mrowshift_ew_x:
// ASM: mrowshift.ew.x
// ASM-LABEL: mrowunzip_ew:
// ASM: mrowunzip.ew
// ASM-LABEL: mrowzip_ew:
// ASM: mrowzip.ew
// ASM-LABEL: mrsqrt_ew:
// ASM: mrsqrt.ew
// ASM-LABEL: mrowscatadd_ew:
// ASM: mrowscatadd.ew
// ASM-LABEL: mcolscatadd_ew:
// ASM: mcolscatadd.ew
// ASM-LABEL: mrowscatmax_ew:
// ASM: mrowscatmax.ew
// ASM-LABEL: mcolscatmax_ew:
// ASM: mcolscatmax.ew
// ASM-LABEL: mselge_ew:
// ASM: mselge.ew
// ASM-LABEL: msellt_ew:
// ASM: msellt.ew
// ASM-LABEL: msettyp:
// ASM: msettyp
// ASM-LABEL: msin_ew:
// ASM: msin.ew
// ASM-LABEL: msll_ew:
// ASM: msll.ew
// ASM-LABEL: msll_ew_x:
// ASM: msll.ew.x
// ASM-LABEL: msqrt_ew:
// ASM: msqrt.ew
// ASM-LABEL: msra_ew:
// ASM: msra.ew
// ASM-LABEL: msra_ew_x:
// ASM: msra.ew.x
// ASM-LABEL: msrl_ew:
// ASM: msrl.ew
// ASM-LABEL: msrl_ew_x:
// ASM: msrl.ew.x
// ASM-LABEL: mss_1r:
// ASM: mss.1r
// ASM-LABEL: mss_cm:
// ASM: mss.cm
// ASM-LABEL: mss_rm:
// ASM: mss.rm
// ASM-LABEL: mss_st:
// ASM: mss.st
// ASM-LABEL: mss_tst:
// ASM: mss.tst
// ASM-LABEL: msub_ew:
// ASM: msub.ew
// ASM-LABEL: msub_ew_x:
// ASM: msub.ew.x
// ASM-LABEL: msublog2_ew:
// ASM: msublog2.ew
// ASM-LABEL: msublog2_ew_x:
// ASM: msublog2.ew.x
// ASM-LABEL: mtanh_ew:
// ASM: mtanh.ew
// ASM-LABEL: munpack_ew_x:
// ASM: munpack.ew.x
// ASM-LABEL: mxor_ew:
// ASM: mxor.ew
// ASM-LABEL: mxor_ew_x:
// ASM: mxor.ew.x
// ASM-LABEL: mzero_2d_acc:
// ASM: mzero.2d.acc
// ASM-LABEL: mzero_2d_m:
// ASM: mzero.2d.m

// LOWER-LABEL: llvm.func @ame_acquire(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.acquire"
// LOWER-LABEL: llvm.func @ame_release(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.release"
// LOWER-LABEL: llvm.func @agettyp(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.agettyp"
// LOWER-LABEL: llvm.func @asettyp(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.asettyp"
// LOWER-LABEL: llvm.func @mabs_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mabs.ew"
// LOWER-LABEL: llvm.func @mabsdiff_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mabsdiff.ew"
// LOWER-LABEL: llvm.func @mabsdiff_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mabsdiff.ew.x"
// LOWER-LABEL: llvm.func @madd_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.madd.ew"
// LOWER-LABEL: llvm.func @madd_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.madd.ew.x"
// LOWER-LABEL: llvm.func @mand_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mand.ew"
// LOWER-LABEL: llvm.func @mand_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mand.ew.x"
// LOWER-LABEL: llvm.func @mandnot_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mandnot.ew"
// LOWER-LABEL: llvm.func @mandnot_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mandnot.ew.x"
// LOWER-LABEL: llvm.func @mcmovge_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmovge.ew"
// LOWER-LABEL: llvm.func @mcmovlt_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmovlt.ew"
// LOWER-LABEL: llvm.func @mcmpge_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmpge.ew"
// LOWER-LABEL: llvm.func @mcmpge_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmpge.ew.x"
// LOWER-LABEL: llvm.func @mcmplt_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmplt.ew"
// LOWER-LABEL: llvm.func @mcmplt_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcmplt.ew.x"
// LOWER-LABEL: llvm.func @mcolbcast_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolbcast.ew.x"
// LOWER-LABEL: llvm.func @mcolgather_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolgather.ew"
// LOWER-LABEL: llvm.func @mcolid_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolid.ew"
// LOWER-LABEL: llvm.func @mcolshift_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolshift.ew.x"
// LOWER-LABEL: llvm.func @mcolunzip_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolunzip.ew"
// LOWER-LABEL: llvm.func @mcolzip_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolzip.ew"
// LOWER-LABEL: llvm.func @mconv_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mconv.ew"
// LOWER-LABEL: llvm.func @mbcast_m_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mbcast.m.x"
// LOWER-LABEL: llvm.func @mcos_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcos.ew"
// LOWER-LABEL: llvm.func @mexp2_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mexp2.ew"
// LOWER-LABEL: llvm.func @mfrintm_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mfrintm.ew"
// LOWER-LABEL: llvm.func @mfrintn_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mfrintn.ew"
// LOWER-LABEL: llvm.func @mfrintp_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mfrintp.ew"
// LOWER-LABEL: llvm.func @mfrintz_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mfrintz.ew"
// LOWER-LABEL: llvm.func @mgettyp(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mgettyp"
// LOWER-LABEL: llvm.func @mhdiff_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mhdiff.ew"
// LOWER-LABEL: llvm.func @mhdiff_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mhdiff.ew.x"
// LOWER-LABEL: llvm.func @mldexp_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mldexp.ew"
// LOWER-LABEL: llvm.func @mldexp_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mldexp.ew.x"
// LOWER-LABEL: llvm.func @mldexpacc_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mldexpacc.ew"
// LOWER-LABEL: llvm.func @mldexpacc_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mldexpacc.ew.x"
// LOWER-LABEL: llvm.func @mlog2_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mlog2.ew"
// LOWER-LABEL: llvm.func @mlog2sub_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mlog2sub.ew"
// LOWER-LABEL: llvm.func @mlog2sub_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mlog2sub.ew.x"
// LOWER-LABEL: llvm.func @mls_1r(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.1r"
// LOWER-LABEL: llvm.func @mls_cm(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.cm"
// LOWER-LABEL: llvm.func @mls_rm(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.rm"
// LOWER-LABEL: llvm.func @mls_st(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.st"
// LOWER-LABEL: llvm.func @mls_tst(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.tst"
// LOWER-LABEL: llvm.func @mmax_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmax.ew"
// LOWER-LABEL: llvm.func @mmax_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmax.ew.x"
// LOWER-LABEL: llvm.func @mmean_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmean.ew"
// LOWER-LABEL: llvm.func @mmean_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmean.ew.x"
// LOWER-LABEL: llvm.func @mmin_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmin.ew"
// LOWER-LABEL: llvm.func @mmin_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmin.ew.x"
// LOWER-LABEL: llvm.func @mmov_m_a(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.m.a"
// LOWER-LABEL: llvm.func @mmov_a_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.a.m"
// LOWER-LABEL: llvm.func @mmov_m_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.m.m"
// LOWER-LABEL: llvm.func @mmove8_m_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove8.m.x"
// LOWER-LABEL: llvm.func @mmove16_m_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove16.m.x"
// LOWER-LABEL: llvm.func @mmove32_m_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove32.m.x"
// LOWER-LABEL: llvm.func @mmove64_m_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove64.m.x"
// LOWER-LABEL: llvm.func @mmove8_x_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove8.x.m"
// LOWER-LABEL: llvm.func @mmove16_x_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove16.x.m"
// LOWER-LABEL: llvm.func @mmove32_x_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove32.x.m"
// LOWER-LABEL: llvm.func @mmove64_x_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove64.x.m"
// LOWER-LABEL: llvm.func @mmul_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmul.ew"
// LOWER-LABEL: llvm.func @mmul_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmul.ew.x"
// LOWER-LABEL: llvm.func @mmulacc_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.2d"
// LOWER-LABEL: llvm.func @mmulacc_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.ew"
// LOWER-LABEL: llvm.func @mmulacc_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.ew.x"
// LOWER-LABEL: llvm.func @mmulaccneg_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulaccneg.2d"
// LOWER-LABEL: llvm.func @mmulaccneg_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulaccneg.ew"
// LOWER-LABEL: llvm.func @mmulaccneg_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulaccneg.ew.x"
// LOWER-LABEL: llvm.func @mmuladd_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmuladd.ew"
// LOWER-LABEL: llvm.func @mmuladd_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmuladd.ew.x"
// LOWER-LABEL: llvm.func @mmulatacc_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulatacc.2d"
// LOWER-LABEL: llvm.func @mmulataccneg_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulataccneg.2d"
// LOWER-LABEL: llvm.func @mmulbtacc_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulbtacc.2d"
// LOWER-LABEL: llvm.func @mmulbtaccneg_2d(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulbtaccneg.2d"
// LOWER-LABEL: llvm.func @mmulneg_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulneg.ew"
// LOWER-LABEL: llvm.func @mmulneg_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulneg.ew.x"
// LOWER-LABEL: llvm.func @mmulsub_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulsub.ew"
// LOWER-LABEL: llvm.func @mmulsub_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulsub.ew.x"
// LOWER-LABEL: llvm.func @mor_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mor.ew"
// LOWER-LABEL: llvm.func @mor_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mor.ew.x"
// LOWER-LABEL: llvm.func @mornot_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mornot.ew"
// LOWER-LABEL: llvm.func @mornot_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mornot.ew.x"
// LOWER-LABEL: llvm.func @mpack_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mpack.ew.x"
// LOWER-LABEL: llvm.func @mprefixadd_col(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mprefixadd.col"
// LOWER-LABEL: llvm.func @mprefixadd_row(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mprefixadd.row"
// LOWER-LABEL: llvm.func @mprefixmax_col(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mprefixmax.col"
// LOWER-LABEL: llvm.func @mprefixmax_row(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mprefixmax.row"
// LOWER-LABEL: llvm.func @mrdexp_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrdexp.ew"
// LOWER-LABEL: llvm.func @mrdexpacc_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrdexpacc.ew"
// LOWER-LABEL: llvm.func @mrec_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrec.ew"
// LOWER-LABEL: llvm.func @mreduceadd_col(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreduceadd.col"
// LOWER-LABEL: llvm.func @mreduceadd_row(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreduceadd.row"
// LOWER-LABEL: llvm.func @mreducemax_col(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreducemax.col"
// LOWER-LABEL: llvm.func @mreducemax_row(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreducemax.row"
// LOWER-LABEL: llvm.func @mreducemin_col(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreducemin.col"
// LOWER-LABEL: llvm.func @mreducemin_row(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mreducemin.row"
// LOWER-LABEL: llvm.func @mrowbcast_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowbcast.ew.x"
// LOWER-LABEL: llvm.func @mrowgather_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowgather.ew"
// LOWER-LABEL: llvm.func @mrowid_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowid.ew"
// LOWER-LABEL: llvm.func @mrowshift_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowshift.ew.x"
// LOWER-LABEL: llvm.func @mrowunzip_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowunzip.ew"
// LOWER-LABEL: llvm.func @mrowzip_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowzip.ew"
// LOWER-LABEL: llvm.func @mrsqrt_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrsqrt.ew"
// LOWER-LABEL: llvm.func @mrowscatadd_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowscatadd.ew"
// LOWER-LABEL: llvm.func @mcolscatadd_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolscatadd.ew"
// LOWER-LABEL: llvm.func @mrowscatmax_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mrowscatmax.ew"
// LOWER-LABEL: llvm.func @mcolscatmax_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolscatmax.ew"
// LOWER-LABEL: llvm.func @mselge_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mselge.ew"
// LOWER-LABEL: llvm.func @msellt_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msellt.ew"
// LOWER-LABEL: llvm.func @msettyp(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msettyp"
// LOWER-LABEL: llvm.func @msin_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msin.ew"
// LOWER-LABEL: llvm.func @msll_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msll.ew"
// LOWER-LABEL: llvm.func @msll_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msll.ew.x"
// LOWER-LABEL: llvm.func @msqrt_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msqrt.ew"
// LOWER-LABEL: llvm.func @msra_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msra.ew"
// LOWER-LABEL: llvm.func @msra_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msra.ew.x"
// LOWER-LABEL: llvm.func @msrl_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msrl.ew"
// LOWER-LABEL: llvm.func @msrl_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msrl.ew.x"
// LOWER-LABEL: llvm.func @mss_1r(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.1r"
// LOWER-LABEL: llvm.func @mss_cm(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.cm"
// LOWER-LABEL: llvm.func @mss_rm(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.rm"
// LOWER-LABEL: llvm.func @mss_st(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.st"
// LOWER-LABEL: llvm.func @mss_tst(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.tst"
// LOWER-LABEL: llvm.func @msub_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msub.ew"
// LOWER-LABEL: llvm.func @msub_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msub.ew.x"
// LOWER-LABEL: llvm.func @msublog2_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msublog2.ew"
// LOWER-LABEL: llvm.func @msublog2_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msublog2.ew.x"
// LOWER-LABEL: llvm.func @mtanh_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mtanh.ew"
// LOWER-LABEL: llvm.func @munpack_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.munpack.ew.x"
// LOWER-LABEL: llvm.func @mxor_ew(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mxor.ew"
// LOWER-LABEL: llvm.func @mxor_ew_x(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mxor.ew.x"
// LOWER-LABEL: llvm.func @mzero_2d_acc(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mzero.2d.acc"
// LOWER-LABEL: llvm.func @mzero_2d_m(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mzero.2d.m"
