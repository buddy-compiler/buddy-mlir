// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

memref.global "private" @gv : memref<2x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                       [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]>

func.func @main() -> i64 {
  %0 = arith.constant 0 : i64
  %stride16 = arith.constant 16 : i64
  %stride8 = arith.constant 8 : i64
  %spadAddr = arith.constant 0 : i64
  %arrayA = memref.get_global @gv : memref<2x16xi8>
  %arrayB = memref.alloc() : memref<3x16xi8>
  %arrayC = memref.alloc() : memref<2x8xi8>
  gemmini.print %arrayB : memref<3x16xi8>
  gemmini.print %arrayC : memref<2x8xi8>
  // CHECK: "gemmini.intr.config_st"
  // The mvout op's stride is 16.
  gemmini.config_st %stride16 : i64
  // CHECK: "gemmini.intr.config_ld"
  // The mvin op's stride is 16 
  gemmini.config_ld %stride16 : i64 
  // CHECK: "gemmini.intr.mvin"
  gemmini.mvin %arrayA %spadAddr : memref<2x16xi8> i64
  // CHECK: "gemmini.intr.mvout"
  gemmini.mvout %arrayB %spadAddr : memref<3x16xi8> i64
  // CHECK: "gemmini.intr.config_st"
  // The mvout op's stride is 8 
  gemmini.config_st %stride8 : i64
  // CHECK: "gemmini.intr.mvout"
  gemmini.mvout %arrayC %spadAddr : memref<2x8xi8> i64 
  gemmini.print %arrayB : memref<3x16xi8>
  gemmini.print %arrayC : memref<2x8xi8>
  return %0 : i64
}
