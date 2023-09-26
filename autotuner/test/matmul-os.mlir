// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

memref.global "private" @gv1 : memref<4x4xi8> = dense<[[1, 2, 3, 4],
                                                       [5, 6, 7, 8],
                                                       [9, 10, 11, 12],
                                                       [13, 14, 15, 16]]>
memref.global "private" @gv2 : memref<4x4xi8> = dense<[[1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1]]>

func.func @main() -> i64 {
  %in = memref.get_global @gv1 : memref<4x4xi8>
  %identity = memref.get_global @gv2 : memref<4x4xi8>
  %out = memref.alloc() : memref<4x4xi8>
  gemmini.print %out : memref<4x4xi8> 
  %inSpAddr = arith.constant 0 : i64
  %outSpAddr = arith.constant 4 : i64 
  %identitySpAddr = arith.constant 8 : i64
  %cst4 = arith.constant 4 : i64
  %cst0 = arith.constant 0 : i64
  // CHECK: "gemmini.intr.config_st"
  gemmini.config_st %cst4 : i64
  // CHECK: "gemmini.intr.config_ld"
  gemmini.config_ld %cst4 : i64
  // CHECK: "gemmini.intr.mvin"
  gemmini.mvin %in %inSpAddr : memref<4x4xi8> i64
  // CHECK: "gemmini.intr.config_ld"
  gemmini.config_ld %cst4 : i64
  // CHECK: "gemmini.intr.mvin"
  gemmini.mvin %identity %identitySpAddr : memref<4x4xi8> i64
  // CHECK: "gemmini.intr.config_ex"
  gemmini.config_ex {dataflow = 0 } 
  // CHECK: "gemmini.intr.preload"
  gemmini.preload_zeros %outSpAddr %cst4 %cst4 : i64 i64 i64
  // CHECK: "gemmini.intr.compute_preloaded"
  gemmini.compute_preloaded %inSpAddr %identitySpAddr %cst4 %cst4 %cst4 %cst4 : i64 i64 i64 i64 i64 i64
  // CHECK: "gemmini.intr.mvout"
  gemmini.mvout %out %outSpAddr : memref<4x4xi8> i64
  gemmini.print %out : memref<4x4xi8> 
  return %cst0 : i64
}
