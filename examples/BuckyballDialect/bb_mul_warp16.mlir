// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  %a_sp = arith.constant 100 : i64  
  %b_sp = arith.constant 200 : i64  
  %c_sp = arith.constant 300 : i64  
  %len = arith.constant 64 : i64    
  
  // Use Buckyball's bb_mul_warp16 operation to perform matrix multiplication (16 warp parallel)
  // CHECK: bb_mul_warp16
  "buckyball.bb_mul_warp16"(%a_sp, %b_sp, %c_sp, %len) : (i64, i64, i64, i64) -> ()
  
  return %0 : i8
}

// Dynamic parameter version
func.func @dynamic_test(%addr1: i64, %addr2: i64, %addr3: i64, %length: i64) -> i8 {
  %0 = arith.constant 0 : i8
  
  // Use dynamic parameter bb_mul_warp16 operation
  // CHECK: bb_mul_warp16
  "buckyball.bb_mul_warp16"(%addr1, %addr2, %addr3, %length) : (i64, i64, i64, i64) -> ()
  
  return %0 : i8
}
