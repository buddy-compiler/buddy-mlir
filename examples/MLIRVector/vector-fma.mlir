func.func @main() -> i32 {
  // fma means "fused multiply-add operation", 
  // first do multiply, and then add them up

  // Normal case
  %v0_lhs = arith.constant dense<[0., 1., 2., 3.]> : vector<4xf32>
  %v0_rhs = arith.constant dense<[20., 21., 22., 23.]> : vector<4xf32>
  %v0_acc = arith.constant dense<[30., 31., 32., 33.]> : vector<4xf32>

  // result = (lhs * rhs) + acc
  %r0 = vector.fma %v0_lhs, %v0_rhs, %v0_acc : vector<4xf32>
  vector.print %r0 : vector<4xf32>


  // OK for n-D vector
  %v1_lhs = arith.constant dense<[[0., 1.], [2., 3.]]> : vector<2x2xf32>
  %v1_rhs = arith.constant dense<[[20., 21.], [22., 23.]]> : vector<2x2xf32>
  %v1_acc = arith.constant dense<[[30., 31.], [32., 33.]]> : vector<2x2xf32>
  
  %r1 = vector.fma %v1_lhs, %v1_rhs, %v1_acc : vector<2x2xf32>
  vector.print %r1 : vector<2x2xf32>


  // NOT for integers! although "f" means "fused" instead of "float", 
  // it could only used on float type.
  %v2_lhs = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xi32>
  %v2_rhs = arith.constant dense<[[20, 21], [22, 23]]> : vector<2x2xi32>
  %v2_acc = arith.constant dense<[[30, 31], [32, 33]]> : vector<2x2xi32>
  
  // This will NOT work.
  // %r2 = vector.fma %v2_lhs, %v2_rhs, %v2_acc : vector<2x2xi32>
  // vector.print %r2 : vector<2x2xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
