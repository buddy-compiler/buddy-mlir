func.func @main() -> i32 {
  %c2 = arith.constant 2 : i32
  %v0 = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %v1 = arith.constant dense<[4, 5, 6]> : vector<3xi32>
  %v2 = arith.constant dense<[1, 2, 3, 4, 5, 6]> : vector<6xi32>

  // Normal usage:
  //    │ 4 │ 5 │ 6 │
  // ───┼───┼───┼───┤
  //  1 │ 4 │ 5 │ 6 │
  // ───┼───┼───┼───┤
  //  2 │ 8 │10 │12 │
  // ───┼───┼───┼───┤
  //  3 │12 │15 │18 │
  // ───┴───┴───┴───┘
  %w0 = vector.outerproduct %v0, %v1 : vector<3xi32>, vector<3xi32>
  vector.print %w0 : vector<3x3xi32>


  // Normal usage 2:
  //    │ 1 │ 2 │ 3 │
  // ───┼───┼───┼───┤
  //  4 │ 4 │ 8 │12 │
  // ───┼───┼───┼───┤
  //  5 │ 5 │10 │15 │
  // ───┼───┼───┼───┤
  //  6 │ 6 │12 │18 │
  // ───┴───┴───┴───┘
  %w1 = vector.outerproduct %v1, %v0 : vector<3xi32>, vector<3xi32>
  vector.print %w1 : vector<3x3xi32>


  // outerproduct with different dimension:
  %w2 = vector.outerproduct %v0, %v2 : vector<3xi32>, vector<6xi32>
  vector.print %w2 : vector<3x6xi32>

  
  // outerproduct with scalar:
  %w3 = vector.outerproduct %v0, %c2 : vector<3xi32>, i32
  vector.print %w3 : vector<3xi32>

  // scalar can only be RHS, this one above is not allowed
  // %w3 = vector.outerproduct %c2, %v0 : i32, vector<3xi32>


  // outerproduct only support creating 2-D vector with two 1-D vector
  // For "outer production" (or, tensor production) between vectors with higher 
  // rank, you may refer to vector.contract. 


  // vector.outerproduct can accept a vector as accumulation:
  //      result = acc + (LHS outerproduct RHS)
  // On platform that support fma (fused multiply-add),
  // it will be lowering into fma.
  %acc0 = arith.constant dense<1> : vector<3x3xi32>
  %w4 = vector.outerproduct %v0, %v1, %acc0 : vector<3xi32>, vector<3xi32>
  vector.print %w4 : vector<3x3xi32>


  // use attribute `kind` can specify the combination method for outer 
  // production and accumulation, like this:
  //      result = acc kind (LHS outerproduct RHS)
  %acc1 = arith.constant dense<2> : vector<3x3xi32>
  // this one use multiply as combiner, so the result should be 2 * %w0, or:
  //      result = acc * (LHS outerproduct RHS)
  %w5 = vector.outerproduct %v0, %v1, %acc1 { kind = #vector.kind<mul> }
    : vector<3xi32>, vector<3xi32>
  vector.print %w5 : vector<3x3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
