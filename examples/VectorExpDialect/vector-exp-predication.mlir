module {
  func.func @main() {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %v1 = arith.constant dense<[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]> : vector<8xf32>
    %v2 = arith.constant dense<[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]> : vector<8xf32>
    %mask = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
    %vl = arith.constant 4 : i32
    // Single op for addf.
    %result = vector_exp.predication %mask, %vl : vector<8xi1>, i32 {
      %add = arith.addf %v1, %v2 : vector<8xf32>
      vector.yield %add : vector<8xf32>
    } : vector<8xf32>
    vector.print %result : vector<8xf32>
    // Single op for mulf.
    %result_mul = vector_exp.predication %mask, %vl : vector<8xi1>, i32 {
      %mul = arith.mulf %v1, %v2 : vector<8xf32>
      vector.yield %mul : vector<8xf32>
    } : vector<8xf32>
    vector.print %result_mul : vector<8xf32>
    return
  } 
}
