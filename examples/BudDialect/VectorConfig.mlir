module {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  bud.vector_config %c1, %c2 : f32, f32 {
    %add = arith.addf %c1, %c2 : f32
    vector.print %add : f32
    vector.yield %add : f32
  }
}
