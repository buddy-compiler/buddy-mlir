func.func @main() {
    
  //%con = arith.constant 1 : i1  // result will be 50.0
  %con = arith.constant 0 : i1  // result will be 0.0

  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index

  %initial = arith.constant 25.0 : f32
  %step = arith.constant 1 : index

  %t = arith.constant 5.0 : f32
  %final = scf.if %con -> f32{
    %res = scf.for %iv = %lb to %ub step %step
    iter_args(%resiter = %initial) -> f32 {

      %1 = arith.addf %resiter , %t : f32
      scf.yield %1 : f32
    }

    scf.yield %res : f32           
    } else {
        %res = scf.for %iv = %lb to %ub step %step
        iter_args(%resiter = %initial) -> f32 {
            
          %1 = arith.subf %resiter , %t : f32
          scf.yield %1 : f32
        }
    scf.yield %res : f32         
  }
  vector.print %final : f32
  return
}
