func.func @main() {
  %pred  = arith.constant 0 : i1
  %x, %y = scf.if %pred -> (i32, i32) {
    %x_true = arith.constant 1 : i32
    %y_true = arith.constant 1 : i32 
    scf.yield %x_true, %y_true : i32, i32
  } else {
    %x_false = arith.constant -1 : i32
    %y_false = arith.constant -1 : i32 
    scf.yield %x_false, %y_false : i32, i32
  }
  vector.print %x : i32 
  vector.print %y : i32
  func.return

}
