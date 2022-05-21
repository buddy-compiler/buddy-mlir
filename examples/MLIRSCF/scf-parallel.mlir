func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c10 = arith.constant 10 : index
  %A = arith.constant 11 : index 
  %B = arith.constant 12 : index 
  %0:2=scf.parallel (%i0, %i1) = (%c1, %c3) to (%c2, %c6) step (%c1, %c3) init(%A, %B) -> (index, index) {
    scf.reduce(%i0) : index {
    ^bb0(%lhs: index, %rhs: index):
      vector.print %lhs : index 
      vector.print %rhs : index 
      %1 = arith.addi %lhs, %rhs : index
      vector.print %1 : index 
      scf.reduce.return %1 : index
    }
    scf.reduce(%i1) : index {
    ^bb0(%lhs: index, %rhs: index):
      vector.print %lhs : index 
      vector.print %rhs : index 
      %2 = arith.muli %lhs, %rhs : index
      vector.print %2 : index 
      scf.reduce.return %2 : index
    }
  }
  vector.print %0#0 : index 
  vector.print %0#1 : index 
  func.return

}
