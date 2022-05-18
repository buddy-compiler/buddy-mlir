func.func @main() {
  %c0 = arith.constant 1 : index
  %max0 = affine.max affine_map<(d0) -> (0, d0)> (%c0) 
  vector.print %max0 : index
  %min = affine.min affine_map<(d0) -> (0, d0)> (%c0) 
  vector.print %min : index 
  func.return

}
