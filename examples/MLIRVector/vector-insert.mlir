func.func @main() -> i32 {
  %0 = arith.constant dense<[[[1., 9., 8.], [12., 13., 14.], [23., 45., 78.]],
                              [[12., 67., 89.], [16., 17., 10.], [15., 78., 65.]],
                              [[134., 56., 27.], [90., 87., 65.], [12., 45., 78.]]]> : vector<3x3x3xf32>
  vector.print %0 : vector<3x3x3xf32>

  %constant = arith.constant 2. : f32
  %1 = vector.insert %constant, %0[0, 0, 0] : f32 into vector<3x3x3xf32> // Insert 2 at position (0,0,0), this will replace 1 
  vector.print %1 : vector<3x3x3xf32>

  %2 = arith.constant dense<[12., 34., 70.]> : vector<3xf32>
  vector.print %2 : vector<3xf32>

  %3 = vector.insert %2, %0[2, 1] : vector<3xf32> into vector<3x3x3xf32>  // Insert a 1d vector at (2,1), replaces [90.,87.,65.]
  vector.print %3 : vector<3x3x3xf32>
    
  %4 = arith.constant dense<[[12, 11],
                             [23, 24]]> : vector<2x2xi32>
  vector.print %4 : vector<2x2xi32>

  %vec = arith.constant dense<[100, 101]> : vector<2xi32>
  %5 = vector.insert %vec, %4[1] : vector<2xi32> into vector<2x2xi32> // Will replace [23,24], with [100,101]

  vector.print %5 : vector<2x2xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
