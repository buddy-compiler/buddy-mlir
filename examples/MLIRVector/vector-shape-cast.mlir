func.func @main() -> i32 {
  // vector.shape_cast cast vector<... x a1 x a2 x ... x an x ... x T> to
  // vector<... x b x ... x T>, where a1 * a2 * ... * an == b

  %0 = arith.constant dense<[
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 
    19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.
  ]> : vector<36xf32>

  // This cast should not move any element.
  %1 = vector.shape_cast %0 : vector<36xf32> to vector<3x2x2x3xf32>
  vector.print %1 : vector<3x2x2x3xf32>

  %2 = vector.shape_cast %1 : vector<3x2x2x3xf32> to vector<3x4x3xf32>
  vector.print %2 : vector<3x4x3xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
