func.func @main() -> i32 {
  // vector.print can print out a vector or scalars with types that can be 
  // elements of a vector. It is widely used in examples or debugging.

  // print a vector
  %v0 = arith.constant dense<[32, 43]> : vector<2xi8>
  vector.print %v0 : vector<2xi8>

  // print a n-D vector
  %v1 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
  vector.print %v1 : vector<2x2xf32>

  // print a scalar
  %e2 = arith.constant 3.0 : f32
  vector.print %e2 : f32

  %ret = arith.constant 0 : i32
  return %ret : i32
}