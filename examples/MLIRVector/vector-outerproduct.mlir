func.func @main() -> i32 {
  %0 = arith.constant dense<[12, 11, 78, 90, 23, 56]> : vector<6xi32>
  vector.print %0 : vector<6xi32>

  %1 = arith.constant dense<[13, 56, 78, 89, 95, 23]> : vector<6xi32>
  vector.print %1 : vector<6xi32>

  %2 = vector.outerproduct %0, %1 : vector<6xi32>, vector<6xi32> // will give vector of shape 6x6
  vector.print %2 : vector<6x6xi32>

  %3 = arith.constant dense<[12, 67, 49]> : vector<3xi32>
  vector.print %3 : vector<3xi32>

  %4 = vector.outerproduct %0, %3 : vector<6xi32>, vector<3xi32> // will give vector of shape 6x3
  vector.print %4 : vector<6x3xi32>

  %cons = arith.constant 4 : i32
  %5 = vector.outerproduct %0, %cons : vector<6xi32>, i32  // will give vector of same shape, formula is [a,c]*d = [a*d,c*d]
  vector.print %5 : vector<6xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
