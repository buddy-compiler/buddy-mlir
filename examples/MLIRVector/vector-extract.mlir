func.func @main() -> i32 {
  %0 = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  vector.print %0 : vector<4xi32>

  %value = vector.extract %0[2] : vector<4xi32>
  vector.print %value : i32

  %1 = arith.constant dense<[[[12, 13, 14], [14, 15, 16]],
                             [[14, 16, 19], [22, 89, 78]]]> : vector<2x2x3xi32>
  vector.print %1 : vector<2x2x3xi32>

  %value1 = vector.extract %1[0, 1] : vector<2x2x3xi32> // extracts [14,15,16]
  vector.print %value1 : vector<3xi32>

  %value2 = vector.extract %1[1, 1, 2] : vector<2x2x3xi32> // extracts 78
  vector.print %value2 : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
