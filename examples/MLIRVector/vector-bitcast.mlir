func.func @main() -> i32 {
  %v0 = arith.constant dense<[10, 20, 56, 90, 12, 90]> : vector<6xi32>
  vector.print %v0 : vector<6xi32>

  %v1 = vector.bitcast %v0 : vector<6xi32> to vector<3xi64>
  vector.print %v1 : vector<3xi64>

  %v2 = vector.bitcast %v0 : vector<6xi32> to vector<6xf32>
  vector.print %v2 : vector<6xf32>

  %v3 = vector.bitcast %v2 : vector<6xf32> to vector<6xi32>
  vector.print %v3 : vector<6xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
