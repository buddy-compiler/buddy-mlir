func.func @main() {
  %v0 = arith.constant dense<[1,2,3,4]> : vector<4xi32>
  vector.print %v0 : vector<4xi32>

  %v1 = vector.bitcast %v0 : vector<4xi32> to vector<2xi64>
  vector.print %v1 : vector<2xi64>

  %v2 = vector.bitcast %v0 : vector<4xi32> to vector<4xf32>
  vector.print %v2 : vector<4xf32>

  %v3 = vector.bitcast %v2 : vector<4xf32> to vector<4xi32>
  vector.print %v3 : vector<4xi32>
  func.return
   
}