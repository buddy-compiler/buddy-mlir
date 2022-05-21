func.func @main() {
  %v0 = arith.constant dense<0> : vector<5x1x4x3xi32>
  %v1 = vector.shape_cast %v0 : vector<5x1x4x3xi32> to vector<20x3xi32>
  
  %v2 = arith.constant dense<0> : vector<10x12x8xi32>
  %v3 = vector.shape_cast %v2 : vector<10x12x8xi32> to vector<5x2x3x4x8xi32>
  func.return 

}