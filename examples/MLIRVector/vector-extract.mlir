func.func @main() {
  %v0 = arith.constant dense<[[[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17]],
                             [[18,19,20,21,22,23],[24,25,26,27,28,29],[30,31,32,33,34,35]]]> : vector<2x3x6xi32>
  %v1 = vector.extract %v0[1, 2, 5] : vector<2x3x6xi32>
  vector.print %v1 : i32 
  %v2 = vector.extract %v0[1, 2] : vector<2x3x6xi32>
  vector.print %v2 : vector<6xi32>
  %v3 = vector.extract %v0[1] : vector<2x3x6xi32>
  vector.print %v3 : vector<3x6xi32> 
  func.return
   
}