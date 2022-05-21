func.func @main() {
  %v0 = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xi32>
  %i0 = vector.reduction <add>, %v0 : vector<6xi32> into i32
  vector.print %i0 : i32
  %i1 = vector.reduction <xor>, %v0 : vector<6xi32> into i32
  vector.print %i1 : i32
  %i2 = vector.reduction <mul>, %v0 : vector<6xi32> into i32
  vector.print %i2 : i32
  %i3 = vector.reduction <or>, %v0 : vector<6xi32> into i32
  vector.print %i3 : i32
  func.return 

}