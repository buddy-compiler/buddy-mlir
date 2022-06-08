func.func @main() {

  %mask1 = vector.constant_mask [5] : vector<10xi1>
  vector.print %mask1 : vector<10xi1>

  %mask2 = vector.constant_mask [2, 2, 2] : vector<3x3x3xi1>
  vector.print %mask2 : vector<3x3x3xi1>

  return
}
