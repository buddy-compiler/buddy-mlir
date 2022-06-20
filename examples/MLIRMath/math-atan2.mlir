func.func @main() {
  %c1 = arith.constant 1.0 : f32
  %res = math.atan2 %c1, %c1 : f32
  vector.print %res : f32
  func.return
}
