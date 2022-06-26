func.func @main() -> i32 {
  %0 = arith.constant dense<[12, 13, 14, 15, 16, 90]> : vector<6xi32>
  vector.print %0 : vector<6xi32>

  %sum = vector.reduction <add>, %0 : vector<6xi32> into i32
  vector.print %sum : i32

  %mul = vector.reduction <mul>, %0 : vector<6xi32> into i32
  vector.print %mul : i32

  %xor = vector.reduction <xor>, %0 : vector<6xi32> into i32
  vector.print %xor : i32

  %and = vector.reduction <and>, %0 : vector<6xi32> into i32
  vector.print %and : i32

  %or = vector.reduction <or>, %0 : vector<6xi32> into i32
  vector.print %or : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
