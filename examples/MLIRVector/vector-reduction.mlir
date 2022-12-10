func.func @main() -> i32 {
  // vector.reduction reduces an 1-D vector “horizontally” into a scalar using 
  // the given operation. For n-D vector reduction, please refer to 
  // vector.multi_reduction
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

  // vector.reduction also allow an optional fused accumulator.
  // doing (%acc kind (vector.reduction <kind>, %vec))
  %acc = arith.constant 1 : i32

  // this will do %acc * (vector.reduction <mul>, %0)
  %mul_1 = vector.reduction <mul>, %0, %acc : vector<6xi32> into i32
  vector.print %mul_1 : i32

  // this will do %acc + (vector.reduction <add>, %0)
  %sum_1 = vector.reduction <add>, %0, %acc : vector<6xi32> into i32
  vector.print %sum_1 : i32

  // this will do min(%acc, (vector.reduction <min>, %0))
  %min_1 = vector.reduction <minsi>, %0, %acc : vector<6xi32> into i32
  vector.print %min_1 : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
