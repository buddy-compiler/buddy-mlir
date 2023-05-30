memref.global "private" @gv0 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : i32
  %base0 = memref.get_global @gv0 : memref<8xi32>

  %v1 = vector.load %base0[%c0] : memref<8xi32>, vector<3xi32>
  vector.print %v1 : vector<3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
