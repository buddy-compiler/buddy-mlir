memref.global "private" @gv : memref<9xi32> = dense<[9, 10, 34, 45, 78, 89, 90, 12, 34]> 

func.func @main() -> i32 {
  %base = memref.get_global @gv : memref<9xi32>
  %mask = arith.constant dense<[1, 1, 1, 0, 1, 0, 0, 1, 1]> : vector<9xi1>

  %pass_thru = arith.constant dense<[12, 34, 45, 67, 13, 78, 90, 75, 45]> : vector<9xi32>
  %c1 = arith.constant 1 : index

  %print_out = vector.maskedload %base[%c1], %mask, %pass_thru
             : memref<9xi32>, vector<9xi1>, vector<9xi32> into vector<9xi32>
  vector.print %print_out : vector<9xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
