memref.global "private" @gv : memref<6x6xi32> = dense<[[10, 11, 12, 34, 35, 89],
                                                       [34, 78, 90, 62, 12, 13],
                                                       [56, 78, 90, 91, 23, 45],
                                                       [123, 45, 67, 89, 25, 123],
                                                       [12, 34, 43, 32, 22, 23],
                                                       [90, 91, 67, 89, 92, 57]]>

func.func @main() -> i32 {
  %cons0 = arith.constant 0 : index
  %cons2 = arith.constant 2 : index 

  %cons1 = arith.constant 1 : index
  %base = memref.get_global @gv : memref<6x6xi32>

  %0 = arith.constant dense<[12, 34, 56, 78, 89, 90, 91, 101]> : vector<8xi32>
  %mask0 = arith.constant dense<[1, 0, 0, 1, 0, 1, 0, 0]> : vector<8xi1>

  %mask1 = arith.constant dense<[1, 1, 1, 1, 0, 0, 1, 0]> : vector<8xi1>
  %1 = arith.constant dense<[12, 23, 45, 67, 89, 90, 91, 98]> : vector<8xi32>

  %res0 = vector.expandload %base[%cons0, %cons1], %mask0, %0 : 
          memref<6x6xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  vector.print %res0 : vector<8xi32>

  %res1 = vector.expandload %base[%cons0, %cons0], %mask1, %1 : 
        memref<6x6xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  vector.print %res1 : vector<8xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
