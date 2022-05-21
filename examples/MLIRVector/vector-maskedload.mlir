memref.global "private" @gv : memref<5xi32> = dense<[0, 1, 2, 3, 4]> 

func.func @main() {
  %base = memref.get_global @gv : memref<5xi32>
  %mask = arith.constant dense<[0, 1, 0, 1, 0]> : vector<5xi1>
  %pass_thru = arith.constant dense<[5, 6, 7, 8, 9]> : vector<5xi32>
  %c0 = arith.constant 1 : index 
  %print_out = vector.maskedload %base[%c0], %mask, %pass_thru
             : memref<5xi32>, vector<5xi1>, vector<5xi32> into vector<5xi32>
  vector.print %print_out : vector<5xi32>      
  func.return
  
}