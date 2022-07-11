memref.global "private" @gv : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                      10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<20xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %vec1 = vector.load %mem[%c0] : memref<20xf32>, vector<8xf32>
  %vec2 = vector.load %mem[%c10] : memref<20xf32>, vector<8xf32>
  %mask = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl = arith.constant 8 : i32
  %res = "llvm.intr.vp.fadd" (%vec1, %vec2, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  vector.print %res : vector<8xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
