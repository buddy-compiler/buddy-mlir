// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

func.func @main() -> i8 {
  %i0 = arith.constant 0 : i8
  %i1I8 = arith.constant 1 : i8
  %i2I8 = arith.constant 2 : i8
  %i2I32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index 
  %c1 = arith.constant 1 : index 
  %aArray = memref.alloc() {alignment = 16} : memref<64x64xi8>
  %bArray = memref.alloc() {alignment = 16}: memref<64x64xi8>
  %cArray = memref.alloc() {alignment = 16}: memref<64x64xi8>
  %dArray = memref.alloc() {alignment = 64} : memref<64x64xi32>
  %dim = memref.dim %aArray, %c0 : memref<64x64xi8> 
  scf.for %i = %c0 to %dim step %c1 {
    scf.for %j = %c0 to %dim step %c1 {
      memref.store %i1I8, %aArray[%i, %j] : memref<64x64xi8> 
      memref.store %i1I8, %bArray[%i, %j] : memref<64x64xi8> 
      memref.store %i2I32, %dArray[%i, %j] : memref<64x64xi32>
    }
  }
  
  gemmini.print %aArray : memref<64x64xi8>
  gemmini.print %bArray : memref<64x64xi8>
  gemmini.print %dArray : memref<64x64xi32>
  // CHECK: "gemmini.intr.config_ld"
  // CHECK: "gemmini.intr.mvin"
  // CHECK: "gemmini.intr.preload"
  // CHECK: "gemmini.intr.compute_preloaded"
  // CHECK: "gemmini.intr.compute_accumulated"
  // CHECK: "gemmini.intr.mvout"
  gemmini.tile_matmul  %aArray %bArray %cArray %dArray {dataflow=0} : memref<64x64xi8> memref<64x64xi8> memref<64x64xi8> memref <64x64xi32>
  gemmini.print %cArray : memref<64x64xi8>
  return %i0 : i8
}
