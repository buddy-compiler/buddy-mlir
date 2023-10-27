// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

memref.global "private" @g1 : memref<5x5xi8> = dense<[[1, 0, 0, 1, 0], [1, -1, 1, 0, 0], [-1, 0, 1, -1, 1], [1, 0, 0, 1, 0], [-1, 0, 0, -1, 0]]>
memref.global "private" @g2 : memref<5x5xi8> = dense<[[1, -1, 0, 0, 1], [1, 0, -1, 0, -1], [-1, -1, 0, -1, 1], [-1, 0, 0, 1, 0], [1, 0, 0, -1, 0]]>


func.func @main() -> i8 {
  %i0 = arith.constant 0 : i8
  %i1I8 = arith.constant 1 : i8
  %minus1 = arith.constant -2 : i8
  %i2I8 = arith.constant 2 : i8
  %i2I32 = arith.constant 2 : i32
  %dI32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index 
  %c1 = arith.constant 1 : index
  %aArray = memref.get_global @g1 : memref<5x5xi8>
  %bArray = memref.get_global @g2 : memref<5x5xi8>
  %cArray = memref.alloc()  : memref<5x5xi8>
  %dArray = memref.alloc()  : memref<5x5xi32>
  %dim_I = memref.dim %aArray, %c0 : memref<5x5xi8>
  %dim_J = memref.dim %bArray, %c1 : memref<5x5xi8>
  %dim_K = memref.dim %aArray, %c1 : memref<5x5xi8>

  scf.for %i3 = %c0 to %dim_I step %c1 {
    scf.for %j3 = %c0 to %dim_J step %c1 {
      memref.store %dI32, %dArray[%i3, %j3] : memref<5x5xi32>
    }
  }

  gemmini.tile_matmul %aArray %bArray %cArray %dArray {dataflow=1}: memref<5x5xi8> memref<5x5xi8> memref<5x5xi8> memref<5x5xi32>
  gemmini.print %cArray : memref<5x5xi8>
  
  // CHECK: "gemmini.intr.config_ex"
  // CHECK: "gemmini.intr.config_st"
  // CHECK: "gemmini.intr.config_ld"
  // CHECK: "gemmini.intr.loop_ws_config_bounds"
  // CHECK: "gemmini.intr.loop_ws_config_addrs_ab"
  // CHECK: "gemmini.intr.loop_ws_config_addrs_dc"
  // CHECK: "gemmini.intr.loop_ws_config_strides_ab"
  // CHECK: "gemmini.intr.loop_ws_config_strides_dc"
  // CHECK: "gemmini.intr.loop_ws"
  // CHECk: "gemmini.intr.flush"
  gemmini.tile_matmul %aArray %bArray %cArray %dArray {dataflow=1, act=1}: memref<5x5xi8> memref<5x5xi8> memref<5x5xi8> memref<5x5xi32>
  gemmini.print %cArray : memref<5x5xi8>
  return %i0 : i8
}
