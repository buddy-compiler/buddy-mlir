// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Define global input matrices, all initialized to 1
"memref.global"() {sym_name = "input_a", type = memref<128x256xi8>, initial_value = dense<1> : tensor<128x256xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "input_b", type = memref<256x512xi8>, initial_value = dense<1> : tensor<256x512xi8>, visibility = "private"} : () -> ()

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  // Get input matrices
  %a = memref.get_global @input_a : memref<128x256xi8>
  %b = memref.get_global @input_b : memref<256x512xi8>
  
  // Allocate output matrix
  %c = memref.alloc() : memref<128x512xi8>
  
  // Set warp number to 16
  %warp_num = arith.constant 16 : i64
  
  // Use Buckyball's bb_tile_matmul operation to perform tile matrix multiplication
  // CHECK: bb_tile_matmul
  "buckyball.bb_tile_matmul"(%a, %b, %c) {warpNum = 16 : i64} : 
    (memref<128x256xi8>, memref<256x512xi8>, memref<128x512xi8>) -> ()
  
  return %0 : i8
} 