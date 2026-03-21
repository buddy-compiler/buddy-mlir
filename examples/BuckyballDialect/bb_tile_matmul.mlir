// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Bebop scratchpad: i8 banks depth * 16B <= BANK_SIZE; C bank is cols=4 so mvout line_bytes=64
// and depthC * 64 <= BANK_SIZE => depthC <= 256 (depthC = M*(N/16)).
"memref.global"() {sym_name = "input_a", type = memref<32x128xi8>, initial_value = dense<1> : tensor<32x128xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "input_b", type = memref<128x128xi8>, initial_value = dense<1> : tensor<128x128xi8>, visibility = "private"} : () -> ()

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  // Get input matrices
  %a = memref.get_global @input_a : memref<32x128xi8>
  %b = memref.get_global @input_b : memref<128x128xi8>

  // Allocate output matrix
  %c = memref.alloc() : memref<32x128xi8>

  // Use Buckyball bb_matmul for one full static matmul (tiling is handled by tile dialect + convert-tile-to-buckyball)
  // CHECK: bb_mul_warp16
  "buckyball.bb_matmul"(%a, %b, %c) :
    (memref<32x128xi8>, memref<128x128xi8>, memref<32x128xi8>) -> ()

  memref.dealloc %c : memref<32x128xi8>

  return %0 : i8
}
