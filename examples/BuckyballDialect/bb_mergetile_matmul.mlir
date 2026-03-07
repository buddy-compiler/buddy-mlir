// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Define global input matrices, all initialized to 1
"memref.global"() {sym_name = "input_a", type = memref<128x64xi8>, initial_value = dense<1> : tensor<128x64xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "input_b", type = memref<64x256xi8>, initial_value = dense<1> : tensor<64x256xi8>, visibility = "private"} : () -> ()

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  // Get input matrices
  %a = memref.get_global @input_a : memref<128x64xi8>
  %b = memref.get_global @input_b : memref<64x256xi8>
  
  // Allocate output matrix
  %c = memref.alloc() : memref<128x256xi8>
  
  // Define constants for dimension access
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  // Get dimensions dynamically
  %m = memref.dim %a, %c0 : memref<128x64xi8>
  %k1 = memref.dim %a, %c1 : memref<128x64xi8>
  %k2 = memref.dim %b, %c0 : memref<64x256xi8>
  %n = memref.dim %b, %c1 : memref<64x256xi8>
  
  // Get scratchpad addresses 
  %a_sp = arith.constant 1000 : i64  // Matrix A scratchpad address
  %b_sp = arith.constant 2000 : i64  // Matrix B scratchpad address
  %c_sp = arith.constant 3000 : i64  // Matrix C scratchpad address
  
  // Define merge tile dimensions
  %meta_m_num = arith.constant 8 : i64  // Number of meta tiles in M dimension
  %meta_n_num = arith.constant 16 : i64  // Number of meta tiles in N dimension
  %meta_k_num = arith.constant 4 : i64  // Number of meta tiles in K dimension
  
  // Define meta tile lengths
  %meta_m_len = arith.constant 16 : i64  // Meta tile size in M dimension
  %meta_n_len = arith.constant 16 : i64  // Meta tile size in N dimension
  %meta_k_len = arith.constant 16 : i64  // Meta tile size in K dimension
  
  // Convert meta_k_len to index type for subview
  %meta_k_len_idx = arith.index_cast %meta_k_len : i64 to index
  
  // Create merge tile views for matrices A and B with dynamic sizes
  // Use index constants for subview dimensions to match types
  %a_merge_tile = memref.subview %a[0, 0][%m, %meta_k_len_idx][1, 1] : memref<128x64xi8> to memref<?x?xi8>
  %b_merge_tile = memref.subview %b[0, 0][%meta_k_len_idx, %n][1, 1] : memref<64x256xi8> to memref<?x?xi8>
  
  // Use Buckyball's bb_mergetile_matmul operation to perform merge tile matrix multiplication
  // CHECK: bb_mergetile_matmul
  "buckyball.bb_mergetile_matmul"(%a_merge_tile, %a, %b_merge_tile, %b, 
                                 %a_sp, %b_sp, %c_sp,
                                 %meta_m_num, %meta_n_num, %meta_k_num, 
                                 %meta_m_len, %meta_n_len, %meta_k_len) {} : 
    (memref<?x?xi8>, memref<128x64xi8>, 
     memref<?x?xi8>, memref<64x256xi8>, 
     i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
  
  return %0 : i8
} 