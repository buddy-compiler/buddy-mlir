// RUN: buddy-opt %s \
// RUN:     -lower-buckyball

// Define global input matrices, all initialized to 1
"memref.global"() {sym_name = "input_a", type = memref<64x64xi8>, initial_value = dense<1> : tensor<64x64xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "input_b", type = memref<64x64xi8>, initial_value = dense<1> : tensor<64x64xi8>, visibility = "private"} : () -> ()

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  // Get input matrices
  %a = memref.get_global @input_a : memref<64x64xi8>
  %b = memref.get_global @input_b : memref<64x64xi8>
  
  // Allocate output matrix
  %c = memref.alloc() : memref<64x64xi8>
  
  // Get scratchpad addresses
  %a_sp = arith.constant 100 : i64  // Matrix A scratchpad address
  %b_sp = arith.constant 200 : i64  // Matrix B scratchpad address
  %c_sp = arith.constant 300 : i64  // Matrix C scratchpad address
  
  // Define meta tile dimensions
  %meta_m_num = arith.constant 4 : i64  // Number of meta tiles in M dimension
  %meta_n_num = arith.constant 4 : i64  // Number of meta tiles in N dimension
  
  // 添加metaMLen和metaNLen参数
  %meta_m_len = arith.constant 16 : i64  // Meta tile size in M dimension
  %meta_n_len = arith.constant 16 : i64  // Meta tile size in N dimension
  
  // Load matrix B into scratchpad first
  "buckyball.mvin"(%b, %b_sp) {} : (memref<64x64xi8>, i64) -> ()
  
  // Create meta tile view of matrix A
  %a_meta_tile = memref.subview %a[0, 0][32, 32][1, 1] : memref<64x64xi8> to memref<32x32xi8, strided<[64, 1]>>
  
  // Load meta tile A into scratchpad
  "buckyball.mvin"(%a_meta_tile, %a_sp) {} : (memref<32x32xi8, strided<[64, 1]>>, i64) -> ()
  
  // Use Buckyball's bb_metatile_matmul operation to perform meta tile matrix multiplication
  // CHECK: bb_metatile_matmul
  "buckyball.bb_metatile_matmul"(%a_meta_tile, %a, %a_sp, %b_sp, %c_sp, %meta_m_num, %meta_n_num, %meta_m_len, %meta_n_len) {} : 
    (memref<32x32xi8, strided<[64, 1]>>, memref<64x64xi8>, i64, i64, i64, i64, i64, i64, i64) -> ()
  
  // Store result back to memory
  "buckyball.mvout"(%c, %c_sp) {} : (memref<64x64xi8>, i64) -> ()
  
  return %0 : i8
}

// 动态参数版本
func.func @dynamic_test(
    %a_meta_tile: memref<?x?xi8, strided<[?, 1]>>, 
    %a: memref<?x?xi8>, 
    %a_sp: i64, 
    %b_sp: i64, 
    %c_sp: i64, 
    %meta_m_num: i64, 
    %meta_n_num: i64,
    %meta_m_len: i64,
    %meta_n_len: i64
) -> i8 {
  %0 = arith.constant 0 : i8
  
  // Use dynamic parameter bb_metatile_matmul operation
  "buckyball.mvin"(%a_meta_tile, %a_sp) {} : (memref<?x?xi8, strided<[?, 1]>>, i64) -> ()
  
  "buckyball.bb_metatile_matmul"(%a_meta_tile, %a, %a_sp, %b_sp, %c_sp, %meta_m_num, %meta_n_num, %meta_m_len, %meta_n_len) {} : 
    (memref<?x?xi8, strided<[?, 1]>>, memref<?x?xi8>, i64, i64, i64, i64, i64, i64, i64) -> ()
  
  return %0 : i8
}

func.func @minimal_test() -> i8 {
  %zero = arith.constant 0 : i8
  
  %a = memref.alloc() : memref<128x128xi8>
  %a_meta = memref.subview %a[0, 0][16, 16][1, 1] : memref<128x128xi8> to memref<16x16xi8, strided<[128, 1]>>
  // %a_meta = memref.alloc() : memref<16x16xi8, strided<[32, 1]>>
  
  %a_sp = arith.constant 100 : i64
  %b_sp = arith.constant 200 : i64
  %c_sp = arith.constant 300 : i64
  %m_num = arith.constant 2 : i64
  %n_num = arith.constant 2 : i64 
  %m_len = arith.constant 16 : i64
  %n_len = arith.constant 16 : i64
  
  // Call operation
  // %a_meta: MemRefRankOf<[AnyType], [2]>:$aMetaTileArray
  // %a: MemRefRankOf<[AnyType], [2]>:$aMemArray
  // %a_sp: I64:$aSpAddrStart
  // %b_sp: I64:$bSpAddrStart
  // %c_sp: I64:$cSpAddrStart
  // %m_num: I64:$metaMNum
  // %n_num: I64:$metaNNum
  // %m_len: I64:$metaMLen
  // %n_len: I64:$metaNLen
  "buckyball.bb_metatile_matmul"(%a_meta, %a, %a_sp, %b_sp, %c_sp, %m_num, %n_num, %m_len, %n_len) : 
      (memref<16x16xi8, strided<[128, 1]>>, memref<128x128xi8>, i64, i64, i64, i64, i64, i64, i64) -> ()
  
  return %zero : i8
} 