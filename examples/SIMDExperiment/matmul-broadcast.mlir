#map1 = affine_map<(d0) -> (d0 ceildiv 32)>
module{
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @matmul(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c0_f32_vec = vector.splat %c0_f32 : vector<32xf32>

    %a_row = memref.dim %a, %c0 : memref<?x?xf32>
    %a_col = memref.dim %a, %c1 : memref<?x?xf32>
    %b_row = memref.dim %b, %c0 : memref<?x?xf32>
    %b_col = memref.dim %b, %c1 : memref<?x?xf32>

    affine.for %b_row_idx = 0 to %b_row {
      affine.for %a_row_idx = 0 to %a_row {
        affine.for %b_col_idx = 0 to #map1(%b_col) {
          %a_ele = memref.load %a[%a_row_idx, %b_row_idx] : memref<?x?xf32>
          %a_vec = vector.broadcast %a_ele : f32 to vector<32xf32>
          // Check tail.
          %b_col_cur = arith.muli %b_col_idx, %c32 : index
          %tail_len = arith.subi %b_col, %b_col_cur : index
          %tail_flag = arith.cmpi sge, %tail_len, %c32 : index
          scf.if %tail_flag {
            %b_vec = affine.vector_load %b[%b_row_idx, %b_col_idx * 32] : memref<?x?xf32>, vector<32xf32>
            %c_vec = affine.vector_load %c[%a_row_idx, %b_col_idx * 32] : memref<?x?xf32>, vector<32xf32>
            %result_vec = vector.fma %a_vec, %b_vec, %c_vec : vector<32xf32>
            affine.vector_store %result_vec, %c[%a_row_idx, %b_col_idx * 32] : memref<?x?xf32>, vector<32xf32>
          } else {
            %mask_vec = vector.create_mask %tail_len : vector<32xi1>
            %b_col_idx_tail = arith.muli %b_col_idx, %c32 : index
            %b_vec_tail = vector.maskedload %b[%b_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
            %c_vec_tail = vector.maskedload %c[%a_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
            %result_vec_tail = vector.fma %a_vec, %b_vec_tail, %c_vec_tail : vector<32xf32>
            vector.maskedstore %c[%a_row_idx, %b_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?xf32>, vector<32xi1>, vector<32xf32>
          }
        }
      }
    }
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 64 : index
    %cN = arith.constant 64 : index
    %cK = arith.constant 64 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<?x?xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%B:memref<?x?xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%C:memref<?x?xf32>)

    call @matmul(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return 
  }
}
