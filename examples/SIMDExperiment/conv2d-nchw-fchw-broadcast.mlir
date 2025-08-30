#map1 = affine_map<(d0) -> (d0 ceildiv 32)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  // Allocate and fill the memref according to the given layout.
  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  // Convoluation implementation.
  func.func @conv_2d_nchw_fchw(%input: memref<?x?x?x?xf32>,
                               %kernel: memref<?x?x?x?xf32>,
                               %output: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c0_f32_vec = vector.splat %c0_f32 : vector<32xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    // Get the n size. (batch)
    %n = memref.dim %input, %c0 :  memref<?x?x?x?xf32>
    // Get the f size. (feature)
    %f = memref.dim %kernel, %c0 :  memref<?x?x?x?xf32>
    // Get the c size. (channel)
    %c = memref.dim %kernel, %c1 :  memref<?x?x?x?xf32>
    // Get the 2D output size. (row and column)
    %output_row = memref.dim %output, %c2 :  memref<?x?x?x?xf32>
    %output_col = memref.dim %output, %c3 :  memref<?x?x?x?xf32>
    // Get the 2D kernel size. (row and column)
    %kernel_row = memref.dim %kernel, %c2 :  memref<?x?x?x?xf32>
    %kernel_col = memref.dim %kernel, %c3 :  memref<?x?x?x?xf32>

    affine.for %n_idx = %c0 to %n {
      affine.for %f_idx = %c0 to %f {
        affine.for %c_idx = %c0 to %c {
          affine.for %output_row_idx = %c0 to %output_row {
            affine.for %kernel_row_idx = %c0 to %kernel_row {
              affine.for %kernel_col_idx = %c0 to %kernel_col {
                affine.for %output_col_idx = %c0 to #map1(%output_col) {
                  // Check sparsity.
                  %kernel_ele = memref.load %kernel[%f_idx, %c_idx, %kernel_row_idx, %kernel_col_idx] : memref<?x?x?x?xf32>
                  %sparsity_flag = arith.cmpf one, %kernel_ele, %c0_f32 : f32
                  scf.if %sparsity_flag {
                    // Check tail.
                    %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<32xf32>
                    %output_col_cur = arith.muli %output_col_idx, %c32 : index
                    %tail_len = arith.subi %output_col, %output_col_cur : index
                    %tail_flag = arith.cmpi sge, %tail_len, %c32 : index
                    scf.if %tail_flag {
                      %input_vec = affine.vector_load %input[%n_idx, %c_idx, %output_row_idx + %kernel_row_idx, %kernel_col_idx + %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
                      %output_vec = affine.vector_load %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
                      %result_vec = vector.fma %input_vec, %kernel_vec, %output_vec : vector<32xf32>
                      affine.vector_store %result_vec, %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx * 32] : memref<?x?x?x?xf32>, vector<32xf32>
                    } else {
                      %mask_vec = vector.create_mask %tail_len : vector<32xi1>
                      %input_row_idx_tail = arith.addi %output_row_idx, %kernel_row_idx : index
                      %output_col_idx_tail = arith.muli %output_col_idx, %c32 : index
                      %input_col_idx_tail = arith.addi %kernel_col_idx, %output_col_idx_tail : index
                      %input_vec_tail = vector.maskedload %input[%n_idx, %c_idx, %input_row_idx_tail, %input_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                      %output_vec_tail = vector.maskedload %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %c0_f32_vec : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                      %result_vec_tail = vector.fma %input_vec_tail, %kernel_vec, %output_vec_tail : vector<32xf32>
                      vector.maskedstore %output[%n_idx, %f_idx, %output_row_idx, %output_col_idx_tail], %mask_vec, %result_vec_tail : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }

  func.func @main() {
    // Intput and kernel value.
    %cst = arith.constant 1.000000e+00 : f32
    // Output value.
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Define layout.
    %input_n = arith.constant 1 : index
    %input_c = arith.constant 64 : index
    %input_h = arith.constant 58 : index
    %input_w = arith.constant 58 : index

    %kernel_f = arith.constant 64 : index
    %kernel_c = arith.constant 64 : index
    %kernel_h = arith.constant 3 : index
    %kernel_w = arith.constant 3 : index

    %output_n = arith.constant 1 : index
    %output_f = arith.constant 64 : index
    %output_h = arith.constant 56 : index
    %output_w = arith.constant 56 : index

    // Define input, kernel, and output memref.
    %input = call @alloc_f32(%input_n, %input_c, %input_h, %input_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %kernel = call @alloc_f32(%kernel_f, %kernel_c, %kernel_h, %kernel_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %output = call @alloc_f32(%output_n, %output_f, %output_h, %output_w, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    // Perform convolution
    call @conv_2d_nchw_fchw(%input, %kernel, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

    // Print the output
    %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

    memref.dealloc %output : memref<?x?x?x?xf32>
    memref.dealloc %input : memref<?x?x?x?xf32>
    memref.dealloc %kernel : memref<?x?x?x?xf32>
    return
  }
}
