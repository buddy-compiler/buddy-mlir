#map = affine_map<(d0) -> (d0 floordiv 9)>
#map1 = affine_map<(d0, d1) -> (d0 floordiv 56 + (d1 mod 9) floordiv 3)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 56) * 56 - (d1 floordiv 3) * 3)>
#map3 = affine_map<(d0) -> (d0 ceildiv 32)>

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

// Optimize GEMM with broadcast method.
func.func @batch_matmul_optimize(%a: memref<1x64x576xf32>, %b: memref<1x576x3136xf32>, %c: memref<1x64x3136xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %step = arith.constant 32 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_f32_vec = vector.splat %c0_f32 : vector<32xf32>

  %a_row = memref.dim %a, %c1 : memref<1x64x576xf32>
  %a_col = memref.dim %a, %c2 : memref<1x64x576xf32>
  %b_row = memref.dim %b, %c1 : memref<1x576x3136xf32>
  %b_col = memref.dim %b, %c2 : memref<1x576x3136xf32>

  affine.for %b_row_idx = 0 to %b_row {
    affine.for %a_row_idx = 0 to %a_row {
      affine.for %b_col_idx = 0 to #map3(%b_col) {
        %a_ele = memref.load %a[%c0, %a_row_idx, %b_row_idx] : memref<1x64x576xf32>
        %a_vec = vector.broadcast %a_ele : f32 to vector<32xf32>
        %b_col_cur = arith.muli %b_col_idx, %step : index
        %tail_len = arith.subi %b_col, %b_col_cur : index
        %tail_flag = arith.cmpi sge, %tail_len, %step : index
        scf.if %tail_flag {
          %b_vec = affine.vector_load %b[%c0, %b_row_idx, %b_col_idx * 32] : memref<1x576x3136xf32>, vector<32xf32>
          %c_vec = affine.vector_load %c[%c0, %a_row_idx, %b_col_idx * 32] : memref<1x64x3136xf32>, vector<32xf32>
          %result_vec = vector.fma %a_vec, %b_vec, %c_vec : vector<32xf32>
          affine.vector_store %result_vec, %c[%c0, %a_row_idx, %b_col_idx * 32] : memref<1x64x3136xf32>, vector<32xf32>
        } else {
          %mask_vec = vector.create_mask %tail_len : vector<32xi1>
          %b_col_idx_tail = arith.muli %b_col_idx, %step : index
          %b_vec_tail = vector.maskedload %b[%c0, %b_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<1x576x3136xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %c_vec_tail = vector.maskedload %c[%c0, %a_row_idx, %b_col_idx_tail], %mask_vec, %c0_f32_vec : memref<1x64x3136xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %result_vec_tail = vector.fma %a_vec, %b_vec_tail, %c_vec_tail : vector<32xf32>
          vector.maskedstore %c[%c0, %a_row_idx, %b_col_idx_tail], %mask_vec, %result_vec_tail : memref<1x64x3136xf32>, vector<32xi1>, vector<32xf32>
        }
      }
    }
  }
  return
}

func.func @conv2d_nchw_fchw_im2col(%input: memref<?x?x?x?xf32>, %kernel: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  %input_specific = memref.cast %input : memref<?x?x?x?xf32> to memref<1x64x58x58xf32>
  %kernel_specific = memref.cast %kernel : memref<?x?x?x?xf32> to memref<64x64x3x3xf32>
  %output_specific = memref.cast %output : memref<?x?x?x?xf32> to memref<1x64x56x56xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index // 576 = 64 * 3 * 3 = kernel's c*h*w
  %c3136 = arith.constant 3136 : index // 3136 = 56 * 56 = output's h*w
  %c64 = arith.constant 64 : index
  %kernel_collapse = memref.collapse_shape %kernel_specific [[0], [1, 2, 3]] : memref<64x64x3x3xf32> into memref<64x576xf32>
  %output_collapse = memref.collapse_shape %output_specific [[0], [1], [2, 3]] : memref<1x64x56x56xf32> into memref<1x64x3136xf32>
  %input_collapse = memref.alloc() {alignment = 64 : i64} : memref<1x576x3136xf32>
  // Apply im2col.
  scf.for %idx0 = %c0 to %c1 step %c1 {
    scf.for %idx1 = %c0 to %c576 step %c1 {
      scf.for %idx2 = %c0 to %c3136 step %c1 {
        %0 = affine.apply #map(%idx1)
        %1 = affine.apply #map1(%idx2, %idx1)
        %2 = affine.apply #map2(%idx2, %idx1)
        %3 = memref.load %input_specific[%idx0, %0, %1, %2] : memref<1x64x58x58xf32>
        memref.store %3, %input_collapse[%idx0, %idx1, %idx2] : memref<1x576x3136xf32>
      }
    }
  }
  // Implement optimized GEMM.
  %kernel_expand = memref.expand_shape %kernel_collapse [[0, 1], [2]] : memref<64x576xf32> into memref<1x64x576xf32>
  func.call @batch_matmul_optimize(%kernel_expand, %input_collapse, %output_collapse) : (memref<1x64x576xf32>, memref<1x576x3136xf32>, memref<1x64x3136xf32>) -> ()
  // Apply col2im.
  %result_mem = memref.expand_shape %output_collapse [[0], [1], [2, 3]] : memref<1x64x3136xf32> into memref<1x64x56x56xf32>
  memref.dealloc %input_collapse : memref<1x576x3136xf32>
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

  %kernel_n = arith.constant 64 : index
  %kernel_c = arith.constant 64 : index
  %kernel_h = arith.constant 3 : index
  %kernel_w = arith.constant 3 : index

  %output_n = arith.constant 1 : index
  %output_c = arith.constant 64 : index
  %output_h = arith.constant 56 : index
  %output_w = arith.constant 56 : index

  // Define input, kernel, and output memref.
  %input = call @alloc_f32(%input_n, %input_c, %input_h, %input_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %kernel = call @alloc_f32(%kernel_n, %kernel_c, %kernel_h, %kernel_w, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %output = call @alloc_f32(%output_n, %output_c, %output_h, %output_w, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  // Perform convolution
  call @conv2d_nchw_fchw_im2col(%input, %kernel, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

  // Print the output
  %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

  memref.dealloc %output : memref<?x?x?x?xf32>
  memref.dealloc %input : memref<?x?x?x?xf32>
  memref.dealloc %kernel : memref<?x?x?x?xf32>
  return
}
