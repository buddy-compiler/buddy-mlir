// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  func.func @blis_sgemm_vectorized(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %m = memref.dim %a, %c0 : memref<?x?xf32>
    %n = memref.dim %c, %c1 : memref<?x?xf32>
    %k = memref.dim %a, %c1 : memref<?x?xf32>

    // BLIS blocking parameters with vectorization-friendly sizes
    %nc = arith.constant 256 : index
    %kc = arith.constant 128 : index
    %mc = arith.constant 64 : index
    %mr = arith.constant 8 : index  // 8 rows unrolled
    %nr = arith.constant 32 : index // 32 columns for vector<32xf32>

    // BLIS 5-loop structure
    scf.parallel (%jc) = (%c0) to (%n) step (%nc) {
      %jc_end = arith.addi %jc, %nc : index
      %jc_bound = arith.cmpi slt, %jc_end, %n : index
      %jc_actual_end = arith.select %jc_bound, %jc_end, %n : index
      %nc_actual = arith.subi %jc_actual_end, %jc : index

      affine.for %pc = 0 to %k step 128 {
        %pc_end = arith.addi %pc, %kc : index
        %pc_bound = arith.cmpi slt, %pc_end, %k : index
        %pc_actual_end = arith.select %pc_bound, %pc_end, %k : index
        %kc_actual = arith.subi %pc_actual_end, %pc : index

        // Pack B 
        %B_packed = memref.alloc(%kc_actual, %nc_actual) : memref<?x?xf32>

        scf.for %kp = %c0 to %kc_actual step %c1 {
          scf.for %j = %c0 to %nc_actual step %c1 {
            %b_row_idx = arith.addi %pc, %kp : index
            %b_col_idx = arith.addi %jc, %j : index
            %b_val = memref.load %b[%b_row_idx, %b_col_idx] : memref<?x?xf32>
            memref.store %b_val, %B_packed[%kp, %j] : memref<?x?xf32>
          }
        }

        scf.parallel (%ic) = (%c0) to (%m) step (%mc) {
          %ic_end = arith.addi %ic, %mc : index
          %ic_bound = arith.cmpi slt, %ic_end, %m : index
          %ic_actual_end = arith.select %ic_bound, %ic_end, %m : index
          %mc_actual = arith.subi %ic_actual_end, %ic : index

          // Pack A
          %A_packed = memref.alloc(%mc_actual, %kc_actual) : memref<?x?xf32>
          
          scf.for %i = %c0 to %mc_actual step %c1 {
            scf.for %kp = %c0 to %kc_actual step %c1 {
              %a_row_idx = arith.addi %ic, %i : index 
              %a_col_idx = arith.addi %pc, %kp : index
              %a_val = memref.load %a[%a_row_idx, %a_col_idx] : memref<?x?xf32>
              memref.store %a_val, %A_packed[%i, %kp] : memref<?x?xf32>
            }
          }

          scf.for %jr = %c0 to %nc_actual step %nr {
            %jr_end = arith.addi %jr, %nr : index
            %jr_bound = arith.cmpi slt, %jr_end, %nc_actual : index
            %jr_actual_end = arith.select %jr_bound, %jr_end, %nc_actual : index
            %nr_actual = arith.subi %jr_actual_end, %jr : index

            scf.for %n_idx = %c0 to %nr_actual step %nr {
              %n_idx_end = arith.addi %n_idx, %nr : index
              %n_idx_bound = arith.cmpi slt, %n_idx_end, %nr_actual : index
              %n_idx_actual_end = arith.select %n_idx_bound, %n_idx_end, %nr_actual : index
              %cols_to_process = arith.subi %n_idx_actual_end, %n_idx : index
              
              // if 32 columns?
              %can_vectorize = arith.cmpi sge, %cols_to_process, %nr : index
              scf.if %can_vectorize {
                // 32 columns  vectorize
                scf.for %ir = %c0 to %mc_actual step %mr {
                  %ir_end = arith.addi %ir, %mr : index
                  %ir_bound = arith.cmpi slt, %ir_end, %mc_actual : index
                  %ir_actual_end = arith.select %ir_bound, %ir_end, %mc_actual : index
                  %mr_actual = arith.subi %ir_actual_end, %ir : index
                  
                  // if 8 rows?
                  %has_full_rows = arith.cmpi sge, %mr_actual, %c8 : index
                  scf.if %has_full_rows {
                    // 8 rows vectorize
                    %ir_0 = arith.addi %ir, %c0 : index
                    %ir_1 = arith.addi %ir, %c1 : index
                    %ir_2 = arith.addi %ir, %c2 : index
                    %ir_3 = arith.addi %ir, %c3 : index
                    %ir_4 = arith.addi %ir, %c4 : index
                    %ir_5 = arith.addi %ir, %c5 : index
                    %ir_6 = arith.addi %ir, %c6 : index
                    %ir_7 = arith.addi %ir, %c7 : index

                    %sum_init = arith.constant dense<0.> : vector<32xf32>
                    %sum_iter_vec_0, %sum_iter_vec_1, %sum_iter_vec_2, %sum_iter_vec_3,
                    %sum_iter_vec_4, %sum_iter_vec_5, %sum_iter_vec_6, %sum_iter_vec_7
                        = scf.for %k_inner = %c0 to %kc_actual step %c1
                        iter_args(%sum_vec_0 = %sum_init,
                                  %sum_vec_1 = %sum_init,
                                  %sum_vec_2 = %sum_init,
                                  %sum_vec_3 = %sum_init,
                                  %sum_vec_4 = %sum_init,
                                  %sum_vec_5 = %sum_init,
                                  %sum_vec_6 = %sum_init,
                                  %sum_vec_7 = %sum_init
                                  )
                        -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>,
                            vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>) {
                      // Load A values for 8 rows
                      %a_val_0 = memref.load %A_packed[%ir_0, %k_inner] : memref<?x?xf32>
                      %a_val_1 = memref.load %A_packed[%ir_1, %k_inner] : memref<?x?xf32>
                      %a_val_2 = memref.load %A_packed[%ir_2, %k_inner] : memref<?x?xf32>
                      %a_val_3 = memref.load %A_packed[%ir_3, %k_inner] : memref<?x?xf32>
                      %a_val_4 = memref.load %A_packed[%ir_4, %k_inner] : memref<?x?xf32>
                      %a_val_5 = memref.load %A_packed[%ir_5, %k_inner] : memref<?x?xf32>
                      %a_val_6 = memref.load %A_packed[%ir_6, %k_inner] : memref<?x?xf32>
                      %a_val_7 = memref.load %A_packed[%ir_7, %k_inner] : memref<?x?xf32>

                      // Broadcast A values to vectors
                      %a_vec_0 = vector.broadcast %a_val_0 : f32 to vector<32xf32>
                      %a_vec_1 = vector.broadcast %a_val_1 : f32 to vector<32xf32>
                      %a_vec_2 = vector.broadcast %a_val_2 : f32 to vector<32xf32>
                      %a_vec_3 = vector.broadcast %a_val_3 : f32 to vector<32xf32>
                      %a_vec_4 = vector.broadcast %a_val_4 : f32 to vector<32xf32>
                      %a_vec_5 = vector.broadcast %a_val_5 : f32 to vector<32xf32>
                      %a_vec_6 = vector.broadcast %a_val_6 : f32 to vector<32xf32>
                      %a_vec_7 = vector.broadcast %a_val_7 : f32 to vector<32xf32>

                      // Load B vector (32 columns)
                      %b_vec = vector.load %B_packed[%k_inner, %n_idx] : memref<?x?xf32>, vector<32xf32>

                      // Fused multiply-add
                      %res_sum_vec_0 = vector.fma %a_vec_0, %b_vec, %sum_vec_0 : vector<32xf32>
                      %res_sum_vec_1 = vector.fma %a_vec_1, %b_vec, %sum_vec_1 : vector<32xf32>
                      %res_sum_vec_2 = vector.fma %a_vec_2, %b_vec, %sum_vec_2 : vector<32xf32>
                      %res_sum_vec_3 = vector.fma %a_vec_3, %b_vec, %sum_vec_3 : vector<32xf32>
                      %res_sum_vec_4 = vector.fma %a_vec_4, %b_vec, %sum_vec_4 : vector<32xf32>
                      %res_sum_vec_5 = vector.fma %a_vec_5, %b_vec, %sum_vec_5 : vector<32xf32>
                      %res_sum_vec_6 = vector.fma %a_vec_6, %b_vec, %sum_vec_6 : vector<32xf32>
                      %res_sum_vec_7 = vector.fma %a_vec_7, %b_vec, %sum_vec_7 : vector<32xf32>

                      scf.yield %res_sum_vec_0, %res_sum_vec_1, %res_sum_vec_2, %res_sum_vec_3,
                                  %res_sum_vec_4, %res_sum_vec_5, %res_sum_vec_6, %res_sum_vec_7
                          : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>,
                          vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
                    }

                    // Store results back to C
                    %c_row_0 = arith.addi %ic, %ir_0 : index
                    %c_row_1 = arith.addi %ic, %ir_1 : index
                    %c_row_2 = arith.addi %ic, %ir_2 : index
                    %c_row_3 = arith.addi %ic, %ir_3 : index
                    %c_row_4 = arith.addi %ic, %ir_4 : index
                    %c_row_5 = arith.addi %ic, %ir_5 : index
                    %c_row_6 = arith.addi %ic, %ir_6 : index
                    %c_row_7 = arith.addi %ic, %ir_7 : index
                    %c_col_idx = arith.addi %jc, %jr : index
                    %c_col_actual = arith.addi %c_col_idx, %n_idx : index

                    %c_vec_0 = vector.load %c[%c_row_0, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_1 = vector.load %c[%c_row_1, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_2 = vector.load %c[%c_row_2, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_3 = vector.load %c[%c_row_3, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_4 = vector.load %c[%c_row_4, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_5 = vector.load %c[%c_row_5, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_6 = vector.load %c[%c_row_6, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    %c_vec_7 = vector.load %c[%c_row_7, %c_col_actual] : memref<?x?xf32>, vector<32xf32>

                    %final_vec_0 = arith.addf %c_vec_0, %sum_iter_vec_0 : vector<32xf32>
                    %final_vec_1 = arith.addf %c_vec_1, %sum_iter_vec_1 : vector<32xf32>
                    %final_vec_2 = arith.addf %c_vec_2, %sum_iter_vec_2 : vector<32xf32>
                    %final_vec_3 = arith.addf %c_vec_3, %sum_iter_vec_3 : vector<32xf32>
                    %final_vec_4 = arith.addf %c_vec_4, %sum_iter_vec_4 : vector<32xf32>
                    %final_vec_5 = arith.addf %c_vec_5, %sum_iter_vec_5 : vector<32xf32>
                    %final_vec_6 = arith.addf %c_vec_6, %sum_iter_vec_6 : vector<32xf32>
                    %final_vec_7 = arith.addf %c_vec_7, %sum_iter_vec_7 : vector<32xf32>

                    vector.store %final_vec_0, %c[%c_row_0, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_1, %c[%c_row_1, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_2, %c[%c_row_2, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_3, %c[%c_row_3, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_4, %c[%c_row_4, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_5, %c[%c_row_5, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_6, %c[%c_row_6, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                    vector.store %final_vec_7, %c[%c_row_7, %c_col_actual] : memref<?x?xf32>, vector<32xf32>
                  } else {
                    // No 8 rows,not vectorize
                      scf.for %jj = %c0 to %nr step %c1 {
                      scf.for %ii = %ir to %ir_actual_end step %c1 {
                        %sum_init = arith.constant 0.0 : f32
                        %sum_iter = scf.for %k_inner = %c0 to %kc_actual step %c1
                            iter_args(%sum = %sum_init) -> (f32) {
                          %a_val = memref.load %A_packed[%ii, %k_inner] : memref<?x?xf32>
                          %b_index=arith.addi %n_idx,%jj : index
                          %b_val = memref.load %B_packed[%k_inner, %b_index] : memref<?x?xf32>
                          %prod = arith.mulf %a_val, %b_val : f32
                          %new_sum = arith.addf %sum, %prod : f32
                          scf.yield %new_sum : f32
                        }
                        %c_row_idx = arith.addi %ic, %ii : index
                        %c_col_idx = arith.addi %jc, %jr : index
                        %c_index=arith.addi %n_idx,%jj : index
                        %c_col_actual = arith.addi %c_col_idx, %c_index : index
                        %current_val = memref.load %c[%c_row_idx, %c_col_actual] : memref<?x?xf32>
                        %final_sum = arith.addf %current_val, %sum_iter : f32
                        memref.store %final_sum, %c[%c_row_idx, %c_col_actual] : memref<?x?xf32>
                      }
                    }
                  }
                }
              } else {
                // No 32 columns,not vectorize
                scf.for %n_idx_tail = %n_idx to %n_idx_actual_end step %c1 {
                  scf.for %ir = %c0 to %mc_actual step %mr {
                    %ir_end = arith.addi %ir, %mr : index
                    %ir_bound = arith.cmpi slt, %ir_end, %mc_actual : index
                    %ir_actual_end = arith.select %ir_bound, %ir_end, %mc_actual : index

                    scf.for %ii = %ir to %ir_actual_end step %c1 {
                      %sum_init = arith.constant 0.0 : f32
                      %sum_iter = scf.for %k_inner = %c0 to %kc_actual step %c1
                          iter_args(%sum = %sum_init) -> (f32) {
                        %a_val = memref.load %A_packed[%ii, %k_inner] : memref<?x?xf32>
                        %b_val = memref.load %B_packed[%k_inner, %n_idx_tail] : memref<?x?xf32>
                        %prod = arith.mulf %a_val, %b_val : f32
                        %new_sum = arith.addf %sum, %prod : f32
                        scf.yield %new_sum : f32
                      }
                      
                      %c_row_idx = arith.addi %ic, %ii : index
                      %c_col_idx = arith.addi %jc, %jr : index
                      %c_col_actual = arith.addi %c_col_idx, %n_idx_tail : index
                      %current_val = memref.load %c[%c_row_idx, %c_col_actual] : memref<?x?xf32>
                      %final_sum = arith.addf %current_val, %sum_iter : f32
                      memref.store %final_sum, %c[%c_row_idx, %c_col_actual] : memref<?x?xf32>
                    }
                  }
                }
              }
            }
          }
          memref.dealloc %A_packed : memref<?x?xf32>
        }
        memref.dealloc %B_packed : memref<?x?xf32>
      }
    }
    return
  }

   func.func @main(){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // little martix test data(for correct)
    %cM = arith.constant 37 : index  //37%8=5
    %cN = arith.constant 43 : index  //43%16=11
    %cK = arith.constant 1123 : index  

    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    %c0_f32 = arith.constant 0.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C_vec = memref.alloc(%cM, %cN) : memref<?x?xf32>
    %C_ref = memref.alloc(%cM, %cN) : memref<?x?xf32>

    // init
    linalg.fill ins(%cf1 : f32) outs(%A:memref<?x?xf32>)
    linalg.fill ins(%cf2 : f32) outs(%B:memref<?x?xf32>)
    linalg.fill ins(%c0_f32 : f32) outs(%C_vec:memref<?x?xf32>)
    linalg.fill ins(%c0_f32: f32) outs(%C_ref:memref<?x?xf32>)

    linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                outs(%C_ref: memref<?x?xf32>)
    //%print_C = memref.cast %C_ref : memref<?x?xf32> to memref<*xf32>
    //call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    call @blis_sgemm_vectorized(%A, %B, %C_vec) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    //%print_V = memref.cast %C_vec : memref<?x?xf32> to memref<*xf32>
    //call @printMemrefF32(%print_V) : (memref<*xf32>) -> ()
    // ==================== Verification ====================
    %tolerance = arith.constant 1.0e-5 : f32
    %init_correct = arith.constant 1.0 : f32

    %all_vec_correct = scf.for %i = %c0 to %cM step %c1 iter_args(%vec_correct_so_far = %init_correct) -> (f32) {
      %row_vec_correct = scf.for %j = %c0 to %cN step %c1 iter_args(%row_correct_so_far = %vec_correct_so_far) -> (f32) {
        %ref_val = memref.load %C_ref[%i, %j] : memref<?x?xf32>
        %vec_val = memref.load %C_vec[%i, %j] : memref<?x?xf32>
        %diff_vec = arith.subf %ref_val, %vec_val : f32
        %abs_diff_vec = math.absf %diff_vec : f32
        %vec_correct = arith.cmpf olt, %abs_diff_vec, %tolerance : f32
        %vec_correct_f32 = arith.uitofp %vec_correct : i1 to f32
        %new_row_correct = arith.mulf %row_correct_so_far, %vec_correct_f32 : f32
        scf.yield %new_row_correct : f32
      }
      scf.yield %row_vec_correct : f32
    }
    vector.print %all_vec_correct : f32

    // CHECK: 1

    // ==================== Performance ====================
    %cM_large = arith.constant 1024 : index
    %cN_large = arith.constant 1536 : index
    %cK_large = arith.constant 8960 : index

    %A_large = memref.alloc(%cM_large, %cK_large) : memref<?x?xf32>
    %B_large = memref.alloc(%cK_large, %cN_large) : memref<?x?xf32>
    %C_vec_large = memref.alloc(%cM_large, %cN_large) : memref<?x?xf32>
    %C_ref_large = memref.alloc(%cM_large, %cN_large) : memref<?x?xf32>
    
    
    linalg.fill ins(%cf1 : f32) outs(%A_large:memref<?x?xf32>)
    linalg.fill ins(%cf2 : f32) outs(%B_large:memref<?x?xf32>)
    linalg.fill ins(%c0_f32 : f32) outs(%C_vec_large:memref<?x?xf32>)
    linalg.fill ins(%c0_f32 : f32) outs(%C_ref_large:memref<?x?xf32>)

    //Standard version
    %t_start_ref = call @rtclock() : () -> f64
    linalg.matmul ins(%A_large, %B_large: memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C_ref_large: memref<?x?xf32>)
    %t_end_ref = call @rtclock() : () -> f64
    %time_ref = arith.subf %t_end_ref, %t_start_ref : f64

    // BLIS version
    %t_start_vec = call @rtclock() : () -> f64
    call @blis_sgemm_vectorized(%A_large, %B_large, %C_vec_large) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    %t_end_vec = call @rtclock() : () -> f64
    %time_vec = arith.subf %t_end_vec, %t_start_vec : f64

    vector.print %time_ref : f64
    vector.print %time_vec : f64

    // CHECK: {{[0-9]+\.[0-9]+}}
    // CHECK: {{[0-9]+\.[0-9]+}}

    memref.dealloc %C_vec_large : memref<?x?xf32>
    memref.dealloc %B_large : memref<?x?xf32>
    memref.dealloc %A_large : memref<?x?xf32>

    memref.dealloc %C_ref : memref<?x?xf32>
    memref.dealloc %C_vec : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>

    return
  }
}
