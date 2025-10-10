// RUN: buddy-opt  %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  // BLIS-style SGEMM implementation based on "Anatomy of High-Performance Many-Threaded Matrix Multiplication"
  // Key BLIS innovations implemented:
  // 1. Data packing strategy for cache optimization
  // 2. True micro-kernel operating on packed data
  // 3. five-level loop nesting with proper data flow
  // 4. Cache hierarchy-aware blocking parameters
  func.func @sgemm_handwritten(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    %t_start = call @rtclock() : () -> f64

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    // BLIS Framework: Cache-aware blocking parameters
    // Level 1: L3 cache blocking (main memory to L3 cache)
    %nc = arith.constant 4096 : index  // Block columns of C and B
    
    // Level 2: L2 cache blocking (L3 cache to L2 cache)  
    %kc = arith.constant 256 : index   // Block panels of A and B
    %mc = arith.constant 256 : index   // Block rows of A and C
    
    // Level 3: Register blocking (L1 cache to registers)
    %nr = arith.constant 32 : index     // Register blocking for columns
    %mr = arith.constant 8 : index      // Register blocking for rows

    // Matrix dimensions
    %m = memref.dim %A, %c0 : memref<?x?xf32>
    %n = memref.dim %C, %c1 : memref<?x?xf32>
    %k = memref.dim %A, %c1 : memref<?x?xf32>

    // BLIS Loop Nesting Structure: jc -> pc -> ic [micro-kernel: jr × ir]
    // Level 1: jc loop - Block columns of C and B (L3 cache blocking)
    scf.for %jc = %c0 to %n step %nc {
      %n_remaining = arith.subi %n, %jc : index
      %nc_actual = arith.minsi %nc, %n_remaining : index
      
      // Pack B block (k × nc_actual) into contiguous memory - ONE TIME PACKING
      %packed_B = memref.alloc(%k, %nc_actual) : memref<?x?xf32>
      scf.for %k_pack = %c0 to %k step %c1 {
        scf.for %j_pack = %c0 to %nc_actual step %c1 {
          %b_j = arith.addi %jc, %j_pack : index
          %b_val = memref.load %B[%k_pack, %b_j] : memref<?x?xf32>
          memref.store %b_val, %packed_B[%k_pack, %j_pack] : memref<?x?xf32>
        }
      }
      
      // Level 2: pc loop - Block panels of A and B (L2 cache blocking)
      scf.for %pc = %c0 to %k step %kc {
        %k_remaining = arith.subi %k, %pc : index
        %kc_actual = arith.minsi %kc, %k_remaining : index
        
        // Pack A block (m × kc_actual) into contiguous memory - ONE TIME PACKING
        %packed_A = memref.alloc(%m, %kc_actual) : memref<?x?xf32>
        scf.for %i_pack = %c0 to %m step %c1 {
          scf.for %k_pack = %c0 to %kc_actual step %c1 {
            %a_k = arith.addi %pc, %k_pack : index
            %a_val = memref.load %A[%i_pack, %a_k] : memref<?x?xf32>
            memref.store %a_val, %packed_A[%i_pack, %k_pack] : memref<?x?xf32>
          }
        }
        
        // Level 3: ic loop - Block rows of A and C (L2 cache blocking)
        scf.for %ic = %c0 to %m step %mc {
          %m_remaining = arith.subi %m, %ic : index
          %mc_actual = arith.minsi %mc, %m_remaining : index
          
          // BLIS Micro-Kernel: jr × ir loops operating on packed data
          // This is the true micro-kernel that operates on contiguous memory
          scf.for %jr = %c0 to %nc_actual step %nr {
            %n_remaining_inner = arith.subi %nc_actual, %jr : index
            %nr_actual = arith.minsi %nr, %n_remaining_inner : index
            
            scf.for %ir = %c0 to %mc_actual step %mr {
              %m_remaining_inner = arith.subi %mc_actual, %ir : index
              %mr_actual = arith.minsi %mr, %m_remaining_inner : index
              
              // True BLIS Micro-Kernel: Vectorized computation on packed data
              // This operates on contiguous memory blocks for optimal cache performance
              %sum_init = arith.constant dense<0.> : vector<32xf32>
              
              // Micro-kernel computation: mr_actual rows × nr_actual columns
              %micro_result_0, %micro_result_1, %micro_result_2, %micro_result_3,
              %micro_result_4, %micro_result_5, %micro_result_6, %micro_result_7
                  = scf.for %kr = %c0 to %kc_actual step %c1
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
                
                // Load from packed A (contiguous memory access)
                %a_row_0 = memref.load %packed_A[%ir, %kr] : memref<?x?xf32>
                %a_row_1 = memref.load %packed_A[%ir + %c1, %kr] : memref<?x?xf32>
                %a_row_2 = memref.load %packed_A[%ir + %c2, %kr] : memref<?x?xf32>
                %a_row_3 = memref.load %packed_A[%ir + %c3, %kr] : memref<?x?xf32>
                %a_row_4 = memref.load %packed_A[%ir + %c4, %kr] : memref<?x?xf32>
                %a_row_5 = memref.load %packed_A[%ir + %c5, %kr] : memref<?x?xf32>
                %a_row_6 = memref.load %packed_A[%ir + %c6, %kr] : memref<?x?xf32>
                %a_row_7 = memref.load %packed_A[%ir + %c7, %kr] : memref<?x?xf32>
                
                // Broadcast A elements
                %a_vec_0 = vector.broadcast %a_row_0 : f32 to vector<32xf32>
                %a_vec_1 = vector.broadcast %a_row_1 : f32 to vector<32xf32>
                %a_vec_2 = vector.broadcast %a_row_2 : f32 to vector<32xf32>
                %a_vec_3 = vector.broadcast %a_row_3 : f32 to vector<32xf32>
                %a_vec_4 = vector.broadcast %a_row_4 : f32 to vector<32xf32>
                %a_vec_5 = vector.broadcast %a_row_5 : f32 to vector<32xf32>
                %a_vec_6 = vector.broadcast %a_row_6 : f32 to vector<32xf32>
                %a_vec_7 = vector.broadcast %a_row_7 : f32 to vector<32xf32>
                
                // Load from packed B (contiguous memory access)
                %b_vec = vector.load %packed_B[%kr, %jr] : memref<?x?xf32>, vector<32xf32>
                
                // Fused multiply-add operations
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
              
              // Store results back to C with proper indexing
              %c_j = arith.addi %jc, %jr : index
              %c_i_0 = arith.addi %ic, %ir : index
              %c_i_1 = arith.addi %c_i_0, %c1 : index
              %c_i_2 = arith.addi %c_i_0, %c2 : index
              %c_i_3 = arith.addi %c_i_0, %c3 : index
              %c_i_4 = arith.addi %c_i_0, %c4 : index
              %c_i_5 = arith.addi %c_i_0, %c5 : index
              %c_i_6 = arith.addi %c_i_0, %c6 : index
              %c_i_7 = arith.addi %c_i_0, %c7 : index
              
              vector.store %micro_result_0, %C[%c_i_0, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_1, %C[%c_i_1, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_2, %C[%c_i_2, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_3, %C[%c_i_3, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_4, %C[%c_i_4, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_5, %C[%c_i_5, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_6, %C[%c_i_6, %c_j] : memref<?x?xf32>, vector<32xf32>
              vector.store %micro_result_7, %C[%c_i_7, %c_j] : memref<?x?xf32>, vector<32xf32>
            }
          }
          
          // Clean up packed A buffer
          memref.dealloc %packed_A : memref<?x?xf32>
        }
      }
      
      // Clean up packed B buffer
      memref.dealloc %packed_B : memref<?x?xf32>
    }

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 1024 : index
    %cN = arith.constant 1536 : index
    %cK = arith.constant 8960 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    %c0 = arith.constant 0.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<?x?xf32>)

    linalg.fill
    ins(%cf2 : f32)
    outs(%B:memref<?x?xf32>)

    linalg.fill
    ins(%c0 : f32)
    outs(%C:memref<?x?xf32>)

    call @sgemm_handwritten(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
