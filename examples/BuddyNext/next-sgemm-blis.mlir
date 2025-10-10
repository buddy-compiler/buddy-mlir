// RUN: buddy-opt  %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -expand-strided-metadata \
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
  
  // Cache blocking
  // L3 Cache Blocking
  func.func private @get_NC() -> index { %c1024 = arith.constant 1024 : index return %c1024 : index }
  // L2 Cache Blocking
  func.func private @get_MC() -> index { %c256 = arith.constant 256 : index return %c256 : index }
  // L1 Cache Blocking
  func.func private @get_KC() -> index { %c128 = arith.constant 128 : index return %c128 : index }
  
  // Register blocking
  func.func private @get_MR() -> index { %c4 = arith.constant 4 : index return %c4 : index }
  func.func private @get_NR() -> index { %c8 = arith.constant 8 : index return %c8 : index }

  // Micro Kernel: Calc C_block += A_sliver * B_sliver
  func.func @micro_kernel(%k_c: index, %a_sliver: memref<?xf32>, %b_sliver: memref<?xf32>, %c: memref<?x?xf32>, %i_start: index, %j_start: index, %n_dim: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %MR = func.call @get_MR() : () -> index
    %NR = func.call @get_NR() : () -> index

    scf.for %i = %c0 to %MR step %c1 {
      %c_i = arith.addi %i_start, %i : index
      %row_acc = memref.alloca(%NR) : memref<?xf32>

      // Initialize accumulators with current C values.
      scf.for %j = %c0 to %NR step %c1 {
        %c_j_init = arith.addi %j_start, %j : index
        %c_val_init = memref.load %c[%c_i, %c_j_init] : memref<?x?xf32>
        memref.store %c_val_init, %row_acc[%j] : memref<?xf32>
      }

      // Accumulate along the K dimension using the packed A/B panels.
      scf.for %l = %c0 to %k_c step %c1 {
        %a_idx_base = arith.muli %l, %MR : index
        %a_idx = arith.addi %a_idx_base, %i : index
        %a_val = memref.load %a_sliver[%a_idx] : memref<?xf32>

        %b_idx_base = arith.muli %l, %NR : index
        scf.for %j = %c0 to %NR step %c1 {
          %b_idx = arith.addi %b_idx_base, %j : index
          %b_val = memref.load %b_sliver[%b_idx] : memref<?xf32>
          %acc_val = memref.load %row_acc[%j] : memref<?xf32>
          %prod = arith.mulf %a_val, %b_val : f32
          %sum = arith.addf %acc_val, %prod : f32
          memref.store %sum, %row_acc[%j] : memref<?xf32>
        }
      }

      // Commit accumulators back to C.
      scf.for %j = %c0 to %NR step %c1 {
        %c_j = arith.addi %j_start, %j : index
        %acc_val_final = memref.load %row_acc[%j] : memref<?xf32>
        memref.store %acc_val_final, %c[%c_i, %c_j] : memref<?x?xf32>
      }
    }
    return
  }

  // Pack A matrix block (column-major)
  func.func @pack_a(%m_c: index, %k_c: index, %a: memref<?x?xf32>, %i_c: index, %p_c: index, %a_tilde: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %j = %c0 to %k_c step %c1 {
      scf.for %i = %c0 to %m_c step %c1 {
        %row_idx = arith.addi %i_c, %i : index
        %col_idx = arith.addi %p_c, %j : index
        %src_val = memref.load %a[%row_idx, %col_idx] : memref<?x?xf32>
        
        // Column-major packing: a_tilde[j * m_c + i]
        %dst_idx_base = arith.muli %j, %m_c : index
        %dst_idx = arith.addi %dst_idx_base, %i : index
        memref.store %src_val, %a_tilde[%dst_idx] : memref<?xf32>
      }
    }
    return
  }

  // Pack B matrix block (row-major)
  func.func @pack_b(%k_c: index, %n_c: index, %b: memref<?x?xf32>, %p_c: index, %j_c: index, %b_tilde: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %k_c step %c1 {
      scf.for %j = %c0 to %n_c step %c1 {
        %row_idx = arith.addi %p_c, %i : index
        %col_idx = arith.addi %j_c, %j : index
        %src_val = memref.load %b[%row_idx, %col_idx] : memref<?x?xf32>
        
        // Row-major packing: b_tilde[i * n_c + j]
        %dst_idx_base = arith.muli %i, %n_c : index
        %dst_idx = arith.addi %dst_idx_base, %j : index
        memref.store %src_val, %b_tilde[%dst_idx] : memref<?xf32>
      }
    }
    return
  }

  // Macro Kernel
  func.func @macro_kernel(%m_c: index, %n_c: index, %k_c: index, %c: memref<?x?xf32>, 
                         %i_c: index, %j_c: index, %n_dim: index, %a_tilde: memref<?xf32>, %b_tilde: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %MR = func.call @get_MR() : () -> index
    %NR = func.call @get_NR() : () -> index

    // Loop 2 (jr): Iterate B~'s columns
    scf.for %j_r = %c0 to %n_c step %NR {
      // Loop 1 (ir): Iterate A~'s rows
      scf.for %i_r = %c0 to %m_c step %MR {
        // Calculate starting indices in C
        %c_i_start = arith.addi %i_c, %i_r : index
        %c_j_start = arith.addi %j_c, %j_r : index
        
        // Calculate A sliver's starting position (column-major packed)
        %a_offset = arith.muli %i_r, %k_c : index
        %a_sliver = memref.subview %a_tilde[%a_offset] [%MR] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %a_sliver_cast = memref.cast %a_sliver : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
        
        // Calculate B sliver's starting position (row-major packed)
        %b_offset = arith.muli %j_r, %k_c : index
        %b_sliver = memref.subview %b_tilde[%b_offset] [%NR] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %b_sliver_cast = memref.cast %b_sliver : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
        
        func.call @micro_kernel(%k_c, %a_sliver_cast, %b_sliver_cast, %c, %c_i_start, %c_j_start, %n_dim) : 
          (index, memref<?xf32>, memref<?xf32>, memref<?x?xf32>, index, index, index) -> ()
      }
    }
    return
  }

  func.func @sgemm_blis_32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %t_start = func.call @rtclock() : () -> f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %N = memref.dim %c, %c1 : memref<?x?xf32>
    %K = memref.dim %a, %c1 : memref<?x?xf32>

    %NC = func.call @get_NC() : () -> index
    %MC = func.call @get_MC() : () -> index
    %KC = func.call @get_KC() : () -> index

    // Allocate packed buffers
    %mc_kc_size = arith.muli %MC, %KC : index
    %a_tilde = memref.alloc(%mc_kc_size) : memref<?xf32>
    
    %kc_nc_size = arith.muli %KC, %NC : index
    %b_tilde = memref.alloc(%kc_nc_size) : memref<?xf32>

    // Loop 5 (jc): Iterate over C and B's columns (N dimension L3 cache blocking)
    scf.for %j_c = %c0 to %N step %NC {
      %n_c_temp = arith.subi %N, %j_c : index
      %n_c = arith.minsi %n_c_temp, %NC : index

      // Loop 4 (pc): Iterate over A and B's shared K dimension (L3 cache blocking)
      scf.for %p_c = %c0 to %K step %KC {
        %k_c_temp = arith.subi %K, %p_c : index
        %k_c = arith.minsi %k_c_temp, %KC : index

        func.call @pack_b(%k_c, %n_c, %b, %p_c, %j_c, %b_tilde) : (index, index, memref<?x?xf32>, index, index, memref<?xf32>) -> ()

        // Loop 3 (ic): Iterate over C and A's rows (M dimension L2 cache blocking)
        scf.for %i_c = %c0 to %M step %MC {
          %m_c_temp = arith.subi %M, %i_c : index
          %m_c = arith.minsi %m_c_temp, %MC : index

          func.call @pack_a(%m_c, %k_c, %a, %i_c, %p_c, %a_tilde) : (index, index, memref<?x?xf32>, index, index, memref<?xf32>) -> ()

          func.call @macro_kernel(%m_c, %n_c, %k_c, %c, %i_c, %j_c, %N, %a_tilde, %b_tilde) : (index, index, index, memref<?x?xf32>, index, index, index, memref<?xf32>, memref<?xf32>) -> ()
        }
      }
    }

    memref.dealloc %a_tilde : memref<?xf32>
    memref.dealloc %b_tilde : memref<?xf32>

    %t_end = func.call @rtclock() : () -> f64
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

    call @sgemm_blis_32(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    %i = arith.constant 0 : index
    %j = arith.constant 0 : index
    %val = memref.load %C[%i, %j] : memref<?x?xf32>
    vector.print %val : f32

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
